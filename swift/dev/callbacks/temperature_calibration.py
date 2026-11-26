# Copyright (c) ModelScope Contributors. All rights reserved.
# Post-training temperature calibration + ECE reporting for decision (System-1) models.

from __future__ import annotations
import math
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from swift.dev.callbacks.base import CALLBACK, CallbackControl, CallbackState, TrainCallback
from swift.dev.model.decision_head import KIND_NAMES, masked_softmax
from swift.dev.utils import get_logger
from swift.dev.utils.plugin import PluginRegistry
from swift.dev.utils.scoring import brier_per_question, brier_score, collect, ece_score, nll_per_question

logger = get_logger()

MIN_CALIBRATION_ITEMS = 10
HEAD_TEMPERATURE_RANGE = (0.05, 20.0)
# Coarse-grid + golden-section parameters for the Brier path.
_GRID_LO, _GRID_HI, _GRID_STEPS = 0.05, 10.0, 50
_GOLDEN_RATIO = (5**0.5 - 1) / 2  # ~0.618
_GOLDEN_ITERS = 30

__all__ = ["TemperatureCalibrationCallback"]


def _clamp_temperature(t: float) -> float:
    """Clamp a temperature to a sensible range; a non-finite fit degrades to T=1 (uncalibrated).

    Python's max/min propagate NaN, so without this guard a diverged LBFGS fit would write NaN
    into the head's temperature buffers and persist it to scoring_head.json, poisoning every
    later forward.
    """
    if not math.isfinite(t):
        return 1.0
    return max(HEAD_TEMPERATURE_RANGE[0], min(HEAD_TEMPERATURE_RANGE[1], t))


def _fit_temperature_brier(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Find T that minimises the Brier score of ``softmax(logits / T)`` vs target."""

    def brier_at(T: float) -> float:
        return brier_score(masked_softmax(logits / T, mask), target, mask)

    lo, hi = _GRID_LO, _GRID_HI
    best_T, best_brier = 1.0, brier_at(1.0)
    for i in range(_GRID_STEPS + 1):
        T = lo + (hi - lo) * i / _GRID_STEPS
        b = brier_at(T)
        if b < best_brier:
            best_brier, best_T = b, T
    span = (hi - lo) / _GRID_STEPS
    a = max(HEAD_TEMPERATURE_RANGE[0], best_T - span)
    b = min(HEAD_TEMPERATURE_RANGE[1], best_T + span)
    for _ in range(_GOLDEN_ITERS):
        c = b - _GOLDEN_RATIO * (b - a)
        d = a + _GOLDEN_RATIO * (b - a)
        if brier_at(c) < brier_at(d):
            b = d
        else:
            a = c
    return _clamp_temperature((a + b) / 2)


def _fit_temperature_nll(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Find T that minimises NLL (cross-entropy) using LBFGS with strong_wolfe."""
    z = logits.float().clone()
    z[~mask] = -1e4
    tgt = target.float()
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        log_probs = torch.log_softmax(z / log_t.exp(), dim=-1)
        loss = -(tgt * log_probs).sum(dim=-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return _clamp_temperature(float(log_t.exp().item()))


def _fit_one_temperature(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_name: str) -> float:
    """Fit a single scalar T on (logits, target, mask): Brier (grid+golden) or NLL (LBFGS)."""
    if logits.shape[0] == 0:
        return 1.0
    if loss_name == "nll":
        t = _fit_temperature_nll(logits, target, mask)
    else:
        t = _fit_temperature_brier(logits, target, mask)
    if not math.isfinite(t):
        logger.warning(
            f"temperature_calibration: {loss_name} fit produced a non-finite temperature; keeping T=1.0 (uncalibrated)."
        )
    return _clamp_temperature(t)


def _stack_items(items: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a list of item dicts into padded [N, M] tensors."""
    max_opts = max(it["logits"].shape[0] for it in items)
    padded = []
    for key, pad_value in (("logits", 0.0), ("target", 0.0), ("mask", False)):
        rows = []
        for it in items:
            t = it[key]
            n = t.shape[0]
            rows.append(F.pad(t, (0, max_opts - n), value=pad_value) if n < max_opts else t)
        padded.append(torch.stack(rows))
    return tuple(padded)


@PluginRegistry.register(CALLBACK, "temperature_calibration")
class TemperatureCalibrationCallback(TrainCallback):
    """Two-stage (global head T + per-kind relative T) post-training calibration.

    The recipe binds the model (and optionally an eval batch iterator) after assembly; without a
    binding, or for heads without a temperature buffer, the callback no-ops. Reporting mirrors the
    legacy flow: a pre-training baseline on ``on_train_begin`` (report-only), then on
    ``on_train_end`` the fitted temperatures, per-kind Brier before/after, and 2-fold
    cross-validated accuracy / ECE / Brier / NLL.

    Mode-blind by design: temperature read/write goes through the ONE method face
    ``model.task_head_temperatures(...)`` -- a direct in-process call locally, a ray proxy
    call through to the workers, or the HTTP endpoint behind a remote handle; this callback
    cannot tell the modes apart. Served logits are always collected RAW (scaled back up by the
    temperature each row was served with), so the fitted temperatures are absolute and correct
    whatever the buffers held before.
    """

    def __init__(self, args=None, loss_name: str = "brier", min_items: Optional[int] = None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.loss_name = str(loss_name).lower().strip()
        if self.loss_name not in ("brier", "nll"):
            logger.warning(f'temperature_calibration: unknown loss_name={self.loss_name!r}, falling back to "brier".')
            self.loss_name = "brier"
        # Legacy parity: the env knob lets a test (or a small run) lower the minimum item count the
        # fitter demands; an explicit min_items argument still wins over the environment.
        if min_items is None:
            min_items = int(os.environ.get("SWIFT_CALIBRATION_MIN_ITEMS", MIN_CALIBRATION_ITEMS))
        self.min_items = int(min_items)
        self._model = None
        self._eval_batches: Optional[Callable[[], Iterable[Tuple[Dict, Any]]]] = None
        self._base_metrics: Optional[Dict[str, float]] = None
        self._base_items = 0

    def bind(self, model, eval_batches: Optional[Callable[[], Iterable[Tuple[Dict, Any]]]] = None) -> None:
        """Attach the model (for head discovery + buffer writes) and the eval batch provider.

        ``eval_batches`` yields ``(inputs, outputs)`` pairs already run in eval mode; the recipe's
        eval loop is the natural provider (outputs carry logits / option_mask / kinds / counts,
        inputs carry the padded target distribution as ``labels``).
        """
        self._model = model
        self._eval_batches = eval_batches

    def _served(self) -> Tuple[Optional[float], Optional[Dict[int, float]]]:
        """(global T or None, per-kind EFFECTIVE temperature) as currently served."""
        state = self._model.task_head_temperatures()['temperatures']
        if state is None:
            return None, None
        global_t, relative = state['global_temperature'], state['kind_temperatures']
        if len(relative) != len(KIND_NAMES) or not all(math.isfinite(t) and t > 0 for t in relative) or (
                global_t is not None and not (math.isfinite(global_t) and global_t > 0)):
            raise ValueError("The decision head returned invalid temperatures (need positive finite values).")
        served = {kind: (relative[kind] if global_t is None else global_t * relative[kind])
                  for kind in range(len(KIND_NAMES))}
        return global_t, served

    # -- events ----------------------------------------------------------------

    def on_train_begin(self, state: CallbackState, control: CallbackControl) -> None:
        """Report-only baseline: metrics with the loaded/factory temperatures, nothing fitted."""
        if self._eval_batches is None or self._model is None:
            return
        global_t, served = self._served()
        if served is None:
            return
        logger.info("temperature_calibration: evaluating base model on the eval set (baseline before training)...")
        try:
            collected = self._collect_eval(served)
        except Exception as e:
            logger.warning(f"temperature_calibration: failed to evaluate base model ({e}), skipping baseline.")
            return
        if not self._is_main_process() or not collected:
            return
        all_items = self._flatten_collected(collected)
        if not all_items:
            return
        # Logits are collected raw; dividing by the served temperature reproduces the
        # probabilities the loaded/factory buffers actually serve.
        self._base_metrics = self._metrics_at(all_items, lambda it: served[it["kind"]])
        self._base_items = len(all_items)
        logger.info(
            f"temperature_calibration: base metrics (pre-training, n={self._base_items}): "
            f"accuracy={self._base_metrics['accuracy']:.4f}, ece={self._base_metrics['ece']:.4f}, "
            f"brier={self._base_metrics['brier']:.4f}, nll={self._base_metrics['nll']:.4f}"
        )

    def on_train_end(self, state: CallbackState, control: CallbackControl) -> None:  # noqa: C901
        """Fit once and write the temperatures back; a failed run keeps its loaded/factory values."""
        if state.training_failed:
            logger.info("temperature_calibration: training failed; keeping the loaded/factory temperatures.")
            return
        if self._model is None:
            return
        global_t, served = self._served()
        if served is None:
            logger.info("temperature_calibration: head has no temperature buffer, skipping.")
            return
        if self._eval_batches is None:
            logger.info(
                "temperature_calibration: no eval batches bound (no eval dataset?), skipping; "
                "temperature stays at its loaded/factory value."
            )
            return

        logger.info(f"temperature_calibration: collecting logits on eval set (loss={self.loss_name})...")
        collected = self._collect_eval(served)
        if not collected:
            logger.info("temperature_calibration: no questions collected from eval set, skipping.")
            return
        # Every process fits the same collected CPU tensors and calls the same collective
        # write. Returning early on nonzero ranks would leave their buffers stale (or hang PP).

        all_items = self._flatten_collected(collected)
        if len(all_items) < self.min_items:
            logger.info(
                f"temperature_calibration: only {len(all_items)} questions (< {self.min_items}), skipping calibration."
            )
            return

        n_kinds = len(KIND_NAMES)

        # --- Two-stage fit on ALL data (production T), on RAW logits ---
        logits, target, mask = _stack_items(all_items)
        head_t = _fit_one_temperature(logits, target, mask, self.loss_name)
        relative_ts: Dict[int, float] = {}
        fitted_kinds: List[int] = []
        for kind in range(n_kinds):
            kind_items = [it for it in all_items if it["kind"] == kind]
            if len(kind_items) < self.min_items:
                relative_ts[kind] = 1.0
                logger.info(
                    f"temperature_calibration: kind {KIND_NAMES[kind]} has only {len(kind_items)} "
                    f"questions (< {self.min_items}); keeping relative T=1.0."
                )
                continue
            k_logits, k_target, k_mask = _stack_items(kind_items)
            rel_t = _fit_one_temperature(k_logits / max(head_t, 1e-8), k_target, k_mask, self.loss_name)
            relative_ts[kind] = rel_t
            fitted_kinds.append(kind)

        head_clamped = _clamp_temperature(head_t)
        cv_metrics = self._cross_validated_metrics(all_items, has_global=global_t is not None)
        # With a global buffer the kind buffers hold RELATIVE temperatures and the global factor
        # carries the head T; without one each kind buffer must hold the absolute value.
        has_global = global_t is not None
        kind_values = (list(relative_ts.values()) if has_global else
                       [_clamp_temperature(head_clamped * relative_ts[kind]) for kind in range(n_kinds)])
        # The write goes through the same mode-blind face; the ack carries the values now in
        # effect, and persistence rides the final checkpoint (the head-save callback locally,
        # the server's own save remotely).
        applied = self._model.task_head_temperatures(update=dict(
            global_temperature=head_clamped if has_global else None,
            kind_temperatures=kind_values,
        ))['temperatures']
        if (applied is None or (applied['global_temperature'] is not None) != has_global
                or len(applied['kind_temperatures']) != n_kinds):
            raise ValueError("The decision head did not acknowledge every fitted temperature.")
        applied_global = applied['global_temperature']
        applied_kinds = applied['kind_temperatures']
        # The effective temperatures now in effect, for reporting.
        final_ts = {
            kind: (applied_kinds[kind] if applied_global is None else applied_global * applied_kinds[kind])
            for kind in range(n_kinds)
        }

        if not self._is_main_process():
            return

        by_kind_report: Dict[int, Dict] = {}
        for kind in fitted_kinds:
            kind_items = [it for it in all_items if it["kind"] == kind]
            kl, kt, km = _stack_items(kind_items)
            by_kind_report[kind] = {
                "brier_before": brier_score(masked_softmax(kl / served[kind], km), kt, km),
                "brier_after": brier_score(masked_softmax(kl / final_ts[kind], km), kt, km),
                "temperature": final_ts[kind],
                "count": len(kind_items),
            }

        parts = []
        for kind in range(n_kinds):
            if kind in by_kind_report:
                d = by_kind_report[kind]
                parts.append(
                    f"{KIND_NAMES[kind]}: T={d['temperature']:.3f} "
                    f"(Brier {d['brier_before']:.4f} -> {d['brier_after']:.4f}, n={d['count']})"
                )
        logger.info(f"temperature_calibration done (loss={self.loss_name}, head_T={head_t:.3f}): {'; '.join(parts)}")
        if cv_metrics is None:
            logger.info('temperature_calibration: insufficient records for out-of-fold metrics.')
            return
        logger.info(
            f"temperature_calibration CV (out-of-fold): accuracy={cv_metrics['accuracy']:.4f}, "
            f"ece={cv_metrics['ece']:.4f}, brier={cv_metrics['brier']:.4f}, "
            f"nll={cv_metrics['nll']:.4f}"
        )
        if self._base_metrics is not None:
            base = self._base_metrics
            logger.info(
                f"temperature_calibration: base (pre-training, n={self._base_items}) vs tuned "
                f"(out-of-fold): accuracy {base['accuracy']:.4f} -> {cv_metrics['accuracy']:.4f}, "
                f"ece {base['ece']:.4f} -> {cv_metrics['ece']:.4f}, "
                f"brier {base['brier']:.4f} -> {cv_metrics['brier']:.4f}, "
                f"nll {base['nll']:.4f} -> {cv_metrics['nll']:.4f}"
            )

    # -- collection ---------------------------------------------------------

    @staticmethod
    def _check_batch_alignment(meta, kinds, counts, logits, option_mask, target) -> None:
        """Fail loudly when the scored batch desyncs from the client-side records.

        counts / kinds come back from the (possibly remote) forward while the records' meta
        stayed with the batch; equality is the alignment invariant the fit depends on.
        """
        if meta is None:
            raise ValueError("Decision calibration batches must carry decision_meta (n_options/counts/kinds).")
        n_options = [int(n) for n in meta["n_options"]]
        if len(n_options) != len(kinds) or [int(k) for k in meta["kinds"]] != [int(k) for k in kinds]:
            raise ValueError("Decision kinds do not match the client batch.")
        if any(not 0 <= k < len(KIND_NAMES) for k in kinds):
            raise ValueError(f"Decision kinds must be within [0, {len(KIND_NAMES)}), got {sorted(set(kinds))}.")
        if counts is None:
            raise ValueError("Decision calibration batches must carry per-record question counts.")
        counts_list = counts.tolist() if isinstance(counts, torch.Tensor) else [int(c) for c in counts]
        if counts_list != [int(c) for c in meta["counts"]]:
            raise ValueError("Decision record counts do not match the client batch.")
        if sum(counts_list) != len(kinds) or logits.shape[0] != len(kinds):
            raise ValueError("Decision logits do not match the client question count.")
        if n_options and logits.shape[1] < max(n_options):
            raise ValueError("Decision logits are narrower than the records' option counts.")
        for i, n in enumerate(n_options):
            if not option_mask[i, :n].all() or option_mask[i, n:].any():
                raise ValueError("Decision option mask does not match the record's option count.")
        if target.shape[0] != logits.shape[0] or target.shape[1] < logits.shape[1]:
            raise ValueError("Decision targets must have one row per question and cover the option width.")
        if not target.isfinite().all() or (target < 0).any() or (target[~option_mask] != 0).any():
            raise ValueError("Decision targets must be finite, nonnegative, and zero outside real option slots.")

    def _collect_logits_and_targets(self, served: Dict[int, float]) -> Dict[int, Dict]:  # noqa: C901
        """Run the bound eval batches and collect per-kind (RAW logits, target, mask, record_ids).

        The head serves logits already divided by its temperature buffers, so each row is scaled
        back up by the per-kind effective temperature first: the fits then see the raw scores and
        every applied temperature is absolute, correct whatever the buffers held before.
        """
        kind_collected: Dict[int, Dict[str, List[torch.Tensor]]] = {}
        global_record_counter = 0
        for inputs, outputs in self._eval_batches():
            outputs = collect(outputs)
            logits = outputs.get("logits")
            if logits is None:
                continue  # PP stages without readout still participate in every eval forward.
            logits = logits.detach().float().cpu()
            option_mask = outputs.get("option_mask")
            if option_mask is None:
                raise ValueError(
                    "decision outputs must carry option_mask ([total_q, max_opt], True = real option "
                    "slot); a missing mask means the head forward is broken."
                )
            option_mask = option_mask.detach().bool().cpu()
            kinds = outputs.get("kinds")
            if kinds is None:
                continue
            kinds = kinds.detach().cpu().tolist()
            counts = outputs.get("counts")
            labels = inputs.get("labels")
            if labels is None:
                continue  # no labels -> cannot calibrate
            target = labels.detach().cpu()
            if target.dim() == 1:
                target = F.one_hot(target.long(), num_classes=logits.shape[1])
            target = target.float()
            if (logits.dim() != 2 or option_mask.shape != logits.shape or target.dim() != 2):
                raise ValueError('Decision logits, masks and targets must be aligned two-dimensional tensors.')

            # The labels may be padded wider than the batch's own max option count (the wire
            # width); align the logits/mask to the target width so every check below sees one.
            if logits.shape[1] < target.shape[1]:
                pad = target.shape[1] - logits.shape[1]
                logits = F.pad(logits, (0, pad))
                option_mask = F.pad(option_mask, (0, pad))
            self._check_batch_alignment(inputs.get("decision_meta"), kinds, counts, logits, option_mask, target)
            # Unscale by the per-row served temperature, then verify the real slots are finite.
            logits = logits * torch.tensor([served[k] for k in kinds], dtype=torch.float32).unsqueeze(1)
            if not logits.masked_select(option_mask).isfinite().all():
                raise ValueError("Decision logits must be finite on real option slots.")

            counts_list = counts.tolist() if isinstance(counts, torch.Tensor) else [int(c) for c in counts]
            record_ids_batch = []
            for c in counts_list:
                record_ids_batch.extend([global_record_counter] * c)
                global_record_counter += 1

            for i, kind in enumerate(kinds):
                if not target[i].any():
                    continue  # unlabeled row (the loss's positive-mass rule) -> nothing to fit against
                slot = kind_collected.setdefault(kind, {"logits": [], "target": [], "mask": [], "record_ids": []})
                slot["logits"].append(logits[i])
                slot["target"].append(target[i])
                slot["mask"].append(option_mask[i])
                slot["record_ids"].append(record_ids_batch[i])

        result: Dict[int, Dict] = {}
        for kind, parts in kind_collected.items():
            if not parts["logits"]:
                continue
            stacked = _stack_items(
                [
                    {"logits": t, "target": y, "mask": m}
                    for t, y, m in zip(parts["logits"], parts["target"], parts["mask"])
                ]
            )
            result[kind] = {
                "logits": stacked[0],
                "target": stacked[1],
                "mask": stacked[2],
                "record_ids": parts["record_ids"],
            }
        return result

    # -- multi-rank coordination (DDP) -----------------------------------------

    @staticmethod
    def _is_main_process() -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _collect_eval(self, served: Dict[int, float]) -> Dict[int, Dict]:
        """Propagate collection errors to every process before collective fitting/writing."""
        error, collected = None, {}
        try:
            collected = self._collect_logits_and_targets(served)
        except Exception as exc:
            error = str(exc)
        return self._gather_collected(collected, error)

    def _gather_collected(self, collected: Dict[int, Dict], error: Optional[str] = None) -> Dict[int, Dict]:
        """Merge one representative per data shard, preserving records for cross-validation."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            if error is not None:
                raise ValueError(f'Decision calibration collection failed: {error}')
            return collected
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        mesh = self._model.device_mesh
        # TP/CP/PP peers saw the same records. The mesh selects one last-PP representative
        # per data rank, exactly as the Ray forward collector does.
        if rank not in mesh.get_collect_ranks():
            collected = {}
        payload = {}
        for kind, d in collected.items():
            payload[kind] = {
                "logits": d["logits"].detach().cpu().tolist(),
                "target": d["target"].detach().cpu().tolist(),
                "mask": d["mask"].detach().cpu().tolist(),
                "record_ids": [int(rid) + rank * 1_000_000_000 for rid in d["record_ids"]],
            }
        shards = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(shards, (error, payload))
        errors = [message for message, _ in shards if message is not None]
        if errors:
            raise ValueError(f'Decision calibration collection failed: {errors[0]}')

        merged: Dict[int, Dict[str, list]] = {}
        for _, shard in shards:
            for kind, d in shard.items():
                slot = merged.setdefault(kind, {"logits": [], "target": [], "mask": [], "record_ids": []})
                for key in ("logits", "target", "mask", "record_ids"):
                    slot[key].extend(d[key])
        result: Dict[int, Dict] = {}
        for kind, d in merged.items():
            if not d["logits"]:
                continue
            width = max(len(r) for r in d["logits"])

            def pad_rows(rows, pad_value, dt, width=width):
                return torch.tensor([list(r) + [pad_value] * (width - len(r)) for r in rows], dtype=dt)

            result[kind] = {
                "logits": pad_rows(d["logits"], 0.0, torch.float32),
                "target": pad_rows(d["target"], 0.0, torch.float32),
                "mask": pad_rows(d["mask"], False, torch.bool),
                "record_ids": d["record_ids"],
            }
        return result

    @staticmethod
    def _flatten_collected(collected: Dict[int, Dict]) -> List[Dict]:
        all_items: List[Dict] = []
        for kind, kind_data in collected.items():
            logits, target, mask = kind_data["logits"], kind_data["target"], kind_data["mask"]
            record_ids = kind_data["record_ids"]
            for i in range(logits.shape[0]):
                all_items.append(
                    {
                        "logits": logits[i],
                        "target": target[i],
                        "mask": mask[i],
                        "kind": kind,
                        "record_id": record_ids[i],
                    }
                )
        return all_items

    # -- cross-validated metrics -----------------------------------------------

    def _cross_validated_metrics(
        self, all_items: List[Dict], *, has_global: bool
    ) -> Optional[Dict[str, float]]:
        """Fit on the opposite record fold; never substitute an in-sample fit for CV."""
        n_kinds = len(KIND_NAMES)
        record_ids = sorted({it["record_id"] for it in all_items})
        if len(record_ids) < 2:
            return None
        half = {rid: (i % 2) for i, rid in enumerate(record_ids)}

        fold_head_t: Dict[int, float] = {}
        fold_relative: Dict[int, Dict[int, float]] = {0: {}, 1: {}}
        for fold in (0, 1):
            train_items = [it for it in all_items if half[it["record_id"]] != fold]
            if len(train_items) < self.min_items:
                return None
            tr_logits, tr_target, tr_mask = _stack_items(train_items)
            fh = _fit_one_temperature(tr_logits, tr_target, tr_mask, self.loss_name)
            fold_head_t[fold] = fh
            for kind in range(n_kinds):
                kind_train = [it for it in train_items if it["kind"] == kind]
                if len(kind_train) >= self.min_items:
                    k_logits, k_target, k_mask = _stack_items(kind_train)
                    fold_relative[fold][kind] = _fit_one_temperature(
                        k_logits / max(fh, 1e-8), k_target, k_mask, self.loss_name
                    )
                else:
                    fold_relative[fold][kind] = 1.0

        def cv_t(it: Dict) -> float:
            eval_fold = half[it["record_id"]]
            t = fold_head_t[eval_fold] * fold_relative[eval_fold].get(it["kind"], 1.0)
            return t if has_global else _clamp_temperature(t)

        return self._metrics_at(all_items, cv_t)

    def _metrics_at(self, all_items: List[Dict], t_of_item: Callable[[Dict], float]) -> Dict[str, float]:
        """Evaluate item-specific temperatures in vectorized groups without changing metric means."""
        groups: Dict[float, List[Dict]] = {}
        for it in all_items:
            groups.setdefault(t_of_item(it), []).append(it)
        confidences: List[float] = []
        correct: List[bool] = []
        briers: List[float] = []
        nlls: List[float] = []
        for t, items in groups.items():
            logits, target, mask = _stack_items(items)
            probs = masked_softmax(logits / t, mask)
            confidences.extend(probs.max(dim=-1).values.tolist())
            correct.extend((probs.argmax(dim=-1) == target.argmax(dim=-1)).tolist())
            briers.extend(brier_per_question(probs, target, mask).tolist())
            nlls.extend(nll_per_question(logits, target, mask, t).tolist())
        return {
            "accuracy": float(np.mean(correct)) if correct else 0.0,
            "ece": ece_score(confidences, correct),
            "brier": float(np.mean(briers)) if briers else 0.0,
            "nll": float(np.mean(nlls)) if nlls else 0.0,
        }
