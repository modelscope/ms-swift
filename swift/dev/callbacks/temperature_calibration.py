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
from swift.dev.model.decision_head import KIND_NAMES, get_scoring_head, masked_softmax
from swift.dev.utils import get_logger
from swift.dev.utils.plugin import PluginRegistry
from swift.dev.utils.scoring import brier_per_question, brier_score, ece_score, nll_per_question

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


def _served_temperatures(head) -> Tuple[float, Dict[int, float]]:
    head_t = float(head.head_temperature) if hasattr(head, "head_temperature") else 1.0
    relative = {kind: head.get_temperature(kind) for kind in range(len(KIND_NAMES))}
    return head_t, relative


@PluginRegistry.register(CALLBACK, "temperature_calibration")
class TemperatureCalibrationCallback(TrainCallback):
    """Two-stage (global head T + per-kind relative T) post-training calibration.

    The recipe binds the model (and optionally an eval batch iterator) after assembly; without a
    binding, or for heads without a temperature buffer, the callback no-ops. Reporting mirrors the
    legacy flow: a pre-training baseline on ``on_train_begin`` (report-only), then on
    ``on_train_end`` the fitted temperatures, per-kind Brier before/after, and 2-fold
    cross-validated accuracy / ECE / Brier / NLL.
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

    def _head(self):
        if self._model is None:
            return None
        return get_scoring_head(self._model)

    # -- events ----------------------------------------------------------------

    def on_train_begin(self, state: CallbackState, control: CallbackControl) -> None:
        """Report-only baseline: metrics with the loaded/factory temperatures, nothing fitted."""
        head = self._head()
        if head is None or not head.has_kind_temperatures() or self._eval_batches is None:
            return
        logger.info("temperature_calibration: evaluating base model on the eval set (baseline before training)...")
        try:
            collected = self._collect_logits_and_targets()
        except Exception as e:
            logger.warning(f"temperature_calibration: failed to evaluate base model ({e}), skipping baseline.")
            return
        collected = self._gather_collected(collected)
        if not self._is_main_process() or not collected:
            return
        all_items = self._flatten_collected(collected)
        if not all_items:
            return
        head_t, relative = _served_temperatures(head)
        self._base_metrics = self._in_sample_metrics(all_items, head_t, relative)
        self._base_items = len(all_items)
        logger.info(
            f"temperature_calibration: base metrics (pre-training, n={self._base_items}): "
            f"accuracy={self._base_metrics['accuracy']:.4f}, ece={self._base_metrics['ece']:.4f}, "
            f"brier={self._base_metrics['brier']:.4f}, nll={self._base_metrics['nll']:.4f}"
        )

    def on_train_end(self, state: CallbackState, control: CallbackControl) -> None:  # noqa: C901
        head = self._head()
        if head is None:
            return
        if not head.has_kind_temperatures():
            logger.info("temperature_calibration: head has no temperature buffer, skipping.")
            return
        if self._eval_batches is None:
            logger.info(
                "temperature_calibration: no eval batches bound (no eval dataset?), skipping; "
                "temperature stays at its loaded/factory value."
            )
            return

        logger.info(f"temperature_calibration: collecting logits on eval set (loss={self.loss_name})...")
        try:
            collected = self._collect_logits_and_targets()
        except Exception as e:
            logger.warning(
                f"temperature_calibration: failed to collect logits ({e}); "
                "temperature stays at its loaded/factory value."
            )
            return
        collected = self._gather_collected(collected)
        if not self._is_main_process() or not collected:
            logger.info("temperature_calibration: no questions collected from eval set, skipping.")
            return

        all_items = self._flatten_collected(collected)
        if len(all_items) < self.min_items:
            logger.info(
                f"temperature_calibration: only {len(all_items)} questions (< {self.min_items}), skipping calibration."
            )
            return

        n_kinds = len(KIND_NAMES)

        # --- Two-stage fit on ALL data (production T) ---
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

        has_head_t_buf = hasattr(head, "head_temperature")
        head_clamped = _clamp_temperature(head_t)
        if has_head_t_buf:
            head.head_temperature = torch.as_tensor(head_clamped, dtype=torch.float32)
        final_ts: Dict[int, float] = {}
        for kind in range(n_kinds):
            rel = relative_ts.get(kind, 1.0)
            if has_head_t_buf:
                head.set_temperature(kind, rel)
                final_ts[kind] = _clamp_temperature(head_clamped * rel)
            else:
                final_t = _clamp_temperature(head_clamped * rel)
                head.set_temperature(kind, final_t)
                final_ts[kind] = final_t

        cv_metrics = self._cross_validated_metrics(all_items, head_t, relative_ts)

        by_kind_report: Dict[int, Dict] = {}
        for kind in fitted_kinds:
            kind_items = [it for it in all_items if it["kind"] == kind]
            kl, kt, km = _stack_items(kind_items)
            t = final_ts[kind]
            by_kind_report[kind] = {
                "brier_before": brier_score(masked_softmax(kl, km), kt, km),
                "brier_after": brier_score(masked_softmax(kl / t, km), kt, km),
                "temperature": t,
                "count": len(kind_items),
            }

        save_dir = state.output_dir or "output"
        head.save_pretrained(save_dir, save_weights=False)

        parts = []
        for kind in range(n_kinds):
            if kind in by_kind_report:
                d = by_kind_report[kind]
                parts.append(
                    f"{KIND_NAMES[kind]}: T={d['temperature']:.3f} "
                    f"(Brier {d['brier_before']:.4f} -> {d['brier_after']:.4f}, n={d['count']})"
                )
        logger.info(f"temperature_calibration done (loss={self.loss_name}, head_T={head_t:.3f}): {'; '.join(parts)}")
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
        logger.info(f"temperature_calibration: written to {os.path.join(save_dir, head.HEAD_META_NAME)}")

    # -- collection ---------------------------------------------------------

    def _collect_logits_and_targets(self) -> Dict[int, Dict]:  # noqa: C901
        """Run the bound eval batches and collect per-kind (logits, target, mask, record_ids)."""
        kind_collected: Dict[int, Dict[str, List[torch.Tensor]]] = {}
        global_record_counter = 0
        for inputs, outputs in self._eval_batches():
            logits = outputs["logits"]
            if logits is None:
                continue
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
            target = labels.detach().float().cpu()
            if target.dim() == 1:
                target = F.one_hot(target, num_classes=logits.shape[1]).float()

            if counts is not None:
                counts_list = counts.tolist() if isinstance(counts, torch.Tensor) else [int(c) for c in counts]
                record_ids_batch = []
                for c in counts_list:
                    record_ids_batch.extend([global_record_counter] * c)
                    global_record_counter += 1
            else:
                record_ids_batch = list(range(global_record_counter, global_record_counter + len(kinds)))
                global_record_counter += len(kinds)

            for i, kind in enumerate(kinds):
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

    @staticmethod
    def _gather_collected(collected: Dict[int, Dict]) -> Dict[int, Dict]:
        """Merge the per-rank sharded collections into the FULL eval set (no-op single-process)."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return collected
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        payload = {}
        for kind, d in collected.items():
            payload[kind] = {
                "logits": d["logits"].detach().cpu().tolist(),
                "target": d["target"].detach().cpu().tolist(),
                "mask": d["mask"].detach().cpu().tolist(),
                "record_ids": [int(rid) + rank * 1_000_000_000 for rid in d["record_ids"]],
            }
        shards: List[Optional[Dict]] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(shards, payload)

        merged: Dict[int, Dict[str, list]] = {}
        for shard in shards:
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
        self, all_items: List[Dict], head_t: float, relative_ts: Dict[int, float]
    ) -> Dict[str, float]:
        """2-fold CV: split records in half, fit on the opposite half, evaluate out-of-fold."""
        n_kinds = len(KIND_NAMES)
        record_ids = sorted({it["record_id"] for it in all_items})
        if len(record_ids) < 2:
            return self._in_sample_metrics(all_items, head_t, relative_ts)
        half = {rid: (i % 2) for i, rid in enumerate(record_ids)}

        fold_head_t: Dict[int, float] = {}
        fold_relative: Dict[int, Dict[int, float]] = {0: {}, 1: {}}
        for fold in (0, 1):
            train_items = [it for it in all_items if half[it["record_id"]] != fold]
            if len(train_items) < self.min_items:
                fold_head_t[fold] = head_t
                for kind in range(n_kinds):
                    fold_relative[fold][kind] = relative_ts.get(kind, 1.0)
                continue
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
            eval_fold = 1 - half[it["record_id"]]
            return _clamp_temperature(fold_head_t[eval_fold] * fold_relative[eval_fold].get(it["kind"], 1.0))

        return self._metrics_at(all_items, cv_t)

    def _metrics_at(self, all_items: List[Dict], t_of_item: Callable[[Dict], float]) -> Dict[str, float]:
        """Shared metric body: apply each item's temperature, then acc / ece / brier / nll.

        Rows are grouped by their temperature so each group stacks into ONE padded batch: the math
        runs as a handful of vectorised ops per distinct T instead of one op per question. ECE and
        the means are order-independent, so grouping does not change the numbers.
        """
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

    def _in_sample_metrics(
        self, all_items: List[Dict], head_t: float, relative_ts: Dict[int, float]
    ) -> Dict[str, float]:
        """Fallback: in-sample metrics using the final T (no CV)."""
        return self._metrics_at(all_items, lambda it: _clamp_temperature(head_t * relative_ts.get(it["kind"], 1.0)))
