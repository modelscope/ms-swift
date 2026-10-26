# Copyright (c) ModelScope Contributors. All rights reserved.
"""Post-training temperature calibration for decision (System-1 scoring) models.

After training finishes, this callback runs the model on the eval dataset in eval mode
(``model.eval()``), collects per-question logits and targets, then fits per-kind
temperature(s) and writes them into the head's non-persistent temperature buffer, persisted
to ``scoring_head.json`` via ``save_pretrained``.

The callback is a no-op for heads that do not carry a temperature buffer (e.g.
``ClefJointSchemaHead`` has no temperature), and for decision models that have no eval dataset.
"""
import numpy as np
import os
import torch
import torch.nn.functional as F
from transformers import TrainerControl, TrainerState
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from swift.utils import get_logger
from .base import TrainerCallback

if TYPE_CHECKING:
    from swift.trainers import Trainer, TrainingArguments

logger = get_logger()

MIN_CALIBRATION_ITEMS = 10
HEAD_TEMPERATURE_RANGE = (0.05, 20.0)
# Coarse-grid + golden-section parameters for the Brier path.
_GRID_LO, _GRID_HI, _GRID_STEPS = 0.05, 10.0, 50
_GOLDEN_RATIO = (5**0.5 - 1) / 2  # ~0.618
_GOLDEN_ITERS = 30


def _brier_score(probs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean Brier score over real option slots.

    Args:
        probs:   [N, M] calibrated probabilities.
        target:  [N, M] one-hot (or soft) target distribution.
        mask:    [N, M] boolean, True = real option.
    Returns:
        Scalar mean Brier score (lower is better).
    """
    diff2 = (probs - target).pow(2) * mask
    n_real = mask.sum(dim=-1).clamp(min=1).to(diff2.dtype)
    per_question = diff2.sum(dim=-1) / n_real
    return float(per_question.mean())


def _nll_score(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, temperature: float) -> float:
    """Mean NLL (cross-entropy) of ``softmax(logits / temperature)`` vs target, masked."""
    scaled = logits.float() / max(temperature, 1e-8)
    scaled = scaled.masked_fill(~mask, float('-inf'))
    log_probs = torch.log_softmax(scaled, dim=-1)
    log_probs = torch.nan_to_num(log_probs, nan=-1e4)
    loss = -(target * log_probs).sum(dim=-1)
    n_real = mask.sum(dim=-1).clamp(min=1).to(loss.dtype)
    return float((loss / n_real).mean())


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = logits.float().masked_fill(~mask, float('-inf'))
    probs = torch.softmax(logits, dim=-1)
    return torch.nan_to_num(probs, nan=0.0)


def _ece_score(confidences: List[float], correct: List[bool], n_bins: int = 15) -> float:
    """Expected Calibration Error via uniform binning on [0, 1].

    Lower is better.  ECE measures the gap between predicted confidence and empirical
    accuracy: a well-calibrated model has ECE ≈ 0.
    """
    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray([1.0 if c else 0.0 for c in correct], dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(conf, bin_edges) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(conf)
    for b in range(n_bins):
        m = bin_idx == b
        cnt = int(m.sum())
        if cnt > 0:
            avg_conf = float(conf[m].mean())
            avg_acc = float(corr[m].mean())
            ece += (cnt / n) * abs(avg_conf - avg_acc)
    return float(ece)


def _clamp_temperature(t: float) -> float:
    """Clamp a temperature to a sensible range"""
    return max(HEAD_TEMPERATURE_RANGE[0], min(HEAD_TEMPERATURE_RANGE[1], t))


class TemperatureCalibrationCallback(TrainerCallback):
    """Uses two-stage fitting (global head T + per-kind relative T) and 2-fold
    cross-validation for honest metric reporting.
    """

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        super().__init__(args, trainer)
        self.loss_name = os.environ.get('SWIFT_CALIBRATION_LOSS', 'brier').lower().strip()
        if self.loss_name not in ('brier', 'nll'):
            logger.warning(f'temperature_calibration: unknown SWIFT_CALIBRATION_LOSS='
                           f'{self.loss_name!r}, falling back to "brier".')
            self.loss_name = 'brier'
        self.min_items = int(os.environ.get('SWIFT_CALIBRATION_MIN_ITEMS', MIN_CALIBRATION_ITEMS))
        self._base_metrics: Optional[Dict[str, float]] = None
        self._base_items = 0

    def on_train_begin(self, args: 'TrainingArguments', state: TrainerState, control: TrainerControl, **kwargs):
        """Report-only: no temperature is fitted or written here. Skips silently under the
        same conditions the post-training calibration would skip (no head, no temperature
        buffer, no eval_dataset). Temperatures used are the loaded/factory values.
        """
        from swift.model.decision_head import get_scoring_head
        head = get_scoring_head(self.trainer.model)
        if head is None:
            return
        if not (hasattr(head, 'temperatures') or hasattr(head, 'temps')):
            return
        if getattr(self.trainer, 'eval_dataset', None) is None:
            return

        logger.info('temperature_calibration: evaluating base model on the eval set '
                    '(baseline before training)...')
        try:
            collected = self._collect_logits_and_targets()
        except Exception as e:
            logger.warning(f'temperature_calibration: failed to evaluate base model ({e}), '
                           'skipping baseline report.')
            return
        # Collective: merge every rank's shard so the baseline covers the FULL eval set
        # (DDP shards the eval dataloader). ALL ranks must reach this call.
        collected = self._gather_collected(collected)
        if not self._is_main_process(args):
            return
        if not collected:
            return

        all_items = self._flatten_collected(collected)
        if not all_items:
            return
        head_t, relative = self._served_temperatures(head)
        self._base_metrics = self._in_sample_metrics(all_items, head_t, relative)
        self._base_items = len(all_items)
        logger.info(f'temperature_calibration: base metrics (pre-training, n={self._base_items}): '
                    f'accuracy={self._base_metrics["accuracy"]:.4f}, '
                    f'ece={self._base_metrics["ece"]:.4f}, '
                    f'brier={self._base_metrics["brier"]:.4f}, '
                    f'nll={self._base_metrics["nll"]:.4f}')

    def on_train_end(self, args: 'TrainingArguments', state: TrainerState, control: TrainerControl, **kwargs):
        from swift.model.decision_head import KIND_NAMES, get_scoring_head
        head = get_scoring_head(self.trainer.model)
        if head is None:
            return

        has_temps = hasattr(head, 'temperatures') or hasattr(head, 'temps')
        if not has_temps:
            logger.info('temperature_calibration: head has no temperature buffer, skipping.')
            return

        eval_dataset = getattr(self.trainer, 'eval_dataset', None)
        if eval_dataset is None:
            logger.info('temperature_calibration: no eval_dataset, skipping (temperature stays at '
                        'its loaded/factory value).')
            return

        logger.info(f'temperature_calibration: collecting logits on eval set '
                    f'(loss={self.loss_name})...')
        try:
            collected = self._collect_logits_and_targets()
        except Exception as e:
            logger.warning(f'temperature_calibration: failed to collect logits ({e}); '
                           'temperature stays at its loaded/factory value.')
            return

        # Collective: merge every rank's shard so T is fitted on the FULL eval set, then
        # fit + persist ONLY on the main process
        collected = self._gather_collected(collected)
        if not self._is_main_process(args):
            return

        if not collected:
            logger.info('temperature_calibration: no questions collected from eval set, skipping.')
            return

        if hasattr(head, 'temperatures'):
            buf = head.temperatures
        else:
            buf = head.temps

        n_kinds = len(KIND_NAMES)

        all_items = self._flatten_collected(collected)

        if len(all_items) < self.min_items:
            logger.info(f'temperature_calibration: only {len(all_items)} questions '
                        f'(< {self.min_items}), skipping calibration.')
            return

        # --- Two-stage fit on ALL data (production T) ---
        head_t = self._fit_head_temperature(all_items)
        relative_ts: Dict[int, float] = {}
        fitted_kinds: List[int] = []
        for kind in range(n_kinds):
            kind_items = [it for it in all_items if it['kind'] == kind]
            if len(kind_items) < self.min_items:
                # Not enough samples for this kind; keep relative T = 1.0.
                relative_ts[kind] = 1.0
                logger.info(f'temperature_calibration: kind {KIND_NAMES[kind]} has only '
                            f'{len(kind_items)} questions (< {self.min_items}); keeping '
                            f'relative T=1.0.')
                continue
            rel_t = self._fit_relative_temperature(kind_items, head_t)
            relative_ts[kind] = rel_t
            fitted_kinds.append(kind)

        has_head_t_buf = hasattr(head, 'head_temperature')
        head_clamped = _clamp_temperature(head_t)
        if has_head_t_buf:
            head.head_temperature = torch.as_tensor(head_clamped, dtype=torch.float32)
        final_ts: Dict[int, float] = {}
        for kind in range(n_kinds):
            rel = relative_ts.get(kind, 1.0)
            if has_head_t_buf:
                # Store the relative part; head_temperature holds the global part.
                buf[kind] = rel
                final_ts[kind] = _clamp_temperature(head_clamped * rel)
            else:
                # Fallback: pack the product into the single buffer.
                final_t = _clamp_temperature(head_clamped * rel)
                buf[kind] = final_t
                final_ts[kind] = final_t

        # --- 2-fold CV for honest out-of-fold metrics ---
        cv_metrics = self._cross_validated_metrics(all_items, head_t, relative_ts)

        by_kind_report: Dict[int, Dict] = {}
        for kind in fitted_kinds:
            kind_items = [it for it in all_items if it['kind'] == kind]
            kl = torch.stack([it['logits'] for it in kind_items])
            kt = torch.stack([it['target'] for it in kind_items])
            km = torch.stack([it['mask'] for it in kind_items])
            t = final_ts[kind]
            by_kind_report[kind] = {
                'brier_before': _brier_score(_masked_softmax(kl, km), kt, km),
                'brier_after': _brier_score(_masked_softmax(kl / t, km), kt, km),
                'temperature': t,
                'count': len(kind_items),
            }

        dirs_to_save = [args.output_dir]
        last_ckpt = getattr(state, 'last_model_checkpoint', None)
        if last_ckpt and os.path.isdir(last_ckpt):
            dirs_to_save.append(last_ckpt)
        for save_dir in dirs_to_save:
            head.save_pretrained(save_dir, save_weights=False)

        parts = []
        for kind in range(n_kinds):
            if kind in by_kind_report:
                d = by_kind_report[kind]
                parts.append(f'{KIND_NAMES[kind]}: T={d["temperature"]:.3f} '
                             f'(Brier {d["brier_before"]:.4f} -> {d["brier_after"]:.4f}, n={d["count"]})')
        logger.info(f'temperature_calibration done (loss={self.loss_name}, '
                    f'head_T={head_t:.3f}): {"; ".join(parts)}')
        logger.info(f'temperature_calibration CV (out-of-fold): '
                    f'accuracy={cv_metrics["accuracy"]:.4f}, '
                    f'ece={cv_metrics["ece"]:.4f}, '
                    f'brier={cv_metrics["brier"]:.4f}, '
                    f'nll={cv_metrics["nll"]:.4f}')
        if self._base_metrics is not None:
            base = self._base_metrics
            logger.info(f'temperature_calibration: base (pre-training, n={self._base_items}) '
                        f'vs tuned (out-of-fold): '
                        f'accuracy {base["accuracy"]:.4f} -> {cv_metrics["accuracy"]:.4f}, '
                        f'ece {base["ece"]:.4f} -> {cv_metrics["ece"]:.4f}, '
                        f'brier {base["brier"]:.4f} -> {cv_metrics["brier"]:.4f}, '
                        f'nll {base["nll"]:.4f} -> {cv_metrics["nll"]:.4f}')
        logger.info(f'temperature_calibration: written to '
                    f'{[os.path.join(d, head.HEAD_META_NAME) for d in dirs_to_save]}')

    # -- Multi-rank coordination (DDP) ----------------------------------------

    @staticmethod
    def _is_main_process(args: 'TrainingArguments') -> bool:
        """True on the main process (rank 0); also True when not distributed."""
        return getattr(args, 'process_index', 0) == 0

    @staticmethod
    def _gather_collected(collected: Dict[int, Dict]) -> Dict[int, Dict]:
        """Merge the per-rank sharded collections into the FULL eval set.

        Under DDP each rank's eval dataloader sees only its shard, so a per-rank fit is
        noisy and every rank writing ``scoring_head.json`` lets the last writer win with
        a shard-fitted temperature.  This all-gathers the small per-rank dicts (CPU
        tensors serialised to lists) and concatenates them, offsetting ``record_ids``
        by rank so CV row-splits stay valid.  No-op when torch.distributed is not
        initialised (single-process training).
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return collected
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        payload = {}
        for kind, d in collected.items():
            payload[kind] = {
                'logits': d['logits'].detach().cpu().tolist(),
                'target': d['target'].detach().cpu().tolist(),
                'mask': d['mask'].detach().cpu().tolist(),
                'record_ids': [int(rid) + rank * 1_000_000_000 for rid in d['record_ids']],
            }
        shards: List[Optional[Dict]] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(shards, payload)

        merged: Dict[int, Dict[str, list]] = {}
        for shard in shards:
            for kind, d in shard.items():
                slot = merged.setdefault(kind, {'logits': [], 'target': [], 'mask': [], 'record_ids': []})
                slot['logits'].extend(d['logits'])
                slot['target'].extend(d['target'])
                slot['mask'].extend(d['mask'])
                slot['record_ids'].extend(d['record_ids'])

        result: Dict[int, Dict] = {}
        for kind, d in merged.items():
            if not d['logits']:
                continue
            width = max(len(r) for r in d['logits'])

            def pad_rows(rows: List[list], pad_value, dt) -> torch.Tensor:
                return torch.tensor([list(r) + [pad_value] * (width - len(r)) for r in rows], dtype=dt)

            result[kind] = {
                'logits': pad_rows(d['logits'], 0.0, torch.float32),
                'target': pad_rows(d['target'], 0.0, torch.float32),
                'mask': pad_rows(d['mask'], False, torch.bool),
                'record_ids': d['record_ids'],
            }
        return result

    @staticmethod
    def _flatten_collected(collected: Dict[int, Dict]) -> List[Dict]:
        all_items: List[Dict] = []
        for kind, kind_data in collected.items():
            logits = kind_data['logits']
            target = kind_data['target']
            mask = kind_data['mask']
            record_ids = kind_data['record_ids']
            for i in range(logits.shape[0]):
                all_items.append({
                    'logits': logits[i],
                    'target': target[i],
                    'mask': mask[i],
                    'kind': kind,
                    'record_id': record_ids[i],
                })
        return all_items

    @staticmethod
    def _served_temperatures(head) -> Tuple[float, Dict[int, float]]:
        buf = head.temperatures if hasattr(head, 'temperatures') else head.temps
        head_t = float(head.head_temperature) if hasattr(head, 'head_temperature') else 1.0
        relative = {kind: float(buf[kind]) for kind in range(len(buf))}
        return head_t, relative

    def _collect_logits_and_targets(self) -> Dict[int, Dict]:
        model = self.trainer.model
        was_training = model.training
        model.eval()

        device = self.trainer.args.device
        dataloader = self.trainer.get_eval_dataloader()

        kind_collected: Dict[int, Dict[str, List[torch.Tensor]]] = {}
        global_record_counter = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                labels = batch.pop('labels', None)
                outputs = model(**batch)
                if outputs.logits is None:
                    continue

                logits = outputs.logits.detach().float().cpu()
                option_mask = getattr(outputs, 'option_mask', None)
                if option_mask is None:
                    option_mask = torch.ones_like(logits, dtype=torch.bool)
                else:
                    option_mask = option_mask.detach().bool().cpu()
                kinds = getattr(outputs, 'kinds', None)
                if kinds is None:
                    continue
                kinds = kinds.detach().cpu().tolist()
                counts = getattr(outputs, 'counts', None)

                # Reconstruct per-question record IDs from counts (questions-per-record).
                if counts is not None:
                    counts_list = counts.detach().cpu().tolist()
                    record_ids_batch = []
                    for c in counts_list:
                        record_ids_batch.extend([global_record_counter] * c)
                        global_record_counter += 1
                else:
                    # No counts: treat each question as its own record.
                    record_ids_batch = list(range(global_record_counter, global_record_counter + len(kinds)))
                    global_record_counter += len(kinds)

                if labels is not None:
                    target = labels.detach().float().cpu()
                    if target.dim() == 1:
                        target = F.one_hot(target, num_classes=logits.shape[1]).float()
                else:
                    continue  # no labels → can't calibrate

                for i, kind in enumerate(kinds):
                    kind_collected.setdefault(kind, {'logits': [], 'target': [], 'mask': [], 'record_ids': []})
                    kind_collected[kind]['logits'].append(logits[i])
                    kind_collected[kind]['target'].append(target[i])
                    kind_collected[kind]['mask'].append(option_mask[i])
                    kind_collected[kind]['record_ids'].append(record_ids_batch[i])

        if was_training:
            model.train()

        result: Dict[int, Dict] = {}
        for kind, parts in kind_collected.items():
            if not parts['logits']:
                continue
            max_opts = max(t.shape[0] for t in parts['logits'])
            padded_logits = []
            padded_target = []
            padded_mask = []
            for logit, target, mask in zip(parts['logits'], parts['target'], parts['mask']):
                n = logit.shape[0]
                if n < max_opts:
                    pad = max_opts - n
                    logit = F.pad(logit, (0, pad), value=0.0)
                    target = F.pad(target, (0, pad), value=0.0)
                    mask = F.pad(mask, (0, pad), value=False)
                padded_logits.append(logit)
                padded_target.append(target)
                padded_mask.append(mask)
            result[kind] = {
                'logits': torch.stack(padded_logits),
                'target': torch.stack(padded_target),
                'mask': torch.stack(padded_mask),
                'record_ids': parts['record_ids'],
            }
        return result

    # -- Temperature fitting ------------------------------------------------

    def _fit_head_temperature(self, items: List[Dict]) -> float:
        """Stage 1: fit a single global temperature on ALL questions (pooled across kinds)."""
        logits, target, mask = self._stack_items(items)
        return self._fit_one_temperature(logits, target, mask)

    def _fit_relative_temperature(self, items: List[Dict], head_t: float) -> float:
        """Stage 2: fit per-kind relative T on logits already scaled by head_t."""
        logits, target, mask = self._stack_items(items)
        # Scale by head_t first, then fit a relative T on top.
        logits_scaled = logits / max(head_t, 1e-8)
        rel_t = self._fit_one_temperature(logits_scaled, target, mask)
        return rel_t

    @staticmethod
    def _stack_items(items: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack a list of item dicts into padded [N, M] tensors."""
        max_opts = max(it['logits'].shape[0] for it in items)
        padded_logits = []
        padded_target = []
        padded_mask = []
        for it in items:
            logit = it['logits']
            target = it['target']
            mask = it['mask']
            n = logit.shape[0]
            if n < max_opts:
                pad = max_opts - n
                logit = F.pad(logit, (0, pad), value=0.0)
                target = F.pad(target, (0, pad), value=0.0)
                mask = F.pad(mask, (0, pad), value=False)
            padded_logits.append(logit)
            padded_target.append(target)
            padded_mask.append(mask)
        return (torch.stack(padded_logits), torch.stack(padded_target), torch.stack(padded_mask))

    def _fit_one_temperature(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
        """Fit a single scalar T on the given (logits, target, mask).

        Dispatches to Brier (grid search + golden section) or NLL (LBFGS) based on
        ``self.loss_name``.
        """
        if logits.shape[0] == 0:
            return 1.0
        if self.loss_name == 'nll':
            return self._fit_temperature_nll(logits, target, mask)
        return self._fit_temperature_brier(logits, target, mask)

    @staticmethod
    def _fit_temperature_brier(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
        """Find T that minimises Brier score of ``softmax(logits / T)`` vs target"""

        def brier_at(T: float) -> float:
            return _brier_score(_masked_softmax(logits / T, mask), target, mask)

        # Coarse grid.
        lo, hi = _GRID_LO, _GRID_HI
        n_steps = _GRID_STEPS
        best_T = 1.0
        best_brier = brier_at(1.0)
        for i in range(n_steps + 1):
            T = lo + (hi - lo) * i / n_steps
            b = brier_at(T)
            if b < best_brier:
                best_brier = b
                best_T = T

        span = (hi - lo) / n_steps
        a = max(HEAD_TEMPERATURE_RANGE[0], best_T - span)
        b = min(HEAD_TEMPERATURE_RANGE[1], best_T + span)
        for _ in range(_GOLDEN_ITERS):
            c = b - _GOLDEN_RATIO * (b - a)
            d = a + _GOLDEN_RATIO * (b - a)
            if brier_at(c) < brier_at(d):
                b = d
            else:
                a = c
        best_T = (a + b) / 2
        return _clamp_temperature(best_T)

    @staticmethod
    def _fit_temperature_nll(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
        """Find T that minimises NLL (cross-entropy) using LBFGS with strong_wolfe."""
        # Mask padded positions with a large negative so they vanish under softmax.
        z = logits.float().clone()
        z[~mask] = -1e4
        tgt = target.float()
        log_t = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            log_probs = torch.log_softmax(z / log_t.exp(), dim=-1)
            loss = -(tgt * log_probs).sum(dim=-1).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        T = float(log_t.exp().item())
        return _clamp_temperature(T)

    # -- 2-fold cross-validation for honest metrics -------------------------

    def _cross_validated_metrics(self, all_items: List[Dict], head_t: float,
                                 relative_ts: Dict[int, float]) -> Dict[str, float]:
        """2-fold CV: split records in half, fit on opposite half, evaluate out-of-fold.

        Returns a dict with out-of-fold ``accuracy``, ``ece``, ``brier``, ``nll``.
        If there are too few distinct records for a meaningful split, falls back to
        in-sample metrics using the final T.
        """
        from swift.model.decision_head import KIND_NAMES
        n_kinds = len(KIND_NAMES)

        record_ids = sorted({it['record_id'] for it in all_items})
        if len(record_ids) < 2:
            return self._in_sample_metrics(all_items, head_t, relative_ts)

        half = {rid: (i % 2) for i, rid in enumerate(record_ids)}

        fold_head_t: Dict[int, float] = {}
        fold_relative: Dict[int, Dict[int, float]] = {0: {}, 1: {}}
        for fold in (0, 1):
            train_items = [it for it in all_items if half[it['record_id']] != fold]
            if len(train_items) < self.min_items:
                fold_head_t[fold] = head_t
                for kind in range(n_kinds):
                    fold_relative[fold][kind] = relative_ts.get(kind, 1.0)
                continue
            fh = self._fit_head_temperature(train_items)
            fold_head_t[fold] = fh
            for kind in range(n_kinds):
                kind_train = [it for it in train_items if it['kind'] == kind]
                if len(kind_train) >= self.min_items:
                    fold_relative[fold][kind] = self._fit_relative_temperature(kind_train, fh)
                else:
                    fold_relative[fold][kind] = 1.0

        confidences: List[float] = []
        correct: List[bool] = []
        briers: List[float] = []
        nlls: List[float] = []
        for it in all_items:
            fold = half[it['record_id']]
            eval_fold = 1 - fold
            t = _clamp_temperature(fold_head_t[eval_fold] * fold_relative[eval_fold].get(it['kind'], 1.0))
            logit = it['logits']
            target = it['target']
            mask = it['mask']
            probs = _masked_softmax(logit / t, mask)
            conf = float(probs.max().item())
            pred = int(probs.argmax().item())
            gold = int(target.argmax().item())
            confidences.append(conf)
            correct.append(pred == gold)
            briers.append(_brier_score(probs.unsqueeze(0), target.unsqueeze(0), mask.unsqueeze(0)))
            nlls.append(_nll_score(logit.unsqueeze(0), target.unsqueeze(0), mask.unsqueeze(0), t))

        return {
            'accuracy': float(np.mean(correct)) if correct else 0.0,
            'ece': _ece_score(confidences, correct),
            'brier': float(np.mean(briers)) if briers else 0.0,
            'nll': float(np.mean(nlls)) if nlls else 0.0,
        }

    def _in_sample_metrics(self, all_items: List[Dict], head_t: float, relative_ts: Dict[int,
                                                                                         float]) -> Dict[str, float]:
        """Fallback: in-sample metrics using the final T (no CV)."""
        confidences: List[float] = []
        correct: List[bool] = []
        briers: List[float] = []
        nlls: List[float] = []
        for it in all_items:
            t = _clamp_temperature(head_t * relative_ts.get(it['kind'], 1.0))
            logit = it['logits']
            target = it['target']
            mask = it['mask']
            probs = _masked_softmax(logit / t, mask)
            confidences.append(float(probs.max().item()))
            correct.append(int(probs.argmax().item()) == int(target.argmax().item()))
            briers.append(_brier_score(probs.unsqueeze(0), target.unsqueeze(0), mask.unsqueeze(0)))
            nlls.append(_nll_score(logit.unsqueeze(0), target.unsqueeze(0), mask.unsqueeze(0), t))
        return {
            'accuracy': float(np.mean(correct)) if correct else 0.0,
            'ece': _ece_score(confidences, correct),
            'brier': float(np.mean(briers)) if briers else 0.0,
            'nll': float(np.mean(nlls)) if nlls else 0.0,
        }
