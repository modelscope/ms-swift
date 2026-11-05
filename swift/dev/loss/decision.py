# Copyright (c) ModelScope Contributors. All rights reserved.
# Losses for the `decision` task_type (typed-decision System-1 scoring models), twinkle contract.
from __future__ import annotations
from typing import Optional

import torch
from twinkle.data_format import InputFeature, LossOutput, ModelOutput
from twinkle.loss import Loss

from swift.dev.model.decision_head import QUESTION_TYPES, masked_softmax
from swift.dev.utils.naming import register_loss

__all__ = ["ScoringLoss", "DistillRPSLoss", "SmoothBrierLoss", "CERPSLoss"]


def _labeled_rows(target: torch.Tensor) -> torch.Tensor:
    """True for target rows carrying a real label. The decision collator pads unlabeled
    questions with all-zero target rows, while a real target distribution (one-hot or a
    full probability sequence) always sums to ~1."""
    return target.sum(dim=-1) > 0


def _rps_score_term(
    outputs: ModelOutput, target: torch.Tensor, option_mask: torch.Tensor, *, normalized: bool
) -> Optional[torch.Tensor]:
    kinds = outputs.get("kinds")
    if kinds is None:
        return None
    score_mask = kinds == QUESTION_TYPES["score"]
    if not bool(score_mask.any()):
        return None
    probs = masked_softmax(outputs["logits"], option_mask)
    gap2 = (probs.cumsum(dim=-1) - target.cumsum(dim=-1)).pow(2)
    if normalized:
        gap2 = gap2[:, :-1]
        denom = (option_mask.sum(dim=-1) - 1).clamp(min=1).to(gap2.dtype)
        per_row = gap2.sum(dim=-1) / denom
    else:
        per_row = gap2.sum(dim=-1)
    # Exclude unlabeled questions (all-zero target rows) from both the numerator and denominator.
    labeled = _labeled_rows(target)
    rps = per_row[score_mask & labeled]
    total_labeled = int(labeled.sum())
    return rps.sum() / total_labeled if total_labeled else rps.sum()


@register_loss("scoring")
class ScoringLoss(Loss):
    """Masked soft cross-entropy over per-question option sets (mean over questions)."""

    require_logits = True
    require_logps = False

    def __call__(self, inputs: InputFeature, outputs: ModelOutput, **kwargs) -> LossOutput:
        logits = outputs["logits"]  # [total_q, max_opt]
        if logits is None:
            raise ValueError(
                'ScoringLoss requires outputs["logits"] (require_logits=True); the forward did not materialize them.'
            )
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("ScoringLoss requires inputs['labels'] = target_probs [total_q, max_opt].")
        option_mask = self._option_mask(outputs, logits)
        # labels is built by the decision collator AFTER twinkle's per-row device move (the flat
        # [total_q, max_opt] target tensor does not exist per row), so it can still sit on CPU while
        # the head's logits are on the model device: align it here, the one consumer that knows both.
        target = labels.to(device=logits.device, dtype=logits.dtype)  # target_probs, pad cols = 0

        # Unlabeled questions arrive as all-zero target rows (mixed-gold batches keep them for the
        # forward pass); they carry no training signal, so mask them out of the CE mean.
        labeled = _labeled_rows(target)
        if bool(labeled.any()):
            per_question = self._masked_soft_ce(logits, option_mask, target)[labeled]  # [n_labeled]
            loss = self._reduce(per_question)
        else:  # a fully unlabeled batch contributes zero loss
            loss = logits.sum() * 0.0
        extra = self._extra_loss(outputs, target, option_mask)
        if extra is not None:
            loss = loss + extra
        return LossOutput(loss=loss, num_tokens=0)

    @staticmethod
    def _option_mask(outputs: ModelOutput, logits: torch.Tensor) -> torch.Tensor:
        mask = outputs.get("option_mask")
        if mask is None or tuple(mask.shape) != tuple(logits.shape):
            raise ValueError(
                "decision outputs must carry option_mask ([total_q, max_opt], True = real option "
                "slot) with the same shape as logits; pad_option_logits always emits it, so a "
                f"missing/mismatched mask ({None if mask is None else tuple(mask.shape)} vs "
                f"{tuple(logits.shape)}) means the head forward is broken."
            )
        return mask

    def micro_batch_scale(self, inputs, indices) -> float:
        # Per-question mean reduction: the scale is the fraction of the batch's questions kept.
        return len(indices) / len(inputs)

    def _masked_soft_ce(self, logits: torch.Tensor, option_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-question CE: softmax over real options only, scored against the target dist"""
        masked_logits = logits.float().masked_fill(~option_mask, float("-inf"))
        # Empty option rows contribute zero; avoid an all -inf log_softmax row.
        masked_logits = masked_logits.masked_fill(~option_mask.any(dim=-1, keepdim=True), 0.0)
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        # Zero padding before multiplication to avoid 0 * -inf (including soft targets).
        log_probs = log_probs.masked_fill(~option_mask, 0.0)
        return -(target * log_probs).sum(dim=-1)

    @staticmethod
    def _reduce(per_question: torch.Tensor) -> torch.Tensor:
        """Mean over this batch's questions."""
        denom = per_question.shape[0]
        if denom == 0:
            return per_question.sum()
        return per_question.sum() / denom

    def _extra_loss(
        self, outputs: ModelOutput, target: torch.Tensor, option_mask: torch.Tensor, **kwargs
    ) -> Optional[torch.Tensor]:
        """Hook for model-specific additive terms; returns a scalar tensor or None."""
        return None


@register_loss("distill_rps")
class DistillRPSLoss(ScoringLoss):
    """Soft-target distillation CE + RPS on `score` questions"""

    RPS_WEIGHT = 0.5

    def __init__(self, rps_weight: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.rps_weight = self.RPS_WEIGHT if rps_weight is None else float(rps_weight)

    def _extra_loss(self, outputs, target, option_mask, **kwargs):
        if self.rps_weight == 0:
            return None
        term = _rps_score_term(outputs, target, option_mask, normalized=True)
        return None if term is None else self.rps_weight * term


@register_loss("smooth_brier")
class SmoothBrierLoss(ScoringLoss):
    """Label-smoothing CE + Brier score"""

    LABEL_SMOOTHING = 0.1
    BRIER_WEIGHT = 1.0

    def __init__(self, label_smoothing: Optional[float] = None, brier_weight: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing = self.LABEL_SMOOTHING if label_smoothing is None else float(label_smoothing)
        self.brier_weight = self.BRIER_WEIGHT if brier_weight is None else float(brier_weight)

    def _masked_soft_ce(self, logits: torch.Tensor, option_mask: torch.Tensor, target: torch.Tensor):
        if self.label_smoothing > 0:
            n_real = option_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(target.dtype)
            uniform = option_mask.to(target.dtype) / n_real
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing * uniform
        return super()._masked_soft_ce(logits, option_mask, target)

    def _extra_loss(self, outputs, target, option_mask, **kwargs):
        if self.brier_weight == 0:
            return None
        probs = masked_softmax(outputs["logits"], option_mask)
        diff2 = (probs - target).pow(2) * option_mask
        n_real = option_mask.sum(dim=-1).clamp(min=1).to(diff2.dtype)
        per_question_brier = diff2.sum(dim=-1) / n_real
        labeled = _labeled_rows(target)
        total_labeled = int(labeled.sum())
        term = per_question_brier[labeled].sum() / total_labeled if total_labeled else per_question_brier[labeled].sum()
        return self.brier_weight * term


@register_loss("ce_rps")
class CERPSLoss(ScoringLoss):
    """Plain CE (no label smoothing) + RPS on `score` questions only"""

    RPS_WEIGHT = 0.5

    def __init__(self, rps_weight: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.rps_weight = self.RPS_WEIGHT if rps_weight is None else float(rps_weight)

    def _extra_loss(self, outputs, target, option_mask, **kwargs):
        if self.rps_weight == 0:
            return None
        term = _rps_score_term(outputs, target, option_mask, normalized=False)
        return None if term is None else self.rps_weight * term
