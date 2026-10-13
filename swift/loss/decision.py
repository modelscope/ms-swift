# Copyright (c) ModelScope Contributors. All rights reserved.
"""Losses for the `decision` task_type (typed-decision System-1 scoring models).

`outputs` is a `ScoringOutput` whose `logits` are per-option scores `[total_q, max_opt]`
(padded, read with `option_mask`); `labels` is the padded target distribution `target_probs`
`[total_q, max_opt]` (one-hot for the CE heads, soft for JEV distillation; pad cols = 0).

`ScoringLoss` implements the shared piece: masked soft cross-entropy over each question's own
option set (softmax restricted to the real options, then CE against the target distribution).
This one form covers both one-hot CE and soft-target distillation,
because CE-with-soft-targets equals KL up to an additive constant that does not affect grads.
Subclasses add calibration terms (RPS for ordinal `score`, Brier for all-option calibration)
by overriding `_extra_loss`, which is summed onto the reduced CE loss.
"""
import os
import torch
from typing import Optional

from swift.model.decision_head import QUESTION_TYPES, masked_softmax
from .base import BaseLoss


class ScoringLoss(BaseLoss):
    """Masked soft cross-entropy over per-question option sets."""

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs) -> torch.Tensor:
        logits = outputs.logits  # [total_q, max_opt]
        option_mask = self._option_mask(outputs, logits)
        target = labels.to(logits.dtype)  # target_probs [total_q, max_opt], pad cols = 0

        per_question = self._masked_soft_ce(logits, option_mask, target)  # [total_q]
        loss = self._reduce(per_question, num_items_in_batch)
        if loss_scale is not None:
            loss = loss * loss_scale

        extra = self._extra_loss(
            outputs, target, option_mask, num_items_in_batch=num_items_in_batch, loss_scale=loss_scale, **kwargs)
        if extra is not None:
            loss = loss + extra
        return loss

    @staticmethod
    def _option_mask(outputs, logits: torch.Tensor) -> torch.Tensor:
        mask = getattr(outputs, 'option_mask', None)
        return torch.ones_like(logits, dtype=torch.bool) if mask is None else mask

    def _masked_soft_ce(self, logits: torch.Tensor, option_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-question CE: softmax over real options only, scored against the target dist.

        Returns `[total_q]` (one scalar per question). Pad slots carry prob 0 (masked_softmax)
        and target 0, so they contribute nothing; the `* option_mask` guards against a stray
        nonzero target landing on a pad column.
        """
        probs = masked_softmax(logits, option_mask)  # [total_q, max_opt], pad cols = 0
        log_probs = torch.log(probs.clamp_min(1e-9))
        return -(target * log_probs * option_mask).sum(dim=-1)

    def _reduce(self, per_question: torch.Tensor, num_items_in_batch: Optional[int]) -> torch.Tensor:
        """Mean over this micro-batch's questions."""
        if num_items_in_batch is not None:
            return per_question.sum() / num_items_in_batch
        denom = per_question.shape[0]
        if denom == 0:
            return per_question.sum()
        return per_question.sum() / denom

    def _extra_loss(self, outputs, target, option_mask, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        """Hook for model-specific additive terms; returns a scalar tensor or None."""
        return None


class DistillRPSLoss(ScoringLoss):
    """Soft-target distillation CE + RPS on `score` questions.

    The inherited masked soft-CE IS the KL term: CE(target, model) = KL(target || model) + H(target),
    and H(target) is constant w.r.t. the model, so the gradients match. The teacher targets are soft
    distributions (`target_probs` from the distill corpus), so CE-with-soft-targets is the right form.

    RPS (Ranked Probability Score) is the ordinal calibration term for the 0..5 `score` kind: the mean
    squared gap between the model's and the target's cumulative distributions over the ordered options.
    It is added as `0.5 * mean_over_all_questions(RPS_q * 1{score})`, sharing the CE's denominator so
    the two terms stay on the same per-question scale.
    """
    RPS_WEIGHT = 0.5

    def _extra_loss(self, outputs, target, option_mask, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        kinds = getattr(outputs, 'kinds', None)
        if kinds is None:
            return None
        score_mask = kinds == QUESTION_TYPES['score']
        if not bool(score_mask.any()):
            return None
        probs = masked_softmax(outputs.logits, option_mask)
        cum_model = probs.cumsum(dim=-1)[:, :-1]
        cum_target = target.cumsum(dim=-1)[:, :-1]
        gap2 = (cum_model - cum_target).pow(2)
        denom = (option_mask.sum(dim=-1) - 1).clamp(min=1).to(gap2.dtype)
        per_row_rps = gap2.sum(dim=-1) / denom  # [total_q]
        rps = per_row_rps[score_mask]
        if num_items_in_batch is not None:
            term = rps.sum() / num_items_in_batch
        else:
            total_q = per_row_rps.shape[0]
            term = rps.sum() / total_q if total_q else rps.sum()
        term = self.RPS_WEIGHT * term
        if loss_scale is not None:
            term = term * loss_scale
        return term


class SmoothBrierLoss(ScoringLoss):
    """Label-smoothing CE + Brier score.

    Targets are HARD (the one correct option per question -> a one-hot `target_probs`), so the
    inherited masked soft-CE is a plain per-question CE; label smoothing is applied to the target over
    its real options before the CE. The Brier score (mean squared gap between the calibrated
    probability and the one-hot target, over real options) is added as the calibration term

    The exact label-smoothing / Brier weights are not published, default to a standard
    `0.1` / `1.0` and are overridable via `CLEF_LABEL_SMOOTHING` / `CLEF_BRIER_WEIGHT`.
    """
    LABEL_SMOOTHING = 0.1
    BRIER_WEIGHT = 1.0

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.label_smoothing = float(os.environ.get('CLEF_LABEL_SMOOTHING', self.LABEL_SMOOTHING))
        self.brier_weight = float(os.environ.get('CLEF_BRIER_WEIGHT', self.BRIER_WEIGHT))

    def _masked_soft_ce(self, logits: torch.Tensor, option_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            n_real = option_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(target.dtype)
            uniform = option_mask.to(target.dtype) / n_real
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing * uniform
        return super()._masked_soft_ce(logits, option_mask, target)

    def _extra_loss(self, outputs, target, option_mask, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        if self.brier_weight == 0:
            return None
        probs = masked_softmax(outputs.logits, option_mask)
        diff2 = (probs - target).pow(2) * option_mask
        n_real = option_mask.sum(dim=-1).clamp(min=1).to(diff2.dtype)
        per_question_brier = diff2.sum(dim=-1) / n_real
        if num_items_in_batch is not None:
            term = per_question_brier.sum() / num_items_in_batch
        else:
            total_q = per_question_brier.shape[0]
            term = per_question_brier.sum() / total_q if total_q else per_question_brier.sum()
        term = self.brier_weight * term
        if loss_scale is not None:
            term = term * loss_scale
        return term


class CERPSLoss(ScoringLoss):
    """Plain CE (no label smoothing) + RPS on `score` questions only.

    Uses PROPER scoring rules only: the inherited masked soft-CE over each question's own option
    columns IS `log_score` (noul 2 cols [yes,no] -> binary log score; choice K+1 cols incl. abstain;
    score L cols), which is the correct proper rule for all three kinds. No label smoothing is applied
    (unlike `SmoothBrierLoss`): smoothing is NOT a proper scoring rule and biases probabilities away
    from the true frequency.

    On top of the CE it adds `rps` for the ordered `score` kind only (`RPS_WEIGHT *
    mean_over_questions(rps * 1{score})`, sharing the CE denominator so both terms stay on the same
    per-question scale) -- the order-aware calibration term the ordinal head is trained with. `rps`
    is computed as a plain sum over the cumulative gap, no per-row normalisation;
    pad columns contribute 0 to both CDFs (probs and target both cumulative-sum to 1 over the real
    options), so the padded width is safe.

    The `relu(sum(mu)-1)` validity penalty is IDENTICALLY ZERO under this
    checkpoint's softmax norm (sum(mu) = 1 - P(abstain) <= 1), so it is omitted. `brier` is available
    only the RPS weight is exposed here, defaulting to 0.5 and overridable via `OMNIJEV_RPS_WEIGHT`
    """
    RPS_WEIGHT = 0.5

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.rps_weight = float(os.environ.get('OMNIJEV_RPS_WEIGHT', self.RPS_WEIGHT))

    def _extra_loss(self, outputs, target, option_mask, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        if self.rps_weight == 0:
            return None
        kinds = getattr(outputs, 'kinds', None)
        if kinds is None:
            return None
        score_mask = kinds == QUESTION_TYPES['score']
        if not bool(score_mask.any()):
            return None
        probs = masked_softmax(outputs.logits, option_mask)
        cum_model = probs.cumsum(dim=-1)
        cum_target = target.cumsum(dim=-1)
        per_row_rps = (cum_model - cum_target).pow(2).sum(dim=-1)  # [total_q]
        rps = per_row_rps[score_mask]
        if num_items_in_batch is not None:
            term = rps.sum() / num_items_in_batch
        else:
            total_q = per_row_rps.shape[0]
            term = rps.sum() / total_q if total_q else rps.sum()
        term = self.rps_weight * term
        if loss_scale is not None:
            term = term * loss_scale
        return term
