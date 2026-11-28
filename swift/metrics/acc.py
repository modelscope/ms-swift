# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from transformers.trainer_utils import EvalPrediction
from typing import Dict, List, Literal, Union

from .base import EvalMetrics


def compute_acc(preds,
                labels,
                *,
                acc_strategy: Literal['token', 'seq'] = 'token',
                is_encoder_decoder: bool = False,
                cu_seqlens=None,
                return_counts: bool = False) -> Dict[str, Union[List[float], torch.Tensor]]:
    """Return per-item correctness, or device-side [correct, valid] counts for accumulation."""
    preds, labels = torch.as_tensor(preds), torch.as_tensor(labels)
    if preds.ndim >= 2 and not is_encoder_decoder:
        labels = labels[..., 1:]
        preds = preds[..., :-1]
    if torch.is_floating_point(labels) or preds.shape != labels.shape:
        return {}

    masks = labels != -100
    correct = preds == labels
    if acc_strategy != 'token' and preds.ndim >= 2:
        if cu_seqlens is not None and masks.shape[0] == 1:
            # Prefix sums match packed slices without reading GPU boundaries in Python.
            bounds = torch.as_tensor(cu_seqlens, device=preds.device, dtype=torch.long)
            bounds = torch.where(bounds < 0, bounds + masks.shape[-1], bounds).clamp(0, masks.shape[-1])
            counts = torch.stack([masks[0], masks[0] & ~correct[0]]).cumsum(dim=-1)
            counts = torch.cat([counts.new_zeros((2, 1)), counts], dim=-1)
            counts = counts[:, bounds[1:]] - counts[:, bounds[:-1]]
            masks = counts[0] > 0
            correct = counts[1] == 0
        else:
            correct = (correct | ~masks).flatten(1).all(dim=-1)
            masks = masks.flatten(1).any(dim=-1)
    if return_counts:
        acc_list = torch.stack([(correct & masks).sum(), masks.sum()])
    else:
        acc_list = correct[masks].tolist()
        if not acc_list:
            return {}
    return {f'{acc_strategy}_acc' if preds.ndim >= 2 else 'acc': acc_list}


class AccMetrics(EvalMetrics):

    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        metric = compute_acc(
            eval_prediction.predictions,
            eval_prediction.label_ids,
            acc_strategy=self.args.acc_strategy,
            is_encoder_decoder=self.trainer.is_encoder_decoder)
        if len(metric) == 0:
            return {}
        return {k: sum(v) / len(v) for k, v in metric.items()}

    def preprocess_logits_for_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        preds = logits.argmax(dim=-1)
        return preds
