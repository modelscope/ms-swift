# Copyright (c) ModelScope Contributors. All rights reserved.
# Backend-agnostic scoring pieces: the privileged teacher view, reward folds, calibration scores.
from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Sequence

import torch

from twinkle.processor import PrivilegedView, encode_privileged_feature, encode_privileged_view


def collect(result: Any) -> Any:
    """Resolve a maybe-lazy remote result to its value.

    A method declared ``lazy_collect=True`` returns a pending handle under ray (``.result()``
    materialises it, raising the worker's exception here) and the plain value when it ran in this
    process. Callers that consume the VALUE, or need a worker-side failure to fail THIS step, route
    through here so they are correct under both.
    """
    return result.result() if hasattr(result, 'result') else result


def copy_inputs(features: Sequence[Any]) -> List[Any]:
    """Defensive-copy a forward's inputs: a forward in this process may pad / annotate them in place.

    Remotely the copies are of references, so the cost is small.
    """
    return [copy.deepcopy(feature) for feature in features]


def build_reward_weights(reward_weights: Optional[Sequence[float]], n_funcs: int) -> torch.Tensor:
    """Build/validate the per-function weight vector (``None`` -> all-ones).

    Public because the advantage API needs the same validation, so the "weights must match the number
    of reward functions" rule has one implementation.

    Raises:
        ValueError: ``len(reward_weights) != n_funcs``.
    """
    if reward_weights is None:
        return torch.ones(n_funcs, dtype=torch.float32)
    if len(reward_weights) != n_funcs:
        raise ValueError(f'reward_weights length {len(reward_weights)} != number of reward funcs {n_funcs}.')
    return torch.tensor(list(reward_weights), dtype=torch.float32)


def weighted_rewards(rewards_per_func: Any, reward_weights: Optional[List[float]]):
    """Combine the ``[N, n_scorers]`` matrix into one ``[N]`` weighted-sum reward per sample.

    The weights are ``reward_weights`` (``--orm_weights``), defaulting to all-ones so an unweighted channel is
    a plain sum; ``nansum`` keeps a judge's parse-failure ``nan`` from poisoning the total.
    """
    weight_tensor = build_reward_weights(reward_weights, rewards_per_func.shape[1]).to(
        device=rewards_per_func.device, dtype=rewards_per_func.dtype)
    return (rewards_per_func * weight_tensor.unsqueeze(0)).nansum(dim=1)


def accumulate_reward_metric(metric: Any,
                             rewards_per_func: Any,
                             completion_lengths: List[int],
                             *,
                             scorer_names: List[str],
                             reward_weights: Optional[List[float]]) -> None:
    """Feed one batch's rewards + completion lengths into a CompletionRewardMetric.

    Reset first, so the metric reflects exactly the batch the upcoming optimizer step(s) train on. The
    matrix is the collected ORM score rows (the driver's batch summary carries them row by row);
    per-scorer columns keep their resolved names; a single-callable fallback resolves no names, so
    only the weighted ``total`` is reported there.
    """
    metric.reset()
    names = scorer_names
    rewards: Dict[str, List[float]] = {}
    for col in range(rewards_per_func.shape[1]):
        if col < len(names):
            rewards[names[col]] = [float(v) for v in rewards_per_func[:, col].tolist()]
    rewards['total'] = [float(v) for v in weighted_rewards(rewards_per_func, reward_weights).tolist()]
    metric.accumulate(rewards=rewards, completion_lengths=list(completion_lengths))


def brier_per_question(probs: "torch.Tensor", target: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    """Per-question Brier score over real option slots (row-wise; the batch form averages these)."""
    diff2 = (probs - target).pow(2) * mask
    n_real = mask.sum(dim=-1).clamp(min=1).to(diff2.dtype)
    return diff2.sum(dim=-1) / n_real


def brier_score(probs: "torch.Tensor", target: "torch.Tensor", mask: "torch.Tensor") -> float:
    """Mean Brier score over real option slots (lower is better)."""
    return float(brier_per_question(probs, target, mask).mean())


def nll_per_question(
    logits: "torch.Tensor", target: "torch.Tensor", mask: "torch.Tensor", temperature: float
) -> "torch.Tensor":
    """Per-question NLL (cross-entropy) of ``softmax(logits / temperature)`` vs target, masked."""
    scaled = logits.float() / max(temperature, 1e-8)
    scaled = scaled.masked_fill(~mask, float("-inf"))
    log_probs = torch.log_softmax(scaled, dim=-1)
    log_probs = torch.nan_to_num(log_probs, nan=-1e4)
    loss = -(target * log_probs).sum(dim=-1)
    n_real = mask.sum(dim=-1).clamp(min=1).to(loss.dtype)
    return loss / n_real


def nll_score(logits: "torch.Tensor", target: "torch.Tensor", mask: "torch.Tensor", temperature: float) -> float:
    """Mean NLL (cross-entropy) of ``softmax(logits / temperature)`` vs target, masked."""
    return float(nll_per_question(logits, target, mask, temperature).mean())


def ece_score(confidences: List[float], correct: List[bool], n_bins: int = 15) -> float:
    """Expected Calibration Error via uniform binning on [0, 1] (lower is better; ~0 = calibrated)."""
    import numpy as np

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
            ece += (cnt / n) * abs(float(conf[m].mean()) - float(corr[m].mean()))
    return float(ece)
