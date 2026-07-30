"""Advantage: turn rewards into per-sequence advantages (L1 atomic API).

Reuses ``swift.rl_core.advantage.compute_advantages`` — swift's backend-agnostic pure function and
the superset implementation (grpo / rloo / reinforce++ estimators; group / batch / none / gdpo
scaling; optional ref-KL-in-reward). This module is a thin, typed entry point over it, so callers do
not need to build tensors by hand.

General API, not recipe-private: inputs are plain rewards (a ``[N, n_funcs]`` matrix or a ``[N]``
list) plus the group size; nothing here knows about rollout sample classes or training loops.
"""
from __future__ import annotations

import torch
from typing import List, Optional, Sequence, Union

__all__ = ['compute_advantages']

# Estimators / scalings supported by swift.rl_core (kept as literals for discoverability).
ESTIMATORS = ('grpo', 'rloo', 'reinforce_plus_plus')
SCALINGS = ('group', 'batch', 'none', 'gdpo')


def compute_advantages(rewards: Union[torch.Tensor, Sequence[float]],
                       num_generations: int,
                       *,
                       reward_weights: Optional[Sequence[float]] = None,
                       advantage_estimator: str = 'grpo',
                       scale_rewards: str = 'group',
                       kl_in_reward: bool = False,
                       beta: float = 0.0,
                       kl_values: Optional[torch.Tensor] = None) -> List[float]:
    """Per-sequence advantages from rewards (delegates to ``swift.rl_core``).

    Args:
        rewards: either a ``[N, n_funcs]`` per-function reward matrix (multi-reward) or a flat ``[N]``
            sequence of already-combined rewards (single reward). Samples must be ordered so each
            consecutive block of ``num_generations`` belongs to one prompt group.
        num_generations: group size K (completions per prompt).
        reward_weights: per-function weights; defaults to all-ones. Only meaningful for a matrix
            input, and its length must equal ``n_funcs``.
        advantage_estimator: one of :data:`ESTIMATORS`.
        scale_rewards: one of :data:`SCALINGS`.
        kl_in_reward: subtract ``beta * kl_values`` from rewards BEFORE normalization (standard
            GRPO/PPO ref-model regularization). Requires ``kl_values``.
        beta: ref-KL penalty coefficient (used only when ``kl_in_reward``).
        kl_values: ``[N]`` per-sample ref-model KL (required when ``kl_in_reward``).

    Returns:
        ``[N]`` advantages as a plain list, aligned with ``rewards``.

    Raises:
        ValueError: unknown estimator/scaling, N not divisible by ``num_generations``, weight-length
            mismatch, or ``kl_in_reward`` without ``kl_values``.
    """
    from swift.dev.reward import build_reward_weights
    from swift.rl_core.advantage import compute_advantages as _compute_advantages

    if advantage_estimator not in ESTIMATORS:
        raise ValueError(f'advantage_estimator {advantage_estimator!r} not in {ESTIMATORS}.')
    if scale_rewards not in SCALINGS:
        raise ValueError(f'scale_rewards {scale_rewards!r} not in {SCALINGS}.')

    rewards_per_func = rewards if isinstance(rewards, torch.Tensor) else torch.tensor(
        list(rewards), dtype=torch.float32)
    if rewards_per_func.ndim == 1:
        rewards_per_func = rewards_per_func.unsqueeze(1)  # [N] -> [N, 1]
    rewards_per_func = rewards_per_func.to(torch.float32)

    n = rewards_per_func.shape[0]
    if num_generations <= 0 or n % num_generations != 0:
        raise ValueError(f'{n} rewards is not divisible by num_generations={num_generations}; '
                         'samples must be grouped as consecutive blocks of num_generations.')
    if kl_in_reward and kl_values is None:
        raise ValueError('kl_in_reward=True requires kl_values ([N] per-sample ref-model KL).')

    weights = build_reward_weights(reward_weights, rewards_per_func.shape[1])
    advantages, _ = _compute_advantages(
        rewards_per_func=rewards_per_func,
        reward_weights=weights,
        num_generations=num_generations,
        advantage_estimator=advantage_estimator,
        scale_rewards=scale_rewards,
        kl_in_reward=kl_in_reward,
        beta=beta,
        kl_values=kl_values,
    )
    return advantages.tolist()
