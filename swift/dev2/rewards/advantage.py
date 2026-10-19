# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import torch
from typing import Any, Dict, List, Optional, Sequence, Union

__all__ = ['build_reward_weights', 'compute_advantages']


def build_reward_weights(reward_weights: Optional[Sequence[float]], n_funcs: int) -> torch.Tensor:
    """Build/validate the per-function weight vector (``None`` -> all-ones)."""
    if reward_weights is None:
        return torch.ones(n_funcs, dtype=torch.float32)
    if len(reward_weights) != n_funcs:
        raise ValueError(f'reward_weights length {len(reward_weights)} != number of reward funcs {n_funcs}.')
    return torch.tensor(list(reward_weights), dtype=torch.float32)

# Estimators / scalings supported by twinkle.advantage (kept as literals for discoverability).
ESTIMATORS = ('grpo', 'rloo', 'reinforce_plus_plus', 'reinforce_plus_plus_baseline', 'remax', 'opo', 'gpg',
              'grpo_passk')
SCALINGS = ('group', 'batch', 'none', 'gdpo')


def compute_advantages(rewards: Union[torch.Tensor, Sequence[float]],
                       num_generations: int,
                       *,
                       reward_weights: Optional[Sequence[float]] = None,
                       advantage_estimator: str = 'grpo',
                       scale_rewards: str = 'group',
                       kl_in_reward: bool = False,
                       beta: float = 0.0,
                       kl_values: Optional[torch.Tensor] = None,
                       reward_baselines: Optional[Union[torch.Tensor, Sequence[float]]] = None,
                       response_lengths: Optional[Union[torch.Tensor, Sequence[int]]] = None,
                       f_norm: float = 1.0) -> List[float]:
    """Per-sequence advantages from rewards (delegates to ``twinkle.advantage``).

    ``rewards`` is a ``[N, n_funcs]`` matrix or a flat ``[N]`` of combined rewards, each consecutive
    block of ``num_generations`` one prompt group. ``kl_in_reward`` subtracts ``beta * kl_values``
    BEFORE normalization; ``remax`` needs ``reward_baselines``, ``opo`` ``response_lengths``, ``gpg``
    ``f_norm``. Returns ``[N]`` advantages as a plain list.
    """
    from twinkle.advantage import (GPGAdvantage, GRPOAdvantage, GRPOPassKAdvantage, OPOAdvantage,
                                   ReinforcePlusPlusAdvantage, ReinforcePPBaselineAdvantage, ReMaxAdvantage,
                                   RLOOAdvantage)

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
    estimator_cls = {
        'grpo': GRPOAdvantage,
        'rloo': RLOOAdvantage,
        'reinforce_plus_plus': ReinforcePlusPlusAdvantage,
        'reinforce_plus_plus_baseline': ReinforcePPBaselineAdvantage,
        'remax': ReMaxAdvantage,
        'opo': OPOAdvantage,
        'gpg': GPGAdvantage,
        'grpo_passk': GRPOPassKAdvantage,
    }[advantage_estimator]
    # Estimator-specific inputs are forwarded as extra kwargs; the twinkle estimators read only what
    # they need (remax -> reward_baselines, opo -> response_lengths, gpg -> f_norm) and ignore the rest.
    extra_kwargs: Dict[str, Any] = {}
    if reward_baselines is not None:
        extra_kwargs['reward_baselines'] = reward_baselines
    if response_lengths is not None:
        extra_kwargs['response_lengths'] = response_lengths
    if advantage_estimator == 'gpg':
        extra_kwargs['f_norm'] = f_norm
    advantages = estimator_cls()(
        rewards_per_func,
        num_generations,
        scale=scale_rewards,
        reward_weights=weights,
        kl_in_reward=kl_in_reward,
        beta=beta,
        kl_values=kl_values,
        **extra_kwargs,
    )
    return advantages.tolist()
