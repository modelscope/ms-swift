"""Reward: resolve reward functions and score completions (L1 atomic API).

Reuses swift's ``swift.rewards.orms`` registry (rule-based ORMs: accuracy / format / cosine /
repetition / soft_overlong / ...) rather than reimplementing reward logic. Each ORM's contract is
``__call__(completions, **columns) -> List[float]``.

General API, not recipe-private: functions take plain ``completions`` (list of strings) plus batched
dataset columns, so any caller (CLI recipe, cookbook loop, server) can use them. Nothing here knows
about rollout sample classes or training loops.

Not covered here: reward *models* (``nn.Module`` + ``rm_plugins``) and async reward functions.
"""
from __future__ import annotations

import torch
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from swift.utils import get_logger

logger = get_logger()

# A reward function: (completions, **columns) -> one score per completion.
RewardFunc = Callable[..., List[float]]

__all__ = ['RewardFunc', 'get_reward_funcs', 'compute_rewards_per_func', 'weight_rewards', 'build_reward_weights']


def get_reward_funcs(reward_funcs: Sequence[Any], config: Optional[Any] = None) -> Tuple[List[RewardFunc], List[str]]:
    """Resolve reward specs to callables + display names.

    A spec is either:
      - a name registered in ``swift.rewards.orms`` -> instantiated as ``orms[name](args=config)``
        (the ORM reads its own hyperparameters off ``config``, e.g. ``cosine_*`` / ``repetition_*``);
      - an already-callable reward function -> passed through unchanged.

    Args:
        reward_funcs: reward specs (registered names and/or callables).
        config: object carrying reward hyperparameters (any object with the fields the chosen ORMs
            read; ``None`` is fine for ORMs that need none).

    Returns:
        ``(funcs, names)``; ``names`` are suitable for per-reward metric keys.

    Raises:
        ValueError: unknown name, or a spec that is neither a name nor callable.
    """
    from swift.rewards import orms

    funcs: List[RewardFunc] = []
    names: List[str] = []
    for spec in reward_funcs:
        if isinstance(spec, str):
            if spec not in orms:
                raise ValueError(f'reward function {spec!r} is not registered in swift.rewards.orms '
                                 f'(available: {sorted(orms)}). Pass a registered name or a callable.')
            func = orms[spec](args=config)
            funcs.append(func)
            names.append(func.__class__.__name__)
        elif callable(spec):
            funcs.append(spec)
            names.append(getattr(spec, '__name__', spec.__class__.__name__))
        else:
            raise ValueError(f'reward function {spec!r} must be a registered name or a callable.')
    return funcs, names


def compute_rewards_per_func(completions: Sequence[str],
                             reward_funcs: Sequence[RewardFunc],
                             columns: Optional[Dict[str, List[Any]]] = None,
                             **extra_kwargs: Any) -> torch.Tensor:
    """Score ``completions`` with every reward function -> ``[N, n_funcs]`` tensor.

    Args:
        completions: one completion string per sample.
        reward_funcs: resolved reward callables (see :func:`get_reward_funcs`).
        columns: batched dataset columns, ``{name: [value_per_sample, ...]}``. Each list must align
            with ``completions`` (so e.g. ``MathAccuracy(completions, solution)`` gets a matching
            ``solution`` list). Passed through as keyword arguments.
        **extra_kwargs: additional keyword arguments forwarded to every reward function.

    Returns:
        ``[N, n_funcs]`` float tensor. A reward returning ``None`` becomes ``nan`` so a broken reward
        is visible rather than silently scoring 0.

    Raises:
        ValueError: a column's length does not match ``completions``, or a reward function returns a
            list whose length does not match ``completions``.
    """
    n = len(completions)
    rewards = torch.zeros((n, len(reward_funcs)), dtype=torch.float32)
    if n == 0:
        return rewards

    columns = columns or {}
    for key, values in columns.items():
        if len(values) != n:
            raise ValueError(f'reward column {key!r} has length {len(values)} but there are {n} completions.')

    kwargs: Dict[str, Any] = {**columns, **extra_kwargs}
    for i, func in enumerate(reward_funcs):
        out = func(list(completions), **kwargs)
        if len(out) != n:
            name = getattr(func, '__name__', func.__class__.__name__)
            raise ValueError(f'reward function {name!r} returned {len(out)} scores for {n} completions.')
        rewards[:, i] = torch.tensor([r if r is not None else torch.nan for r in out], dtype=torch.float32)
    return rewards


def weight_rewards(rewards_per_func: torch.Tensor, reward_weights: Optional[Sequence[float]] = None) -> torch.Tensor:
    """Combine a ``[N, n_funcs]`` reward matrix into ``[N]`` weighted rewards.

    ``reward_weights`` defaults to all-ones (equal weight). ``nan`` entries are ignored per sample
    (``nansum``), matching swift's legacy weighting.
    """
    n_funcs = rewards_per_func.shape[1]
    weights = build_reward_weights(reward_weights, n_funcs)
    return (rewards_per_func * weights.unsqueeze(0)).nansum(dim=1)


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
