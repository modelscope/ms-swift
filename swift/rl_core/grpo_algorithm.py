# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional

from swift.rl_core.data import GRPOSample
from swift.utils import get_logger

logger = get_logger()


def _is_async_reward(func: Callable) -> bool:
    import asyncio
    return asyncio.iscoroutinefunction(func) or asyncio.iscoroutinefunction(getattr(func, '__call__', None))


def compute_rewards_per_func(
    samples: List[GRPOSample],
    reward_funcs: List[Callable],
    reward_model_plugins: List[Optional[Any]],
    device: torch.device,
    trainer_state: Optional[Any] = None,
    extra_reward_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Compute per-function rewards for ``samples``.

    Supports sync reward callables, async reward callables (auto-detected and
    run via ``asyncio.run``), and reward models (``nn.Module`` instances) via
    ``reward_model_plugins``.

    Args:
        samples: On-policy samples carrying completions in ``messages[-1]``.
        reward_funcs: Reward callables / models.
        reward_model_plugins: Optional model plugins aligned with ``reward_funcs``.
        device: Target device for the returned tensor.
        trainer_state: Passed to reward functions as ``trainer_state`` kwarg.

    Returns:
        ``[N, n_funcs]`` tensor of rewards.
    """
    if reward_model_plugins is None:
        reward_model_plugins = [None] * len(reward_funcs)
    async_indices = [i for i, func in enumerate(reward_funcs) if _is_async_reward(func)]

    rewards_per_func = torch.zeros((len(samples), len(reward_funcs)), device=device)
    completions = [s.messages[-1]['content'] for s in samples]

    reward_kwargs: Dict[str, Any] = {'trainer_state': trainer_state}
    if extra_reward_kwargs:
        reward_kwargs.update(extra_reward_kwargs)
    reward_rows = [s.to_reward_row() for s in samples]
    if reward_rows:
        from swift.dataset import RowPreprocessor
        reward_kwargs.update(RowPreprocessor.rows_to_batched(reward_rows))

    for i, (reward_func, reward_model_plugin) in enumerate(zip(reward_funcs, reward_model_plugins)):
        if isinstance(reward_func, nn.Module):
            output = reward_model_plugin(inputs=reward_rows, **reward_kwargs)
            output = [reward if reward is not None else torch.nan for reward in output]
            rewards_per_func[:, i] = torch.tensor(output, dtype=torch.float32, device=device)
        elif i in async_indices:
            # Async rewards are executed below.
            pass
        else:
            output = reward_func(completions, **reward_kwargs)
            output = [reward if reward is not None else torch.nan for reward in output]
            rewards_per_func[:, i] = torch.tensor(output, dtype=torch.float32, device=device)

    if async_indices:
        import asyncio

        async def _run_async_funcs():
            coros = [reward_funcs[idx](completions, **reward_kwargs) for idx in async_indices]
            return await asyncio.gather(*coros)

        for idx, output in zip(async_indices, asyncio.run(_run_async_funcs())):
            output = [r if r is not None else torch.nan for r in output]
            rewards_per_func[:, idx] = torch.tensor(output, dtype=torch.float32, device=device)

    if rewards_per_func.shape[1] > 0 and torch.isnan(rewards_per_func).all(dim=1).any():
        nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
        row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != 'trainer_state'}
        row_reward_kwargs['completion'] = completions[nan_row_idx]
        logger.warning(f'All reward functions returned None for kwargs: {row_reward_kwargs}. '
                       'Please ensure that at least one reward function returns a valid reward.')

    return rewards_per_func


def score_completions(
    samples: List[GRPOSample],
    reward_funcs: List[Callable],
    reward_model_plugins: List[Optional[Any]],
    use_gym_env: bool,
    device: torch.device,
    trainer_state: Optional[Any] = None,
    extra_reward_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Score completions and return per-function rewards.

    When ``use_gym_env`` is set, the ``total_reward`` stored in
    ``sample.rollout_infos`` is appended as an extra reward column.
    """
    if use_gym_env:
        gym_reward = torch.tensor([s.rollout_infos['total_reward'] for s in samples],
                                  dtype=torch.float32,
                                  device=device).unsqueeze(1)
        if not reward_funcs:
            return gym_reward
        func_rewards = compute_rewards_per_func(
            samples,
            reward_funcs,
            reward_model_plugins,
            device=device,
            trainer_state=trainer_state,
            extra_reward_kwargs=extra_reward_kwargs,
        )
        return torch.cat([func_rewards, gym_reward], dim=1)

    return compute_rewards_per_func(
        samples,
        reward_funcs,
        reward_model_plugins,
        device=device,
        trainer_state=trainer_state,
        extra_reward_kwargs=extra_reward_kwargs,
    )


def compute_std_for_dynamic_sampling(
    rewards_per_func: torch.Tensor,
    reward_weights: torch.Tensor,
    num_generations: int,
) -> torch.Tensor:
    """Compute per-sample reward std used by dynamic sampling (DAPO).

    Returns a ``[N]`` tensor; callers are expected to pass already-global rewards.
    """
    rewards = (rewards_per_func * reward_weights.unsqueeze(0)).nansum(dim=1)

    if num_generations > 1:
        grouped = rewards.view(-1, num_generations)
        return grouped.std(dim=1).repeat_interleave(num_generations)
    return torch.zeros_like(rewards)
