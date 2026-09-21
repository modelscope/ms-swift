"""Reward: resolve reward functions and score completions (L1 atomic API).

Reuses swift.dev's ``swift.dev.rewards.orms`` registry (rule-based ORMs: accuracy / format / cosine /
repetition / soft_overlong / ...) rather than reimplementing reward logic. Each ORM's contract is
``__call__(completions, **columns) -> List[float]``.

General API, not recipe-private: functions take plain ``completions`` (list of strings) plus batched
dataset columns, so any caller (CLI recipe, cookbook loop, server) can use them. Nothing here knows
about rollout sample classes or training loops.

Reward models are adapted to the same batch scorer contract through
:func:`build_reward_model_plugins`; asynchronous rule rewards remain out of scope.
"""
from __future__ import annotations
import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from swift.dev.utils import get_logger

logger = get_logger()

# A reward function: (completions, **columns) -> one score per completion.
RewardFunc = Callable[..., List[float]]

__all__ = [
    'RewardFunc',
    'build_reward_model_plugins',
    'build_reward_weights',
    'compute_reward_model_scores',
    'compute_rewards_per_func',
    'get_reward_funcs',
    'weight_rewards',
]


def get_reward_funcs(reward_funcs: Sequence[Any], config: Optional[Any] = None) -> Tuple[List[RewardFunc], List[str]]:
    """Resolve reward specs to callables + display names.

    A spec is either a name registered at the ``reward`` extension point (instantiated as
    ``cls(args=config)``, so the plugin reads its own hyperparameters -- ``cosine_*`` /
    ``repetition_*`` -- off the Config), a plugin class, or an already-callable reward function,
    which is passed through unchanged. Registration itself lives in :mod:`swift.dev.plugin`; this
    function is the reward-shaped door onto it and adds nothing of its own but the naming.

    Args:
        reward_funcs: reward specs (registered names, plugin classes and/or callables).
        config: object carrying reward hyperparameters (any object with the fields the chosen plugins
            read; ``None`` is fine for plugins that need none).

    Returns:
        ``(funcs, names)``; ``names`` are suitable for per-reward metric keys.

    Raises:
        ValueError: unknown name, or a spec that is neither a name, a class nor a callable.
    """
    from swift.dev.plugin import PluginRegistry
    from swift.dev.rewards import REWARD

    funcs: List[RewardFunc] = []
    names: List[str] = []
    for spec in reward_funcs:
        func = PluginRegistry.resolve(REWARD, spec, config=config)
        funcs.append(func)
        names.append(PluginRegistry.display_name(func))
    return funcs, names


class _DefaultRewardModelPlugin:
    """Run a twinkle frozen seq-cls model behind the legacy RM-plugin call contract."""

    def __init__(self, model: Any, template: Any):
        self.model = model
        self.template = template

    def __call__(self, inputs: Sequence[Dict[str, Any]], **kwargs):
        del kwargs
        features = [self.template.encode(copy.deepcopy(row)) for row in inputs]
        outputs = self.model.forward_only(inputs=features, return_logits=True)
        logits = outputs.get('logits') if isinstance(outputs, dict) else None
        if logits is None:
            raise RuntimeError('reward model forward returned no logits.')
        return torch.as_tensor(logits).reshape(-1)


def build_reward_model_plugins(models: Sequence[Any], templates: Sequence[Any],
                               plugin_names: Optional[Sequence[str]] = None) -> Tuple[List[Callable], List[str]]:
    """Construct per-model reward scorers, preserving legacy custom ``rm_plugins`` compatibility."""
    from swift.rewards import rm_plugins

    if len(models) != len(templates):
        raise ValueError(f'reward models/templates length mismatch: {len(models)} != {len(templates)}.')
    names = list(plugin_names) if plugin_names is not None else ['default'] * len(models)
    if len(names) != len(models):
        raise ValueError(f'reward_model_plugin length {len(names)} != reward_model length {len(models)}.')
    plugins: List[Callable] = []
    display_names: List[str] = []
    for model, template, name in zip(models, templates, names):
        if name not in rm_plugins:
            raise ValueError(f'Unknown reward_model_plugin={name!r}; expected one of {sorted(rm_plugins)}.')
        if name == 'default':
            plugin = _DefaultRewardModelPlugin(model, template)
        else:
            raw_model = getattr(model, 'model', model)
            plugin = rm_plugins[name](model=raw_model, template=template)
        plugins.append(plugin)
        display_names.append(getattr(getattr(raw_model if name != 'default' else model, 'config', None),
                                     '_name_or_path', None) or type(plugin).__name__)
    return plugins, display_names


def compute_reward_model_scores(inputs: Sequence[Dict[str, Any]],
                                plugins: Sequence[Callable]) -> torch.Tensor:
    """Score reward rows with per-model plugins -> ``[N, n_models]`` on CPU.

    Each row contains the complete ``messages`` conversation plus the original dataset columns. A
    plugin may return a tensor or a Python sequence; ``None`` is retained as ``nan`` so weighted
    aggregation has the same semantics as rule rewards.
    """
    rewards = torch.zeros((len(inputs), len(plugins)), dtype=torch.float32)
    for index, plugin in enumerate(plugins):
        output = plugin(inputs=copy.deepcopy(list(inputs)))
        if isinstance(output, torch.Tensor):
            scores = output.detach().float().cpu().reshape(-1)
        else:
            scores = torch.tensor(
                [score if score is not None else torch.nan for score in output], dtype=torch.float32).reshape(-1)
        if scores.numel() != len(inputs):
            name = getattr(plugin, '__name__', plugin.__class__.__name__)
            raise ValueError(f'reward model plugin {name!r} returned {scores.numel()} scores for {len(inputs)} inputs.')
        rewards[:, index] = scores
    return rewards


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
