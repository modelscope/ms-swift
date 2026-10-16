"""Reward: resolve reward functions and score completions (L1 atomic API).

Reuses swift.dev's ``swift.dev.rewards.orms`` registry (rule-based ORMs: accuracy / format / cosine /
repetition / soft_overlong / ...) rather than reimplementing reward logic. Each ORM's contract is
``__call__(completions, **columns) -> List[float]``.

General API, not recipe-private: functions take plain ``completions`` (list of strings) plus batched
dataset columns, so any caller (CLI recipe, cookbook loop, server) can use them. Nothing here knows
about rollout sample classes or training loops.

Reward models are adapted to the same batch scorer contract through
:func:`build_reward_model_plugins`. A rule reward may be written ``async def``: the scoring loop
detects the coroutines and resolves them with one ``asyncio.gather``, so their I/O overlaps.
"""
from __future__ import annotations
import asyncio
import copy
import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from swift.dev.utils import get_logger

logger = get_logger()

# A reward function: (completions, **columns) -> one score per completion.
RewardFunc = Callable[..., List[float]]

__all__ = [
    'RewardFunc',
    'build_frozen_reward_model',
    'build_reward_model_plugins',
    'build_reward_weights',
    'compute_rewards_per_func',
    'get_reward_funcs',
    'is_registered_reward',
    'score_completions',
    'weight_rewards',
]


def is_registered_reward(spec: Any) -> bool:
    """True when ``spec`` is a reward RULE rather than a reward-model id.

    The merged ``--orm`` / ``--prm`` selectors mix rule names with reward-model ids in one list, so a
    caller must tell them apart before resolving: a registered name (a key of the ``orms`` registry the
    ``reward`` extension point adopts), a plugin class, or an already-callable reward is a rule scored
    on completions; anything else is a reward-model id, resolved through the model path. Registered-name-
    first is the rule, so an id that happens to equal a rule name is taken as the rule.
    """
    from swift.dev.utils.plugin import PluginRegistry
    from swift.dev.rewards import REWARD

    if isinstance(spec, str):
        return spec in PluginRegistry.kind(REWARD).entries
    return isinstance(spec, type) or callable(spec)


def get_reward_funcs(reward_funcs: Sequence[Any], config: Optional[Any] = None) -> Tuple[List[RewardFunc], List[str]]:
    """Resolve reward specs to callables + display names.

    A spec is either a name registered at the ``reward`` extension point (instantiated as
    ``cls(args=config)``, so the plugin reads its own hyperparameters -- ``cosine_*`` /
    ``repetition_*`` -- off the Config), a plugin class, or an already-callable reward function,
    which is passed through unchanged. Registration itself lives in :mod:`swift.dev.utils.plugin`; this
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
    from swift.dev.utils.plugin import PluginRegistry
    from swift.dev.rewards import REWARD

    funcs: List[RewardFunc] = []
    names: List[str] = []
    for spec in reward_funcs:
        func = PluginRegistry.resolve(REWARD, spec, config=config)
        funcs.append(func)
        names.append(PluginRegistry.display_name(func))
    return funcs, names


class _ScalarRewardModelPlugin:
    """Run a twinkle frozen seq-cls model behind the ``plugin(inputs=rows)`` reward-model call contract.

    The SCALAR head adapter only. Generative and api judges ("score by prompting") are the other
    :class:`~swift.dev.rewards.scorer.RewardScorer` kinds in :mod:`swift.dev.rewards.scorer`, selected by
    ``build_model_reward`` from the checkpoint's real ``task_type``; every kind scores through the same
    :func:`score_completions`, so a reward channel is judge-capable at the contract level. Placing a
    generative judge in online RL (its sampler / tp-aware DeviceGroup) is the phase③ auxiliary-placement work.
    """

    def __init__(self, model: Any, template: Any):
        self.model = model
        self.template = template
        #: ``overlap_by_placement`` key: this reward model's own actor identity. Each RM is a distinct model
        #: on its own ``reward_<i>`` DeviceGroup, so separate reward models score concurrently.
        self.placement_key = id(model)

    def __call__(self, inputs: Sequence[Dict[str, Any]], **kwargs):
        del kwargs
        features = [self.template.encode(copy.deepcopy(row)) for row in inputs]
        # task='seq_cls': this plugin always wraps a num_labels=1 classification head whose forward pools to a
        # [B, 1] score, NOT vocab logits. Forwarding it as the default 'causal_lm' would make twinkle gather
        # label logps out of that [B, 1] score (a shape error on transformers, and on megatron it would miss the
        # seq_cls pooling branch entirely). Passing task='seq_cls' routes both backends through the score head.
        outputs = self.model.forward_only(inputs=features, return_logits=True, task='seq_cls')
        logits = outputs.get('logits') if isinstance(outputs, dict) else None
        if logits is None:
            raise RuntimeError('reward model forward returned no logits.')
        return torch.as_tensor(logits).reshape(-1)


def build_frozen_reward_model(model_id: str,
                              model_config: Any,
                              template_config: Any,
                              *,
                              model_type: Optional[str] = None,
                              revision: Optional[str] = None,
                              template_name: Optional[str] = None,
                              adapter: Optional[str] = None,
                              distributed_config: Optional['DistributedConfig'] = None,
                              remote_group: Optional[str] = None,
                              parallel_spec: Optional[str] = None) -> Tuple[Any, Any]:
    """Build ONE frozen reward model plus its own template -> ``(model, template)``.

    The shared core of the GRPO and best-of-n sampling reward paths, so a reward model is built the
    same way everywhere: its real ``task_type`` / ``num_labels`` come from the checkpoint's metadata
    (``get_model_info_meta``) rather than a hard-coded head -- a scalar RM resolves to ``seq_cls`` with
    ``num_labels=1``, a reranker to ``num_labels=1`` -- and it is built with ``build_model``, in local
    mode unless a ``distributed_config`` is threaded in. An optional frozen LoRA is attached through
    ``configure_frozen_adapter(role='reward')``.

    Callers own the scoring contract: :class:`~swift.dev.rewards.scorer.ModelReward` wraps the pair in
    :func:`build_reward_model_plugins` and adapts it to the ``func(completions, **columns)``
    :class:`~swift.dev.rewards.scorer.RewardScorer` contract that every reward channel scores through.
    This function only builds; it does not score.

    Args:
        model_id: the reward model path/id.
        model_config: the run's ModelConfig, copied then overridden with this reward model's identity
            (model/type/revision/task_type/num_labels) -- the caller's config is not mutated.
        template_config: the run's TemplateConfig, copied; ``template`` is overridden by
            ``template_name`` and ``max_length`` cleared so a reward model is not truncated to the
            policy's budget.
        template_name: this reward model's own chat template (a reward model need not share the
            policy's); None keeps the config's.
        adapter: an optional frozen LoRA checkpoint dir to attach.
        distributed_config: optional parallel config for the reward model itself. None keeps the
            historical single-process local build; threading the run's config lets a ``parallel_spec``
            (or the integer parallel sizes) drive the reward model's DeviceMesh as well.
        remote_group: optional Ray DeviceGroup name for the reward model (mode='ray' only). It
            defaults to None so ``build_model`` uses its own 'model' group; a sampling run threads a
            dedicated reward group so the frozen RM lands on its own GPUs instead of the trainable
            model's.
        parallel_spec: this reward model's OWN parallel layout (e.g. ``dp2``, ``fsdp2``), parsed into a
            ``DeviceMesh`` and handed to ``build_model`` so the frozen RM is placed/sharded by it instead
            of the default pure data-parallel mesh. None keeps that default. A scalar RM is built on the
            transformers backend, which shards by dp/fsdp/ep/ulysses; tp/pp are not weight-sharding dims
            there (they have no sharding effect on a frozen forward), so express those through the
            megatron backend or a generative judge (whose tp reaches the engine) instead.
    """
    from twinkle import DeviceMesh

    from swift.dev.builders import build_model, build_template, load_model_processor
    from swift.dev.config import DistributedConfig
    from swift.dev.recipe.common.assembly import configure_frozen_adapter
    from swift.model import get_model_info_meta

    model_info, _ = get_model_info_meta(model_id, model_type=model_type, revision=revision)
    reward_model_config = copy.copy(model_config)
    reward_model_config.model = model_id
    reward_model_config.model_type = model_type or model_info.model_type
    reward_model_config.model_revision = revision
    reward_model_config.task_type = model_info.task_type
    reward_model_config.num_labels = model_info.num_labels

    _, processor = load_model_processor(reward_model_config)
    reward_template_config = copy.copy(template_config)
    reward_template_config.template = template_name
    reward_template_config.max_length = None
    reward_template = build_template(
        reward_template_config, processor, task_type=reward_model_config.task_type)
    reward_template.max_length = None

    device_mesh = DeviceMesh.from_spec(parallel_spec) if parallel_spec else None
    reward_model = build_model(
        reward_model_config,
        distributed_config or DistributedConfig(mode='local'),
        device_mesh=device_mesh,
        remote_group=remote_group)
    reward_model = configure_frozen_adapter(
        reward_model, reward_template, [adapter] if adapter else [], role='reward')
    if getattr(reward_template, 'use_model', False):
        reward_template.model = getattr(reward_model, 'model', reward_model)
    return reward_model, reward_template


def build_reward_model_plugins(models: Sequence[Any], templates: Sequence[Any]) -> Tuple[List[Callable], List[str]]:
    """Construct one scalar reward-model scorer per model, plus its display name.

    Every model is scored by :class:`_ScalarRewardModelPlugin` (a frozen seq-cls forward). The legacy
    ``rm_plugins`` roster -- a name -> custom-scorer table selected per model by ``reward_model_plugin``
    -- is gone: dev scores a reward model natively, and the "score by prompting" case is the generative
    judge on the infer path, so there is no second scorer registry to keep in sync (nor a half-wired
    'default'-only route into legacy ``swift.rewards``).
    """
    if len(models) != len(templates):
        raise ValueError(f'reward models/templates length mismatch: {len(models)} != {len(templates)}.')
    plugins: List[Callable] = []
    display_names: List[str] = []
    for model, template in zip(models, templates):
        plugin = _ScalarRewardModelPlugin(model, template)
        plugins.append(plugin)
        display_names.append(
            getattr(getattr(model, 'config', None), '_name_or_path', None) or type(plugin).__name__)
    return plugins, display_names


def _resolve_awaitables(results: List[Any]) -> List[Any]:
    """Resolve any awaitable reward results concurrently, keeping the rest in place.

    A rule reward may be written ``async def`` (its scoring does I/O -- an API call, a database
    query), so calling it yields a coroutine instead of scores. Gathering every awaitable in one
    ``asyncio.run`` overlaps those round trips; a synchronous batch has no awaitable and is returned
    untouched, so no event loop is started for the common case.
    """
    pending = [i for i, out in enumerate(results) if inspect.isawaitable(out)]
    if not pending:
        return results

    async def _gather() -> None:
        resolved = await asyncio.gather(*(results[i] for i in pending))
        for i, value in zip(pending, resolved):
            results[i] = value

    asyncio.run(_gather())
    return results


def _as_score_column(output: Any, scorer: Any, n: int) -> torch.Tensor:
    """Normalize one scorer's batch output to a length-``n`` float column (``None`` -> ``nan``).

    A scorer may return a tensor (a scalar reward model's ``forward_only`` logits) or a Python sequence of
    per-sample scores (a rule or a judge); both flatten to one column. ``None`` -- a judge that produced no
    parseable verdict -- is retained as ``nan`` so weighted aggregation ignores it per sample rather than
    counting a broken reward as a genuine 0.

    Raises:
        ValueError: the scorer returned a number of scores that does not match the ``n`` completions.
    """
    if isinstance(output, torch.Tensor):
        values = output.detach().float().cpu().reshape(-1)
    else:
        values = torch.tensor(
            [value if value is not None else torch.nan for value in output], dtype=torch.float32).reshape(-1)
    if values.numel() != n:
        name = getattr(scorer, '__name__', scorer.__class__.__name__)
        raise ValueError(f'reward scorer {name!r} returned {values.numel()} scores for {n} completions.')
    return values


def score_completions(completions: Sequence[str],
                      scorers: Sequence[RewardFunc],
                      columns: Optional[Dict[str, List[Any]]] = None,
                      **extra_kwargs: Any) -> torch.Tensor:
    """Score ``completions`` with every scorer -> ``[N, n_scorers]`` float tensor.

    The ONE batch-scoring entry every reward channel shares. A rule callable, a frozen scalar reward model
    (:class:`~swift.dev.rewards.scorer.ModelReward`) and a generative / API judge all meet the same
    ``__call__(completions, **columns)`` contract, so they sit in one ordered list and are scored together
    rather than through a rule path and a separate reward-model path (the two old contracts converge here).

    Concurrency follows each scorer's optional ``placement_key`` -- the DeviceGroup its forward runs on.
    Scorers on DISJOINT placements overlap on threads (:func:`~swift.dev.recipe.common.scoring.
    overlap_by_placement`); scorers that share one placement -- or declare none, like a rule -- form one
    group. Within a group every scorer is called, then any ``async def`` results are resolved with a single
    ``asyncio.gather``, so a batch of async rules still overlaps its I/O exactly as the rule-only path did.

    Args:
        completions: one completion string per sample.
        scorers: resolved reward callables -- rules from :func:`get_reward_funcs`, model / judge scorers
            from :func:`~swift.dev.rewards.scorer.build_model_reward`. A scorer may be ``async def``.
        columns: batched dataset columns ``{name: [value_per_sample, ...]}``, each aligned with
            ``completions``. Every scorer receives all of them as keyword arguments and takes the ones it
            needs (so ``messages`` reaches a judge while a rule reads ``solution``); the built-in reward
            plugins all accept ``**kwargs``.
        **extra_kwargs: additional keyword arguments forwarded to every scorer.

    Returns:
        ``[N, n_scorers]`` float tensor, columns in ``scorers`` order. A scorer returning ``None`` for a
        sample becomes ``nan`` so a broken reward is visible rather than silently scoring 0.

    Raises:
        ValueError: a column's length does not match ``completions``, or a scorer returns a wrong-length
            list.
    """
    from swift.dev.recipe.common.scoring import overlap_by_placement

    scorer_list = list(scorers)
    n = len(completions)
    rewards = torch.zeros((n, len(scorer_list)), dtype=torch.float32)
    if n == 0:
        return rewards

    columns = columns or {}
    for key, values in columns.items():
        if len(values) != n:
            raise ValueError(f'reward column {key!r} has length {len(values)} but there are {n} completions.')
    kwargs: Dict[str, Any] = {**columns, **extra_kwargs}
    completions_arg = list(completions)

    # Group scorer indices by placement_key, preserving first-appearance order so results map back
    # deterministically. A scorer with no placement_key (a rule, a sampler-reusing / API judge) reads as
    # None and shares the one "no dedicated device" group.
    order: List[Any] = []
    members: Dict[Any, List[int]] = {}
    for index, scorer in enumerate(scorer_list):
        key = getattr(scorer, 'placement_key', None)
        if key not in members:
            members[key] = []
            order.append(key)
        members[key].append(index)

    def score_group(indices: List[int]) -> List[Tuple[int, torch.Tensor]]:
        # One placement: call its scorers serially (they contend for a single device), then overlap this
        # group's awaitables with one gather, so a group of async rules keeps its I/O concurrency.
        raw = [scorer_list[i](completions_arg, **kwargs) for i in indices]
        raw = _resolve_awaitables(raw)
        return [(i, _as_score_column(out, scorer_list[i], n)) for i, out in zip(indices, raw)]

    tasks = [(key, lambda key=key: score_group(members[key])) for key in order]
    for group in overlap_by_placement(tasks):
        for index, column in group:
            rewards[:, index] = column
    return rewards


def compute_rewards_per_func(completions: Sequence[str],
                             reward_funcs: Sequence[RewardFunc],
                             columns: Optional[Dict[str, List[Any]]] = None,
                             **extra_kwargs: Any) -> torch.Tensor:
    """Score ``completions`` with every reward RULE -> ``[N, n_funcs]`` tensor.

    The rule-shaped door onto :func:`score_completions`: a rule callable has no ``placement_key``, so all
    rules land in the one "no dedicated device" group and their ``async def`` coroutines are resolved with a
    single ``asyncio.gather``. Kept as a named alias because rule-only callers (the infer channel, the rule
    tests) read clearer against it; model / judge scorers go through :func:`score_completions` directly.

    Args:
        completions: one completion string per sample.
        reward_funcs: resolved reward callables (see :func:`get_reward_funcs`). A function may be
            ``async def``; its coroutine is resolved alongside the others with one ``asyncio.gather``.
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
    return score_completions(completions, reward_funcs, columns, **extra_kwargs)


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
