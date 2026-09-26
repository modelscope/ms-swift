"""Best-of-n data synthesis: run_sampling orchestration, on twinkle's Sampler.

The simplified successor to ``swift.pipelines.sampling.SwiftSampling`` + ``VanillaSampler``. Job
unchanged: generate n candidates per prompt, score them, keep the best as positives and the worst as
the rejected response -- i.e. produce DPO-shaped training data, not just completions. That
produce-vs-consume split is why this is a separate recipe from ``run_infer`` rather than a flag on it.

The checkpointed resume is kept verbatim in spirit, because a multi-hour synthesis run that cannot
resume is unusable: ``output_file.tmp`` is the live write, ``output_file.resume`` is the snapshot
taken after each completed batch, ``ckpt_state.json`` records the last finished batch index, and the
final move to ``output_file`` is what marks a run complete.

Rule-based scoring needs no engine: ``SamplingConfig.reward_funcs`` names entries in swift's ``orms``
registry (all pure Python) or passes callables, resolved through ``swift.dev.reward``, so switching
vLLM/SGLang cannot change those scores. A MODEL reward channel (``orm_model`` / ``prm_model``) is
different -- ``_resolve_reward_model`` serves each one in one of three ways: a scalar seq_cls RM built
as a frozen twinkle model and scored by ``forward_only``; a generative (LLM-as-judge) RM that reuses
the sampling engine when it shares the sampler's model (optionally as a reward LoRA) and otherwise
builds its own; or an http(s) endpoint judged through an OpenAI-compatible client.

Two deliberate departures from legacy's scoring:
- one weighted reward list instead of separate orm/prm channels combined as ``prm + orm*10``. That
  10x was an unnamed hard-coded priority; ``reward_weights`` says the same thing explicitly.
- the ground-truth answer is no longer scored alongside the candidates. Legacy needed it as an anchor
  because it min-max normalised within each group; a plain weighted sum has no such need, and scoring
  the reference told us nothing we then used.

Also dropped: the md5 response cache (``cache_files``), and the ``client``/OpenAI teacher backend
that ``DistillSampler`` provided.
"""
from __future__ import annotations
import copy
import hashlib
import logging
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import json

if TYPE_CHECKING:
    from swift.dev.config import (
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        ModelConfig,
        PluginConfig,
        QuantizeConfig,
        RLHFConfig,
        RolloutConfig,
        SamplingConfig,
        TemplateConfig,
    )

logger = logging.getLogger(__name__)


def run_sampling(  # noqa: C901
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    sampling_config: SamplingConfig,
    generation_config: Optional[GenerationConfig] = None,
    *,
    multi_turn_config: Optional[RLHFConfig] = None,
    backend: str = 'vllm',
    engine_args: Optional[Dict[str, Any]] = None,
    distributed_config: Optional[DistributedConfig] = None,
    adapters: Optional[List[str]] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    plugin_config: Optional[PluginConfig] = None,
    rollout_config: Optional[RolloutConfig] = None,
    output_dir: str = 'output',
    _shutdown: bool = True,
) -> str:
    """Sample, score, filter, and write DPO-shaped rows. Returns the output file path.

    Returns early without touching the engine when the output file already exists and
    ``override_exist_file`` is False -- so re-running a finished job is a no-op, not a re-do.

    Args:
        backend: 'vllm' / 'sglang' / 'transformers' to sample locally, or 'client' to distil from a
            remote OpenAI-compatible teacher (legacy's ``sampler_type=distill``). With 'client' no
            model is loaded here at all; put the endpoint in ``engine_args``.
        adapters: LoRA to sample with. Ignored by the 'client' backend, which has no local weights.
    """
    from swift.dev.builders import (build_device_mesh_if_dp, build_sampler, build_template, load_model_processor,
                                    load_prompt_rows, to_sampling_params)
    from swift.dev.plugin import PluginRegistry

    # Importing the sandbox module registers the 'tool' extension point (and its built-in plugin), so a
    # user's external ``@register('tool', ...)`` finds the kind already declared when its file is
    # imported just below. Its twinkle env/tool imports are lazy, so this stays cheap when tools are off.
    import swift.dev.rollout.sandbox  # noqa: F401

    # Scoring resolves reward names below, so the run's plugin files must be imported first -- a
    # user-defined ORM named in reward_funcs is otherwise "not registered".
    PluginRegistry.load_configured(plugin_config)

    if sampling_config.n_best_to_keep >= sampling_config.num_return_sequences:
        raise ValueError(f'n_best_to_keep={sampling_config.n_best_to_keep} must be < '
                         f'num_return_sequences={sampling_config.num_return_sequences}: the lowest-scoring '
                         'candidate becomes the rejected response, so it cannot also be a positive.')
    multi_turn_enabled = bool(multi_turn_config and multi_turn_config.max_turns is not None)
    os.makedirs(output_dir, exist_ok=True)
    paths = _CheckpointPaths(output_dir, sampling_config.output_file)
    if os.path.exists(paths.final) and not sampling_config.override_exist_file:
        logger.info(f'run_sampling: {paths.final} exists and override_exist_file is False; nothing to do.')
        return paths.final

    # Resolve the model-reward channels BEFORE twinkle is initialized and the sampler is built: the
    # specs decide whether a reward model needs its own GPU DeviceGroup (which must be declared at
    # initialize time), and a generative judge that reuses the sampler needs its reward LoRA resident in
    # the engine's adapter list at construction time (vLLM/SGLang size their adapter slots up front).
    # The channels themselves are constructed after the sampler, since a judge scores through it.
    sampler_model = model_config.model
    orm_spec = _resolve_reward_model(
        sampling_config.orm_model, sampling_config.orm_adapter, sampler_model, backend)
    prm_spec = _resolve_reward_model(
        sampling_config.prm_model, sampling_config.prm_adapter, sampler_model, backend)
    reuse_adapters = [spec.adapter for spec in (orm_spec, prm_spec)
                      if spec is not None and spec.kind == 'generative_reuse' and spec.adapter]

    # A reward model that keeps its own weights resident on GPU (a scalar seq_cls RM, or a generative
    # judge on a separate model) needs a dedicated DeviceGroup so it does not share the sampler's cards,
    # and only mode='ray' can express that placement. A judge reusing the sampler, or an http(s)
    # endpoint, needs no extra device and is fine in local mode.
    reward_groups = _exclusive_reward_groups(orm_spec, prm_spec)
    if reward_groups and (distributed_config is None or distributed_config.mode != 'ray'):
        raise ValueError(
            f'reward channel(s) {reward_groups} keep their own model resident on GPU and need a dedicated '
            "DeviceGroup, which only DistributedConfig.mode='ray' can place. Run with mode='ray' (and "
            'nproc_per_node), or use a sampler-reusing generative judge / an http(s) endpoint, which need '
            'no extra device.')

    if distributed_config is not None:
        _initialize_for_sampling(distributed_config, reward_groups)

    rows = load_prompt_rows(dataset_config, None, split_dataset_ratio=0.0)
    if not rows:
        raise ValueError('run_sampling got an empty dataset. Set DatasetConfig.dataset or .val_dataset.')

    cache = _CandidateCache(sampling_config.cache_files)

    quant_method = getattr(quantize_config, 'quant_method', None)
    if backend in {'client', 'no'} and quant_method is not None:
        raise ValueError(
            f'quant_method={quant_method!r} cannot affect sampler_engine={backend!r}, which loads no local model. '
            'Configure quantization on the remote server or omit --quant_method.')
    template = None
    if backend == 'client':
        sampler = _ClientSampler(**(engine_args or {}))
    elif backend == 'no':
        sampler = None
    else:
        _, processor = load_model_processor(model_config)
        template = build_template(template_config, processor)
        # A generative judge reusing this sampler scores through a reward LoRA, which must be resident
        # before the first request, so merge it into the sampling adapters the engine is built with.
        sampler_adapters = list(adapters or [])
        for reward_adapter in reuse_adapters:
            if reward_adapter not in sampler_adapters:
                sampler_adapters.append(reward_adapter)
        sampler = build_sampler(
            model_config,
            backend=backend,
            engine_args=engine_args,
            device_mesh=build_device_mesh_if_dp(distributed_config),
            template=template,
            adapters=sampler_adapters or None,
            remote_group=_SAMPLER_GROUP if distributed_config is not None
            and distributed_config.mode == 'ray' else None,
            quantize_config=quantize_config)

    channels = _build_channels(
        sampling_config,
        orm_spec=orm_spec,
        prm_spec=prm_spec,
        sampler=sampler,
        model_config=model_config,
        template_config=template_config,
        backend=backend,
        engine_args=engine_args,
        distributed_config=distributed_config,
        quantize_config=quantize_config)

    rollout = None
    if sampler is not None:
        # One generation entry for single- and multi-turn alike: a RolloutEngine wrapping the already-built
        # sampler (injected, so this recipe keeps owning its lifecycle and calls close(), not shutdown()).
        # Single-turn calls sampler.sample and rebuilds the training feature; multi-turn is configured below
        # and drives twinkle's native engine. With backend='client' the teacher owns chat formatting and
        # exposes no local template, so template is None and both paths run message-only: no token IDs,
        # just the teacher's finished text per turn.
        from swift.dev.rollout import RolloutEngine
        rollout = RolloutEngine(sampler=sampler, template=template)
        if multi_turn_enabled:
            # Tools are opt-in: with RolloutConfig.tools set, a sandbox env pool is built and the tool
            # plugins resolved so each episode leases its own env; otherwise the rollout has no tools.
            from swift.dev.rollout.sandbox import build_tool_sandbox
            env_pool, tool_plugins = build_tool_sandbox(rollout_config)
            rollout.configure_multi_turn(
                max_turns=multi_turn_config.max_turns,
                max_trajectory_tokens=multi_turn_config.max_trajectory_tokens,
                env_pool=env_pool,
                tool_plugins=tool_plugins)

    batches = _plan_batches(len(rows), sampling_config.batch_size, sampling_config.max_batches)
    resume_from, write_mode = paths.prepare(sampling_config.resume)
    params = to_sampling_params(generation_config, num_samples=sampling_config.num_return_sequences)

    # Rollout-token sidecar: only built when save_rollout_tokens is on, and only a local backend can feed
    # it (a message-only client has no tokens; validate rejects that combination). The NPZ dir sits beside
    # the jsonl and each row embeds a path relative to output_dir, so the pair stays portable.
    recorder = None
    if sampling_config.save_rollout_tokens:
        from swift.dev.rollout.recorder import RolloutRecorder
        recorder = RolloutRecorder(paths.final + '.rollout_tokens', base_dir=output_dir, enabled=True)

    try:
        with open(paths.tmp, write_mode, encoding='utf-8') as f:
            for index, (start, end) in enumerate(batches):
                if index <= resume_from:
                    continue
                batch = rows[start:end]
                logger.info(f'run_sampling: batch {index + 1}/{len(batches)} ({len(batch)} prompts)')
                for line in _sample_batch(sampler, rollout, batch, params, template_config, channels, sampling_config,
                                          cache, backend, multi_turn_enabled, recorder):
                    f.write(line)
                f.flush()
                paths.checkpoint(index)
    finally:
        channels.shutdown()
        if rollout is not None:
            rollout.close()
        if _shutdown and sampler is not None:
            sampler.shutdown()

    paths.finalize()
    logger.info(f'run_sampling: wrote {paths.final}')
    return paths.final


#: Name of the twinkle DeviceGroup the sampling engine is placed in under mode='ray'.
_SAMPLER_GROUP = 'sampler'

#: Reward-model kinds that keep their own weights resident on GPU and so need a dedicated DeviceGroup.
#: A ``generative_reuse`` judge scores through the sampler's engine, and an ``api`` judge has no local
#: model -- neither needs its own cards.
_EXCLUSIVE_REWARD_KINDS = frozenset({'scalar', 'generative_independent'})


def _exclusive_reward_groups(orm_spec: Optional['_RewardModelSpec'],
                             prm_spec: Optional['_RewardModelSpec']) -> List[str]:
    """The DeviceGroup names the GPU-resident reward channels need, in declaration order ('orm', 'prm')."""
    groups: List[str] = []
    for spec, name in ((orm_spec, 'orm'), (prm_spec, 'prm')):
        if spec is not None and spec.kind in _EXCLUSIVE_REWARD_KINDS:
            groups.append(name)
    return groups


def plan_sampling_device_groups(nproc_per_node: Optional[int],
                                reward_group_names: List[str]) -> Tuple[List[Tuple[str, List[int]]], int]:
    """Plan the twinkle DeviceGroups for a sampling run. Pure function (no twinkle import).

    The sampler occupies ``[0, nproc_per_node)``; each GPU-resident reward channel gets its own
    disjoint ``nproc_per_node``-wide block after it, so a run with R reward groups needs
    ``nproc_per_node * (1 + R)`` GPUs. Mirrors ``plan_rl_device_groups`` in run_grpo: the reward group
    is sized off the same world size the sampler uses rather than a separate GPU-count knob.

    Returns ``(groups, total_ranks)`` where ``groups`` is a list of ``(name, ranks)`` to hand
    twinkle.initialize and ``total_ranks`` is the summed GPU count.
    """
    if nproc_per_node is None or nproc_per_node < 1:
        raise ValueError(f'nproc_per_node must be >= 1 (the sampler GPU count), got {nproc_per_node!r}.')
    groups: List[Tuple[str, List[int]]] = [(_SAMPLER_GROUP, list(range(nproc_per_node)))]
    start = nproc_per_node
    for name in reward_group_names:
        groups.append((name, list(range(start, start + nproc_per_node))))
        start += nproc_per_node
    return groups, start


def _initialize_for_sampling(distributed_config: DistributedConfig,
                            reward_groups: Optional[List[str]] = None) -> None:
    """Initialize twinkle, giving the sampler -- and each GPU-resident reward model -- its own DeviceGroup.

    A scalar seq_cls/reranker RM (built through ``build_frozen_reward_model`` -> ``build_model``) and an
    independent generative judge keep their weights on GPU, so under mode='ray' each is placed in its own
    disjoint DeviceGroup (see :func:`plan_sampling_device_groups`) rather than sharing the sampler's
    cards; ``build_model`` / ``build_sampler`` target the group by the same name via ``remote_group``.
    A sampler-reusing judge and an http(s) judge need no device and declare no group. Local mode has no
    groups to declare (and the caller forbids a GPU-resident reward there), so it initializes plainly.
    """
    import twinkle
    from twinkle import DeviceGroup

    if distributed_config.mode != 'ray':
        twinkle.initialize(mode='local')
        return
    nproc = distributed_config.nproc_per_node
    if nproc is None:
        raise ValueError("DistributedConfig.nproc_per_node is required in mode='ray' (it sizes the sampler's "
                         'Ray DeviceGroup and each reward group). Pass it explicitly -- there is no default.')
    planned, total_ranks = plan_sampling_device_groups(nproc, reward_groups or [])
    twinkle.initialize(
        mode='ray',
        nproc_per_node=total_ranks,
        groups=[
            DeviceGroup(name=name, ranks=ranks, device_type='GPU', gpus_per_worker=1) for name, ranks in planned
        ])


def _build_channels(sampling_config: SamplingConfig,
                    *,
                    orm_spec: Optional['_RewardModelSpec'] = None,
                    prm_spec: Optional['_RewardModelSpec'] = None,
                    sampler: Any = None,
                    model_config: Optional[ModelConfig] = None,
                    template_config: Optional[TemplateConfig] = None,
                    backend: str = 'vllm',
                    engine_args: Optional[Dict[str, Any]] = None,
                    distributed_config: Optional[DistributedConfig] = None,
                    quantize_config: Optional[QuantizeConfig] = None) -> '_RewardChannels':
    """Resolve reward channels from the configured ORM and PRM funcs plus any model-reward specs.

    The model-reward channels arrive as pre-resolved specs (see :func:`_resolve_reward_model`) rather
    than being resolved here, because a generative judge that reuses the sampler needs the sampler
    already built. Each spec is turned into its callable -- a scalar seq_cls RM scored by forward, a
    generative judge (reusing the sampler or owning its own engine), or an API judge -- by
    :func:`_build_model_reward`.
    """
    from swift.dev.reward import get_reward_funcs

    orm_funcs, orm_names = get_reward_funcs(sampling_config.reward_funcs, sampling_config.reward_config)
    prm_funcs, prm_names = get_reward_funcs(sampling_config.prm_funcs, sampling_config.reward_config)
    ray = distributed_config is not None and distributed_config.mode == 'ray'
    for spec, funcs, names, group in ((orm_spec, orm_funcs, orm_names, 'orm'), (prm_spec, prm_funcs, prm_names, 'prm')):
        if spec is None:
            continue
        # A GPU-resident reward model targets the DeviceGroup declared for its channel, and only under
        # ray (local mode has no groups); a sampler-reusing judge or an API judge stays group-less.
        remote_group = group if ray and spec.kind in _EXCLUSIVE_REWARD_KINDS else None
        funcs.append(
            _build_model_reward(
                spec,
                sampling_config,
                sampler=sampler,
                model_config=model_config,
                template_config=template_config,
                backend=backend,
                engine_args=engine_args,
                distributed_config=distributed_config,
                quantize_config=quantize_config,
                remote_group=remote_group))
        names.append(spec.display_name)
    channels = _RewardChannels(orm_funcs, orm_names, prm_funcs, prm_names)
    if channels.empty:
        logger.info('run_sampling: no reward funcs -- every candidate is emitted as a positive.')
    else:
        logger.info(f'run_sampling: orm={orm_names} (x{sampling_config.orm_channel_weight}) prm={prm_names}, '
                    f'normalize={sampling_config.normalize_rewards}')
    return channels


class _ClientSampler:
    """Sample from a remote OpenAI-compatible teacher, i.e. legacy's ``DistillSampler``.

    Returns twinkle's ``SampleResponse``/``SampledSequence`` -- the slice of the Sampler interface both
    callers read: the single-turn path takes ``.sequences[].decoded``, and twinkle's native
    ``MultiTurnRollout`` drives message-only multi-turn off the same objects. So distillation is a
    backend choice rather than a second code path through the recipe. It is NOT a full twinkle Sampler:
    there is no local model, no device mesh, and no template -- the server owns the chat formatting, and
    the sequences carry text and structured tool calls but no token ids.

    Reasoning models are handled the way legacy did: when the response carries ``reasoning_content``
    separately from ``content``, the two are recombined as
    ``<think>{reasoning}</think>\\n\\n<answer>{content}</answer>``. Dropping the reasoning would train
    the student on conclusions without the derivation, which is the opposite of the point.
    """

    def __init__(self,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 max_workers: int = 8,
                 concurrency: Optional[int] = None,
                 timeout: float = 600.0,
                 max_retries: Optional[int] = None,
                 **client_kwargs: Any):
        """
        Args:
            base_url / api_key / model: the teacher endpoint; ``model`` is required.
            max_workers: local thread-pool size -- how many prompts are issued in parallel.
            concurrency: cap on requests in flight against the endpoint, enforced by twinkle's OpenAI
                protocol client (a per-instance semaphore), or None for no cap. This is the traffic
                knob: the pool wants to be large, a provider's quota wants to be small, and only the
                endpoint knows the latter -- so the two are deliberately separate.
            timeout / max_retries: forwarded to the protocol client (per-request timeout, and the SDK's
                own transient-error retries -- 429 / 5xx / timeouts -- with exponential backoff).
            client_kwargs: anything else the ``openai`` constructor accepts.
        """
        from twinkle import requires
        requires('openai')
        from twinkle_agentic.protocol.openai import OpenAI

        if not model:
            raise ValueError("backend='client' needs the teacher's model name in "
                             "engine_args, e.g. engine_args={'model': 'deepseek-reasoner', "
                             "'base_url': ..., 'api_key': ...}.")
        self.api = OpenAI(
            model=model,
            api_key=api_key or os.environ.get('OPENAI_API_KEY'),
            base_url=base_url,
            concurrency=concurrency,
            timeout=timeout,
            max_retries=max_retries,
            client_kwargs=client_kwargs)
        self.max_workers = max_workers

    def sample(self, inputs: List[Dict[str, Any]], sampling_params: Any = None, **kwargs) -> List[Any]:
        """Query the teacher once per prompt, concurrently.

        Returns twinkle's ``List[SampleResponse]`` -- one per prompt, each carrying ``num_samples``
        ``SampledSequence``s -- so the one object serves both callers: the single-turn path reads
        ``.sequences[].decoded``, and twinkle's native ``MultiTurnRollout`` (whose default response
        callback wants exactly one response holding one sequence) drives message-only multi-turn off
        the same return. Threads rather than asyncio: the calls are network-bound and the recipe's loop
        is synchronous, so a pool keeps the local parallelism while the protocol client caps endpoint
        traffic. A failed request degrades to a single empty ``error`` sequence -- a remote teacher
        rate-limiting one prompt must not end a long distillation run, and the recipe drops empty
        candidates downstream.
        """
        from concurrent.futures import ThreadPoolExecutor

        from twinkle.data_format import SamplingParams
        from twinkle.data_format.sampling import SampleResponse

        params = sampling_params if sampling_params is not None else SamplingParams()
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            groups = list(pool.map(lambda item: self._one(item, params), inputs))
        return [SampleResponse(sequences=group) for group in groups]

    def _one(self, trajectory: Dict[str, Any], sampling_params: Any) -> List[Any]:
        from twinkle.data_format.sampling import SampledSequence

        messages = list(trajectory.get('messages') or [])
        tools = trajectory.get('tools')
        request: Dict[str, Any] = {'messages': messages}
        if tools:
            request['tools'] = tools
        try:
            reply = self.api(request, sampling_params)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f'teacher request failed, dropping this prompt: {exc}')
            return [SampledSequence(stop_reason='error', tokens=[], logprobs=None, decoded='')]
        # The protocol client returns one Message when num_samples==1, else a list. Each is an
        # OpenAI-shaped dict whose tool_calls are already in the {'id', 'type', 'function'} shape
        # twinkle dispatches on, so they pass straight into the assistant turn.
        sequences = []
        for message in (reply if isinstance(reply, list) else [reply]):
            text = _join_reasoning(message)
            assistant: Dict[str, Any] = {'role': 'assistant', 'content': text}
            tool_calls = message.get('tool_calls')
            if tool_calls:
                assistant['tool_calls'] = tool_calls
            # The teacher exposes no token ids, so the assistant turn is carried structurally: the
            # message-level ledger adopts these messages wholesale, which is how the reply's tool_calls
            # reach twinkle's loop (a text-only backend has no markup for it to parse).
            feature: Dict[str, Any] = {'messages': messages + [assistant]}
            if tools:
                feature['tools'] = tools
            sequences.append(
                SampledSequence(
                    stop_reason=_stop_reason(message.get('finish_reason')),
                    tokens=[],
                    logprobs=None,
                    decoded=text,
                    new_input_feature=feature))
        return sequences

    def shutdown(self) -> None:
        """Nothing to release: the protocol client holds no GPU and closes its own connections."""


def _join_reasoning(message: Dict[str, Any]) -> str:
    """Recombine a reasoning model's split output into one trainable string.

    ``message`` is a twinkle protocol ``Message`` dict, which keeps ``reasoning_content`` separate from
    ``content``; dropping the reasoning would train the student on conclusions without the derivation.
    """
    content = message.get('content') or ''
    reasoning = message.get('reasoning_content')
    if not reasoning:
        return content
    return f'<think>{reasoning}</think>\n\n<answer>{content}</answer>'


def _stop_reason(finish_reason: Optional[str]) -> str:
    """Map OpenAI's ``finish_reason`` onto twinkle's StopReason.

    Only ``'length'`` matters downstream -- it flags a turn cut off mid-generation -- so everything
    else (``'stop'``, ``'tool_calls'``, ``None``) collapses to ``'stop'``.
    """
    return 'length' if finish_reason == 'length' else 'stop'


@dataclass
class _Candidate:
    """One generated candidate, optionally carrying its full trajectory and rollout tokens."""

    text: str
    messages: Optional[List[Dict[str, Any]]] = None
    rollout_infos: Optional[Dict[str, Any]] = None
    truncated: bool = False
    multi_turn: bool = False
    # Rollout token payload, empty for a message-only backend or when tokens were not kept. The recorder
    # (save_rollout_tokens) writes these to an NPZ sidecar; the jsonl row then carries the relative path.
    encoded: Optional[Dict[str, Any]] = None
    response_token_ids: List[int] = field(default_factory=list)
    rollout_logprobs: List[float] = field(default_factory=list)
    response_loss_mask: List[int] = field(default_factory=list)


def _candidate_from_sample(sample: Any, *, multi_turn: bool) -> _Candidate:
    """Adopt a RolloutSample's payload into a _Candidate (single- and multi-turn alike).

    Both generation paths now yield the same RolloutSample, so one adapter feeds _Candidate: the token
    half (encoded / response_token_ids / rollout_logprobs / response_loss_mask) is empty for a
    message-only backend and populated for a local one, which is what the recorder persists.
    """
    return _Candidate(
        sample.decoded,
        messages=copy.deepcopy(sample.messages) if sample.messages is not None else None,
        rollout_infos=copy.deepcopy(sample.rollout_infos) if sample.rollout_infos else None,
        truncated=sample.truncated,
        multi_turn=multi_turn,
        encoded=dict(sample.encoded) if sample.encoded else None,
        response_token_ids=list(sample.response_token_ids or []),
        rollout_logprobs=list(sample.rollout_logprobs or []),
        response_loss_mask=list(sample.response_loss_mask or []))


def _sample_batch(
    sampler: Any,
    rollout: Any,
    batch: List[Dict[str, Any]],
    params: Any,
    template_config: TemplateConfig,
    channels: '_RewardChannels',
    sampling_config: SamplingConfig,
    cache: '_CandidateCache',
    backend: str,
    multi_turn_enabled: bool,
    recorder: Any = None,
) -> List[str]:
    """One batch of prompts -> the jsonl lines it contributes.

    Kept whole (sample + score + emit) so the caller's loop stays purely about checkpointing: a batch
    is either fully written and snapshotted or not written at all.

    Prompts already covered by ``cache_files`` skip generation entirely -- they are the expensive part,
    and their candidates are reused verbatim.
    """
    from swift.dev.builders import split_prompt_and_reference

    trajectories, ground_truths = split_prompt_and_reference(batch, template_config)
    wanted = sampling_config.num_return_sequences
    cached = [cache.lookup(trajectory, wanted, multi_turn=multi_turn_enabled) for trajectory in trajectories]

    to_sample = [index for index, hit in enumerate(cached) if hit is None]
    candidates: List[List[_Candidate]] = [hit or [] for hit in cached]
    if to_sample:
        if sampler is None:
            raise ValueError("sampler_engine='no' requires cache_files to cover every input prompt.")
        if rollout is None:
            raise RuntimeError('sampling was requested but no rollout engine was built.')
        selected_trajectories = [trajectories[i] for i in to_sample]
        prompt_extras = []
        for index, trajectory in zip(to_sample, selected_trajectories):
            extra = {key: copy.deepcopy(value) for key, value in batch[index].items() if key != 'messages'}
            extra.update({
                key: copy.deepcopy(value)
                for key, value in trajectory.items() if key != 'messages'
            })
            prompt_extras.append(extra)
        # One generation call for single- and multi-turn alike: RolloutEngine.generate dispatches on whether
        # a multi-turn engine was configured, and both paths yield the same RolloutSample. logprobs are
        # forced so every sample can carry its rollout tokens (the recorder persists them when
        # save_rollout_tokens is on); ``strict`` is a transformers single-turn sampler kwarg, passed
        # through -- the multi-turn engine drives its own per-turn sampling and ignores it.
        sample_kwargs: Dict[str, Any] = {'strict': sampling_config.strict} if backend == 'transformers' else {}
        samples = rollout.generate(
            [trajectory['messages'] for trajectory in selected_trajectories],
            num_samples=wanted,
            sampling_params=asdict(params),
            prompt_extras=prompt_extras,
            force_logprobs=True,
            **sample_kwargs)
        expected = len(to_sample) * wanted
        if len(samples) != expected:
            raise RuntimeError(f'rollout returned {len(samples)} samples, expected {expected}.')
        fresh = []
        for offset in range(0, expected, wanted):
            fresh.append([
                _candidate_from_sample(sample, multi_turn=multi_turn_enabled)
                for sample in samples[offset:offset + wanted]
            ])
        for index, group in zip(to_sample, fresh):
            candidates[index] = group
    if len(to_sample) < len(batch):
        logger.info(f'run_sampling: reused cached candidates for {len(batch) - len(to_sample)}/{len(batch)} prompts')

    lines: List[str] = []
    for row, trajectory, ground_truth, group in zip(batch, trajectories, ground_truths, candidates):
        lines.extend(_emit_rows(row, trajectory, ground_truth, group, channels, sampling_config, recorder))
    return lines


def _candidate_messages(candidate: _Candidate, trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    if candidate.messages is not None:
        return copy.deepcopy(candidate.messages)
    return copy.deepcopy(trajectory['messages']) + [{'role': 'assistant', 'content': candidate.text}]


def _emit_rows(
    row: Dict[str, Any],
    trajectory: Dict[str, Any],
    ground_truth: Optional[str],
    candidates: List[_Candidate],
    channels: '_RewardChannels',
    sampling_config: SamplingConfig,
    recorder: Any = None,
) -> List[str]:
    """One prompt's candidates -> the jsonl lines it contributes (possibly none).

    Without any reward func every candidate is a positive and no rejected trajectory is written: there
    is no ranking, so naming one candidate worse than another would be a fabrication.

    When ``recorder`` is set (save_rollout_tokens), each emitted candidate's tokens go to an NPZ named by
    ``(prompt_id, candidate index)`` -- both deterministic across a resumed replay, so the same row points
    at the same file. The path and reward score ride along in the row; with the recorder off neither is
    added and the schema is unchanged.
    """
    candidates = [candidate for candidate in candidates if candidate.text]
    if not candidates:
        return []

    # prompt_id is the group id _dpo_line stamps as out['id'], so the NPZ name and the row agree.
    prompt_id = _prompt_key(trajectory['messages'])
    keep_tokens = recorder is not None
    record = recorder.record if keep_tokens else None

    if channels.empty:
        lines = []
        for index, positive in enumerate(candidates):
            token_path = record(prompt_id, index, positive) if keep_tokens else None
            lines.append(_dpo_line(row, trajectory, positive, None, ground_truth, rollout_tokens=token_path))
        return lines

    # The ground truth is scored in the same group as the candidates, so normalisation sees it as the
    # anchor -- but it is never emitted as a positive, since it is not something the model produced.
    scored = list(candidates)
    if sampling_config.score_ground_truth and ground_truth:
        scored.append(_Candidate(ground_truth, messages=_candidate_messages(_Candidate(ground_truth), trajectory)))
    scores = channels.score(scored, row, trajectory, sampling_config)[:len(candidates)]

    threshold = sampling_config.reward_threshold
    keep = [i for i in range(len(candidates)) if threshold is None or scores[i] > threshold]
    if not keep:
        return []
    if _is_too_easy(len(keep), len(candidates), sampling_config.easy_query_threshold):
        return []

    ranked = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    negative_index = ranked[-1]
    positives = [i for i in ranked[:sampling_config.n_best_to_keep] if i in keep and i != negative_index]
    if not positives:
        return []
    negative = candidates[negative_index]
    logger.debug(f'scores={[round(s, 4) for s in scores]} positives={positives} negative={negative_index}')
    rejected_tokens = record(prompt_id, negative_index, negative) if keep_tokens else None
    lines = []
    for i in positives:
        lines.append(
            _dpo_line(
                row,
                trajectory,
                candidates[i],
                negative,
                ground_truth,
                rollout_tokens=record(prompt_id, i, candidates[i]) if keep_tokens else None,
                reward_score=scores[i] if keep_tokens else None,
                rejected_rollout_tokens=rejected_tokens,
                rejected_reward_score=scores[negative_index] if keep_tokens else None))
    return lines


class _RewardChannels:
    """The ORM and PRM scoring channels, and how their two scores become one.

    Legacy kept them apart and combined them as ``prm + orm * 10``. That 10 was an unnamed priority --
    it made any ORM difference dominate every PRM difference. Here it is ``orm_channel_weight``, so the
    legacy ranking is reproducible by setting it to 10 and the default (1.0) is an honest sum.
    """

    def __init__(self,
                 orm_funcs: List[Any],
                 orm_names: List[str],
                 prm_funcs: List[Any],
                 prm_names: List[str]):
        self.orm_funcs = orm_funcs
        self.orm_names = orm_names
        self.prm_funcs = prm_funcs
        self.prm_names = prm_names

    @property
    def empty(self) -> bool:
        return not self.orm_funcs and not self.prm_funcs

    def score(self, candidates: List[_Candidate], row: Dict[str, Any], trajectory: Dict[str, Any],
              sampling_config: SamplingConfig) -> List[float]:
        """Score one prompt's candidates -> one float each.

        Dataset columns are broadcast across the candidates. ``messages`` is different: each reward
        receives the complete trajectory that produced that candidate, including intermediate turns.
        """
        total = [0.0] * len(candidates)
        if self.orm_funcs:
            orm = self._channel(
                candidates, row, trajectory, self.orm_funcs, sampling_config.reward_weights, sampling_config)
            total = [t + sampling_config.orm_channel_weight * value for t, value in zip(total, orm)]
        if self.prm_funcs:
            prm = self._channel(
                candidates, row, trajectory, self.prm_funcs, sampling_config.prm_weights, sampling_config)
            total = [t + value for t, value in zip(total, prm)]
        return total

    @staticmethod
    def _channel(candidates: List[_Candidate], row: Dict[str, Any], trajectory: Dict[str, Any], funcs: List[Any],
                 weights: Optional[List[float]], sampling_config: SamplingConfig) -> List[float]:
        from swift.dev.reward import compute_rewards_per_func, weight_rewards

        columns = {key: [value] * len(candidates) for key, value in row.items() if key != 'messages'}
        columns['messages'] = [_candidate_messages(candidate, trajectory) for candidate in candidates]
        columns['rollout_infos'] = [copy.deepcopy(candidate.rollout_infos or {}) for candidate in candidates]
        columns['truncated'] = [candidate.truncated for candidate in candidates]
        rewards_per_func = compute_rewards_per_func([candidate.text for candidate in candidates], funcs, columns)
        scores = weight_rewards(rewards_per_func, weights).tolist()
        # Normalise per channel, before the channels are added: doing it after would let the channel
        # with the larger raw range decide the ranking regardless of the weights.
        return _normalize(scores) if sampling_config.normalize_rewards else scores

    def shutdown(self) -> None:
        for func in self.orm_funcs + self.prm_funcs:
            shutdown = getattr(func, 'shutdown', None)
            if shutdown is not None:
                shutdown()


#: Reward-model ``task_type``s that score by a pooling/classification forward (a scalar per candidate).
_SCALAR_TASK_TYPES = frozenset({'seq_cls', 'reranker'})
#: Reward-model ``task_type``s that are causal_lm-shaped and can be prompted as an LLM-as-judge.
_GENERATIVE_TASK_TYPES = frozenset({'causal_lm', 'generative_reranker'})

#: Upper bound on a judge's generation. Small on purpose: the verdict is a short analysis plus a
#: ``Reward: <score>`` line, and capping it keeps a chatty judge from dominating the run's cost.
_JUDGE_MAX_NEW_TOKENS = 512

#: Built-in LLM-as-judge prompt, overridable via ``SamplingConfig.judge_template``. A ``str.format``
#: template rendered with ``prompt=`` (the conversation that produced the candidate) and
#: ``completion=`` (the candidate itself). It follows swift's ``GenRMPlugin`` convention -- the judge
#: is asked to end with ``Reward: <score>`` in [0, 1] -- so ``_parse_judge_score`` reads it back the
#: same way. It deliberately carries no literal braces beyond the two placeholders, since
#: ``str.format`` would treat any other ``{...}`` as a field.
_DEFAULT_JUDGE_TEMPLATE = (
    'You are a strict reward model. Based on the dialogue history, analyze in detail whether the '
    "candidate response is accurate, complete, and relevant.\n"
    'Assign a reward score between 0 and 1, where 0 indicates completely incorrect and 1 indicates '
    'fully correct. Before finishing your response, assign the reward using the following format:\n\n'
    'Reward: <score>\n\n'
    'For example:\nReward: 0.85\n\n'
    '=== Dialogue history ===\n{prompt}\n\n'
    '=== Candidate response ===\n{completion}\n\n'
    'Now analyze the candidate response and end with a single line "Reward: <score>".')


@dataclass
class _RewardModelSpec:
    """How one model reward channel (``orm_model`` / ``prm_model``) will be served.

    Resolved before the sampler is built (see :func:`_resolve_reward_model`) so a generative judge's
    reward LoRA can be merged into the sampler's adapter list at construction time.

    ``kind`` is one of:
    - ``scalar``: a seq_cls/reranker RM, built frozen and scored by forward logits (:class:`_ModelReward`).
    - ``generative_reuse``: a causal_lm judge sharing the sampler's model, scored through that sampler.
    - ``generative_independent``: a causal_lm judge on its own model, scored through its own engine.
    - ``api``: an http(s) endpoint judged through an OpenAI-compatible client (:class:`_ApiJudgeReward`).
    """

    kind: str
    model_id: Optional[str] = None
    adapter: Optional[str] = None
    task_type: Optional[str] = None

    @property
    def display_name(self) -> str:
        return self.model_id or self.adapter or 'judge'


def _resolve_reward_model(value: Optional[str], adapter: Optional[str], sampler_model: Optional[str],
                          backend: str) -> Optional[_RewardModelSpec]:
    """Decide HOW a model reward channel is served, from its value plus the sampling backend.

    Three facts drive the choice, and their order matters:
    - An http(s) URL is an API judge. This is checked FIRST because ``get_model_info_meta`` would try
      to resolve (and download) the value as a model id, which a URL is not.
    - A value's real ``task_type`` comes from ``get_model_info_meta``: seq_cls/reranker -> a scalar RM;
      causal_lm/generative_reranker -> a generative judge. This is what tells a scalar RM (which cannot
      reuse a generation sampler -- different model class, no LM head to prompt) from a generative one.
    - ``value is None`` with an ``adapter`` means "reuse the sampler's own model as a generative judge,
      plus this reward LoRA"; with neither, there is no model channel at all.

    A generative judge reuses the sampler only when its base is the sampler's own model AND the sampler
    is local (``backend`` not 'client'/'no', which load no local engine); otherwise it builds its own.
    A scalar RM and an independent generative judge are always built locally, regardless of backend --
    the 'client' sampling backend only means the SAMPLER is remote, not that no local model may exist.
    """
    local_backend = backend not in {'client', 'no'}
    if value is None:
        if adapter is None:
            return None
        if not local_backend:
            raise ValueError(f'orm_adapter/prm_adapter={adapter!r} without a matching *_model means "reuse the '
                             f"sampler's model as a generative judge\", but backend={backend!r} loads no local "
                             'sampler to reuse. Point *_model at a local generative model or an http(s) endpoint.')
        return _RewardModelSpec('generative_reuse', model_id=sampler_model, adapter=adapter, task_type='causal_lm')
    if value.startswith('http://') or value.startswith('https://'):
        return _RewardModelSpec('api', model_id=value, adapter=adapter)
    from swift.model import get_model_info_meta

    model_info, _ = get_model_info_meta(value)
    task_type = model_info.task_type or 'causal_lm'
    if task_type in _SCALAR_TASK_TYPES:
        return _RewardModelSpec('scalar', model_id=value, adapter=adapter, task_type=task_type)
    if task_type in _GENERATIVE_TASK_TYPES:
        if local_backend and value == sampler_model:
            return _RewardModelSpec('generative_reuse', model_id=value, adapter=adapter, task_type=task_type)
        return _RewardModelSpec('generative_independent', model_id=value, adapter=adapter, task_type=task_type)
    raise ValueError(f'reward model {value!r} resolved to task_type={task_type!r}, which is neither a scalar '
                     f'RM {sorted(_SCALAR_TASK_TYPES)} nor a generative judge {sorted(_GENERATIVE_TASK_TYPES)}.')


def _build_model_reward(spec: '_RewardModelSpec',
                        sampling_config: SamplingConfig,
                        *,
                        sampler: Any = None,
                        model_config: Optional[ModelConfig] = None,
                        template_config: Optional[TemplateConfig] = None,
                        backend: str = 'vllm',
                        engine_args: Optional[Dict[str, Any]] = None,
                        distributed_config: Optional[DistributedConfig] = None,
                        quantize_config: Optional[QuantizeConfig] = None,
                        remote_group: Optional[str] = None) -> Any:
    """Turn a resolved :class:`_RewardModelSpec` into a ``func(completions, **columns)`` callable."""
    from swift.dev.builders import to_sampling_params

    if spec.kind == 'scalar':
        return _ModelReward(spec.model_id,
                            model_config,
                            template_config,
                            adapter=spec.adapter,
                            distributed_config=distributed_config,
                            remote_group=remote_group)
    if spec.kind == 'api':
        return _ApiJudgeReward(
            spec.model_id, model=(engine_args or {}).get('model'), judge_template=sampling_config.judge_template)
    # Generative judge: greedy decoding so the verdict is deterministic across a run.
    judge_params = to_sampling_params(None, temperature=0.0, max_tokens=_JUDGE_MAX_NEW_TOKENS)
    if spec.kind == 'generative_reuse':
        return _GenerativeJudgeReward(
            sampler, judge_template=sampling_config.judge_template, adapter_path=spec.adapter, params=judge_params)
    judge_sampler = _build_judge_sampler(
        spec, model_config, template_config, backend, engine_args, distributed_config, quantize_config,
        remote_group)
    return _GenerativeJudgeReward(
        judge_sampler,
        judge_template=sampling_config.judge_template,
        adapter_path=spec.adapter,
        params=judge_params,
        owns_sampler=True)


def _build_judge_sampler(spec: '_RewardModelSpec', model_config: ModelConfig, template_config: TemplateConfig,
                         backend: str, engine_args: Optional[Dict[str, Any]],
                         distributed_config: Optional[DistributedConfig],
                         quantize_config: Optional[QuantizeConfig],
                         remote_group: Optional[str] = None) -> Any:
    """Build a standalone generative judge engine for a judge that cannot reuse the sampler.

    A copy of the run's ModelConfig repointed at the judge's own model, with a generation ``task_type``
    (never a pooling one -- that would build an embedding/classify engine with no LM head to prompt).
    The 'client'/'no' sampling backends have no local engine to inherit a backend from, so a real
    sampler backend is substituted. Under ray it is placed in its channel's own reward DeviceGroup
    (``remote_group``) so it does not collide with the sampler; in local mode there is no group to
    target and it shares the process's devices.
    """
    from swift.dev.builders import build_device_mesh_if_dp, build_sampler, build_template, load_model_processor

    judge_backend = backend if backend in {'vllm', 'sglang', 'transformers'} else 'vllm'
    judge_model_config = copy.copy(model_config)
    judge_model_config.model = spec.model_id
    judge_model_config.task_type = spec.task_type or 'causal_lm'
    _, processor = load_model_processor(judge_model_config)
    judge_template = build_template(template_config, processor)
    return build_sampler(
        judge_model_config,
        backend=judge_backend,
        engine_args=engine_args,
        device_mesh=build_device_mesh_if_dp(distributed_config),
        template=judge_template,
        adapters=[spec.adapter] if spec.adapter else None,
        remote_group=remote_group,
        quantize_config=quantize_config)


def _render_conversation(messages: Optional[List[Dict[str, Any]]]) -> str:
    """Flatten a message list into ``Role: content`` lines, matching ``GenRMPlugin.messages_to_query``."""
    lines = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get('role')
        content = message.get('content')
        if not role or not content:
            continue
        lines.append(f'{str(role).capitalize()}: {content}')
    return '\n'.join(lines)


def _parse_judge_score(text: Optional[str]) -> Optional[float]:
    """Read a judge's numeric verdict out of its free-form output.

    Prefers the ``Reward: <score>`` convention the judge is prompted for; falls back to the first bare
    number so a judge that answers ``0.7`` without the label still scores. Returns ``None`` (which
    ``compute_rewards_per_func`` records as ``nan``) when nothing numeric is found, rather than guessing
    0 -- a silent 0 would rank an unparseable verdict as a genuinely bad candidate.
    """
    if not text:
        return None
    match = re.search(r'Reward:\s*([0-1](?:\.\d+)?)', text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r'-?\d+(?:\.\d+)?', text)
    if match:
        return float(match.group(0))
    return None


def _judge_query(template: str, prompt_messages: Optional[List[Dict[str, Any]]], completion: str) -> str:
    """Render one judge prompt: the conversation that produced a candidate, plus the candidate itself."""
    messages = list(prompt_messages or [])
    # The candidate arrives as the trailing assistant turn of ``messages``; strip it so the prompt is the
    # conversation alone and the completion is supplied verbatim through the template's ``completion=``.
    if messages and messages[-1].get('role') == 'assistant':
        messages = messages[:-1]
    return template.format(prompt=_render_conversation(messages), completion=completion)


class _GenerativeJudgeReward:
    """LLM-as-judge reward: prompt a generative model to score each candidate, parse the number out.

    A generative RM is a causal_lm and shares the sampler's role, so when its base is the sampler's own
    model it REUSES that engine -- the reward LoRA (if any) is resident in the sampler's adapter list at
    construction and selected per request via ``adapter_path``. Otherwise it owns a standalone engine
    (``owns_sampler``), released by :meth:`shutdown` (which ``_RewardChannels.shutdown`` calls).

    Scoring runs through a twinkle sampler, NOT a legacy engine: this is the dev-native counterpart of
    swift's ``GenRMPlugin``, reusing its judge-prompt and ``Reward: <score>`` parsing conventions.
    """

    def __init__(self,
                 sampler: Any,
                 *,
                 judge_template: Optional[str] = None,
                 adapter_path: Optional[str] = None,
                 params: Any = None,
                 owns_sampler: bool = False):
        self.sampler = sampler
        self.template = judge_template or _DEFAULT_JUDGE_TEMPLATE
        self.adapter_path = adapter_path
        self.params = params
        self.owns_sampler = owns_sampler

    def __call__(self, completions: List[str], messages=None, **kwargs) -> List[Optional[float]]:
        from swift.dev.builders import sampled_texts

        messages = messages or [[] for _ in completions]
        trajectories = [
            {
                'messages': [{
                    'role': 'user',
                    'content': _judge_query(self.template, prompt, completion)
                }]
            } for prompt, completion in zip(messages, completions)
        ]
        sample_kwargs = {'adapter_path': self.adapter_path} if self.adapter_path is not None else {}
        texts = sampled_texts(self.sampler.sample(trajectories, self.params, **sample_kwargs))
        return [_parse_judge_score(group[0] if group else '') for group in texts]

    def shutdown(self) -> None:
        if self.owns_sampler and self.sampler is not None:
            self.sampler.shutdown()


class _ApiJudgeReward:
    """LLM-as-judge reward served by a remote OpenAI-compatible endpoint.

    The API counterpart of :class:`_GenerativeJudgeReward` -- same judge template and score parser, but
    generation happens on a remote server through the OpenAI client (the pattern ``_ClientSampler``
    already uses, including its thread-pool concurrency). Chosen when ``orm_model`` / ``prm_model`` is
    an http(s) URL. The judge model name is taken from ``engine_args['model']``, as the 'client' sampler
    backend does; there is deliberately no separate field for it.
    """

    def __init__(self,
                 base_url: str,
                 *,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 judge_template: Optional[str] = None,
                 max_workers: int = 8,
                 timeout: float = 600.0):
        from twinkle import requires
        requires('openai')
        from openai import OpenAI

        if not model:
            raise ValueError('an API judge reward needs the judge model name; put it in engine_args '
                             "({'model': ..., 'base_url': ...}) the way the 'client' sampler backend does.")
        self.model = model
        self.client = OpenAI(
            base_url=base_url, api_key=api_key or os.environ.get('OPENAI_API_KEY'), timeout=timeout)
        self.template = judge_template or _DEFAULT_JUDGE_TEMPLATE
        self.max_workers = max_workers

    def __call__(self, completions: List[str], messages=None, **kwargs) -> List[Optional[float]]:
        from concurrent.futures import ThreadPoolExecutor

        messages = messages or [[] for _ in completions]
        queries = [_judge_query(self.template, prompt, completion)
                   for prompt, completion in zip(messages, completions)]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            texts = list(pool.map(self._one, queries))
        return [_parse_judge_score(text) for text in texts]

    def _one(self, query: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': query
                }],
                temperature=0.0,
                max_tokens=_JUDGE_MAX_NEW_TOKENS)
        except Exception as exc:  # noqa: BLE001
            # Degrade to an unparseable (nan) score rather than ending a long run: a rate-limited judge
            # on one candidate must not abort the whole synthesis.
            logger.warning(f'API judge request failed, scoring this candidate as unparseable: {exc}')
            return ''
        return completion.choices[0].message.content or ''

    def shutdown(self) -> None:
        """Nothing to release: the OpenAI client holds no GPU and closes its own connections."""


class _ModelReward:
    """Score candidates with a frozen scalar (seq_cls/reranker) reward model, dev-native.

    Built through the SAME shared builder GRPO uses (:func:`swift.dev.reward.build_frozen_reward_model`),
    so a scalar RM here and in RL are identical: its real ``task_type`` / ``num_labels`` come from the
    checkpoint's metadata rather than a hard-coded head, it is built with ``build_model`` in local mode,
    and an optional frozen LoRA is attached the same way. It scores by ``forward_only`` logits (via the
    'default' RM plugin), NOT by prompting -- that is the generative judge's job.
    """

    def __init__(self,
                 model_id: str,
                 model_config: ModelConfig,
                 template_config: TemplateConfig,
                 *,
                 adapter: Optional[str] = None,
                 model_type: Optional[str] = None,
                 revision: Optional[str] = None,
                 template_name: Optional[str] = None,
                 distributed_config: Optional[DistributedConfig] = None,
                 remote_group: Optional[str] = None):
        from swift.dev.reward import build_frozen_reward_model, build_reward_model_plugins

        model, template = build_frozen_reward_model(
            model_id,
            model_config,
            template_config,
            model_type=model_type,
            revision=revision,
            template_name=template_name,
            adapter=adapter,
            distributed_config=distributed_config,
            remote_group=remote_group)
        plugins, _ = build_reward_model_plugins([model], [template], ['default'])
        self.model = model
        self.plugin = plugins[0]

    def __call__(self, completions: List[str], messages=None, **kwargs) -> List[float]:
        messages = messages or [[] for _ in completions]
        rows = []
        for prompt, completion in zip(messages, completions):
            prompt = list(prompt or [])
            if not prompt or prompt[-1].get('role') != 'assistant' or prompt[-1].get('content') != completion:
                prompt.append({'role': 'assistant', 'content': completion})
            rows.append({'messages': prompt})
        scores = self.plugin(inputs=rows)
        return [float(value) for value in scores.tolist()]

    def shutdown(self) -> None:
        shutdown = getattr(self.model, 'shutdown', None)
        if shutdown is not None:
            shutdown()


def _normalize(scores: List[float]) -> List[float]:
    """Min-max the group into [0, 1], as legacy's ``normalize`` did.

    A degenerate group (every score equal) has no spread to stretch, so it collapses to a constant --
    ``min(1.0, value)`` for a positive score, 0.0 otherwise. Returning the raw values instead would
    make the threshold behave differently for degenerate and non-degenerate groups.
    """
    if not scores:
        return scores
    low, high = min(scores), max(scores)
    if low == high:
        return [min(1.0, low) if low > 0 else 0.0] * len(scores)
    return [(value - low) / (high - low + 1e-5) for value in scores]


class _CandidateCache:
    """Candidates from earlier runs, keyed by prompt, i.e. legacy's ``cache_files``.

    Keyed on the prompt's messages rather than on row order, because the cache files come from other
    runs whose dataset order and slicing need not match this one's. A prompt is only served from cache
    when it has at least as many candidates as this run asks for -- fewer would silently shrink the
    group that the reward ranking then works on.
    """

    def __init__(self, cache_files: List[str]):
        self.by_prompt: Dict[str, List[_Candidate]] = {}
        for path in cache_files or []:
            if not os.path.isfile(path):
                logger.warning(f'cache_files entry {path} does not exist; ignoring it.')
                continue
            self._load(path)
        if self.by_prompt:
            logger.info(f'run_sampling: cache covers {len(self.by_prompt)} prompts from {len(cache_files)} file(s)')

    def _load(self, path: str) -> None:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # A crashed producer leaves a truncated final line; skipping it beats refusing the
                    # whole cache file.
                    logger.warning(f'skipping an unparseable line in {path}')
                    continue
                messages = entry.get('messages')
                if not messages or messages[-1].get('role') != 'assistant':
                    continue
                multi_turn = bool(entry.get('multi_turn'))
                if multi_turn:
                    key = entry.get('id')
                    if not key:
                        continue
                else:
                    key = _prompt_key(messages[:-1])
                self.by_prompt.setdefault(key, []).append(
                    _Candidate(
                        messages[-1].get('content') or '',
                        messages=copy.deepcopy(messages) if multi_turn else None,
                        rollout_infos=copy.deepcopy(entry.get('rollout_infos')),
                        truncated=bool(entry.get('truncated')),
                        multi_turn=multi_turn))

    def lookup(self, trajectory: Dict[str, Any], wanted: int, *, multi_turn: bool = False) -> Optional[List[_Candidate]]:
        if not self.by_prompt:
            return None
        cached = self.by_prompt.get(_prompt_key(trajectory['messages']))
        if cached is None:
            return None
        if multi_turn:
            cached = [candidate for candidate in cached if candidate.multi_turn and candidate.messages is not None]
        else:
            cached = [_Candidate(candidate.text) for candidate in cached]
        return cached[:wanted] if len(cached) >= wanted else None


def _prompt_key(messages: List[Dict[str, Any]]) -> str:
    return hashlib.md5(
        json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8')).hexdigest()


def _is_too_easy(n_passing: int, n_candidates: int, easy_query_threshold: Optional[float]) -> bool:
    """Whether a prompt is too easy to be worth keeping.

    A prompt almost every candidate already answers correctly carries no learning signal, and keeping
    it biases the dataset toward what the model can do. None disables the filter.
    """
    if easy_query_threshold is None:
        return False
    return n_passing / n_candidates >= easy_query_threshold


def _dpo_line(row: Dict[str, Any], trajectory: Dict[str, Any], positive: _Candidate,
              negative: Optional[_Candidate], ground_truth: Optional[str], *,
              rollout_tokens: Optional[str] = None, reward_score: Optional[float] = None,
              rejected_rollout_tokens: Optional[str] = None,
              rejected_reward_score: Optional[float] = None) -> str:
    """Build one output row while preserving complete multi-turn chosen/rejected trajectories.

    The token/score fields are the save_rollout_tokens schema additions: each is written only when the
    recorder produced it (a local backend, tokens kept), so with the feature off the row is unchanged.
    """
    generated_keys = {
        'messages', 'rejected_messages', 'rejected_response', 'multi_turn', 'truncated', 'rollout_infos',
        'rejected_truncated', 'rejected_rollout_infos', 'id'
    }
    out: Dict[str, Any] = {key: value for key, value in row.items() if key not in generated_keys}
    out['messages'] = _candidate_messages(positive, trajectory)
    if negative is not None:
        if positive.multi_turn or negative.multi_turn:
            out['rejected_messages'] = _candidate_messages(negative, trajectory)
        else:
            out['rejected_response'] = negative.text
    if positive.multi_turn:
        out['multi_turn'] = True
        out['truncated'] = positive.truncated
        if positive.rollout_infos:
            out['rollout_infos'] = positive.rollout_infos
        if negative is not None:
            out['rejected_truncated'] = negative.truncated
            if negative.rollout_infos:
                out['rejected_rollout_infos'] = negative.rollout_infos
    if ground_truth is not None:
        out['ground_truth'] = ground_truth
    # Group id: every row from one prompt shares it, so positives can be traced back to their group.
    prompt_repr = json.dumps(trajectory['messages'], sort_keys=True, ensure_ascii=False, default=str)
    out['id'] = hashlib.md5(prompt_repr.encode('utf-8')).hexdigest()
    # save_rollout_tokens sidecar paths + reward scores, present only when the recorder emitted them.
    if rollout_tokens is not None:
        out['rollout_tokens'] = rollout_tokens
    if reward_score is not None:
        out['reward_score'] = reward_score
    if rejected_rollout_tokens is not None:
        out['rejected_rollout_tokens'] = rejected_rollout_tokens
    if rejected_reward_score is not None:
        out['rejected_reward_score'] = rejected_reward_score
    return json.dumps(out, ensure_ascii=False, default=str) + '\n'


def _plan_batches(n_rows: int, batch_size: int, max_batches: Optional[int]) -> List[Tuple[int, int]]:
    """Fixed ``(start, end)`` slices, computed up front so the resume index means the same thing
    across runs. A trailing partial batch is kept, unlike legacy's ``n // batch_size`` truncation."""
    if batch_size < 1:
        raise ValueError(f'SamplingConfig.batch_size must be >= 1, got {batch_size}.')
    batches = [(start, min(start + batch_size, n_rows)) for start in range(0, n_rows, batch_size)]
    return batches[:max_batches] if max_batches else batches


class _CheckpointPaths:
    """The four-file resume scheme, kept in one place so the ordering cannot be got wrong.

    ``final`` only ever appears via the closing move, which is what makes its existence mean "this run
    finished" -- the check ``run_sampling`` opens with depends on that.
    """

    def __init__(self, output_dir: str, output_file: str):
        self.final = os.path.join(output_dir, output_file)
        self.tmp = self.final + '.tmp'
        self.resume = self.final + '.resume'
        self.state = os.path.join(output_dir, 'sampling_state.json')

    def prepare(self, resume: bool) -> Tuple[int, str]:
        """Returns ``(last_finished_batch_index, open_mode)``; -1 means start from the first batch.

        Resuming copies the snapshot back over ``tmp`` first: ``tmp`` may hold a half-written batch
        from the crash, and appending to that would emit a truncated row.
        """
        if not resume:
            for path in (self.tmp, self.resume, self.state):
                if os.path.exists(path):
                    os.remove(path)
            return -1, 'w'

        if os.path.exists(self.resume):
            shutil.copyfile(self.resume, self.tmp)
        last = -1
        if os.path.exists(self.state):
            with open(self.state, 'r', encoding='utf-8') as f:
                last = json.load(f).get('batch_index', -1)
            logger.info(f'run_sampling: resuming after batch index {last}')
        return last, 'a'

    def checkpoint(self, batch_index: int) -> None:
        """Snapshot, then record. In this order: a snapshot without a state file re-does one batch,
        whereas a state file without its snapshot would skip a batch whose rows were never saved."""
        shutil.copyfile(self.tmp, self.resume)
        with open(self.state, 'w', encoding='utf-8') as f:
            json.dump({'batch_index': batch_index}, f)

    def finalize(self) -> None:
        """Publish the snapshot as the output file, keeping any previous one under a timestamp."""
        if os.path.exists(self.final):
            shutil.move(self.final, f'{self.final}.{int(time.time())}')
        source = self.resume if os.path.exists(self.resume) else self.tmp
        shutil.move(source, self.final)
        for path in (self.tmp, self.state):
            if os.path.exists(path):
                os.remove(path)
