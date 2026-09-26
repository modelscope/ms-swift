"""run_infer: offline inference over a dataset, on twinkle's Sampler.

dev counterpart of legacy ``swift infer`` (``swift/pipelines/infer/infer.py::SwiftInfer`` plus
``infer/utils.py``). It covers the same ground as legacy:

- three generation backends (vllm / sglang / transformers; lmdeploy is deliberately dropped),
- LoRA, either applied at request time or merged in first,
- the pooling task types (seq_cls / embedding / reranker), which run a forward pass rather than
  generation: on vLLM/SGLang through the sampler's ``encode``, on transformers through a HF forward,
- a generative reranker, which is a decoder-only causal LM scored off generation (the yes/no logprob
  difference of its first token) on vLLM/SGLang, and through a HF forward on transformers,
- incremental result writing with cross-process gathering, and the acc/rouge metrics. Streaming to
  the terminal is the interactive REPL's job, which now lives in :mod:`swift.dev.recipe.infer_tui`.

What is structured differently from legacy, and why:

- generation and pooling are two functions with one dispatcher, instead of ``task_type`` branches
  threaded through a single ``_batch_infer``. They share the sampler surface but not the call -- one
  decodes tokens via ``sample``, the other runs one forward via ``encode`` (or a HF forward on
  transformers) -- so keeping them apart is what stops each from carrying the other's cases.
- there is no ``__getattr__`` proxy onto the engine. Legacy's ``SwiftInfer.infer`` was actually the
  engine's method, which made the public surface depend on the backend; here the recipe owns it.
- the sampler is built once and shut down in a ``finally``, so a crash mid-run still frees the GPU.
"""
from __future__ import annotations
import copy
import hashlib
import math
import os
import re
import shutil
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import json

import numpy as np

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import (
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        InferConfig,
        ModelConfig,
        PluginConfig,
        QuantizeConfig,
        RLHFConfig,
        RolloutConfig,
        TemplateConfig,
        TunerConfig,
    )

logger = get_logger()

#: How many top logprobs to request when scoring a generative reranker off generation. The positive and
#: negative tokens must both land in this window or the score cannot be read; 20 is within the default
#: ``max_logprobs`` of both vLLM and sglang and comfortably covers the two tokens a trained reranker
#: puts its mass on.
_GENERATIVE_RERANKER_TOP_LOGPROBS = 20


def run_infer(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    infer_config: InferConfig,
    generation_config: Optional[GenerationConfig] = None,
    *,
    rlhf_config: Optional[RLHFConfig] = None,
    rollout_config: Optional[RolloutConfig] = None,
    backend: str = 'vllm',
    engine_args: Optional[Dict[str, Any]] = None,
    distributed_config: Optional[DistributedConfig] = None,
    tuner_config: Optional[TunerConfig] = None,
    adapters: Optional[List[str]] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    plugin_config: Optional[PluginConfig] = None,
    merge_lora: bool = False,
    max_rows: Optional[int] = None,
    split_dataset_ratio: float = 0.01,
    output_path: Optional[str] = None,
    _shutdown: bool = True,
) -> List[Dict[str, Any]]:
    """Infer over a dataset: one pipeline, three task-type paths, and (for generation) three knobs.

    The generative path is a single pipeline -- sample a candidate group per prompt, optionally score
    it, then shape the rows -- whose behavior is set by three orthogonal knobs on ``infer_config``:
    ``num_return_sequences`` (candidates per prompt), the reward funcs (score every candidate, for any
    n), and ``output_format`` ('all' stores the whole group, 'dpo' stores best-of-n pairs). A pooling
    ``task_type`` (seq_cls / embedding / reranker) or a ``generative_reranker`` instead runs a forward
    pass and returns one value per row.

    Args:
        model_config: model id/path, dtype, ``task_type`` (selects the path).
        template_config: chat template, and the ``system`` that overrides the dataset's.
        dataset_config: what to infer over. See ``split_dataset_ratio`` for how the split is chosen.
        infer_config: engine choice plus every generation/output/scoring knob (``infer_backend``,
            ``num_return_sequences``, ``output_format``, ``batch_size``, ``metric``, ``strict``, the
            reward channels, resume / cache / rollout-token dumping).
        generation_config: decoding knobs. ``stream`` only affects the interactive REPL
            (:mod:`swift.dev.recipe.infer_tui`); this dataset pipeline always samples in batch.
        rlhf_config: the reward-func registry surface (``reward_funcs`` / ``reward_weights``) and the
            multi-turn config (``max_turns`` / ``max_trajectory_tokens``), shared with GRPO.
        rollout_config: tool/sandbox config for a multi-turn rollout.
        backend: generation engine -- 'vllm' / 'sglang' / 'transformers' to sample locally, or
            'client' / 'no' for a remote teacher / a cache-only run.
        engine_args: forwarded verbatim to the engine (or to the 'client' teacher).
        distributed_config: DP / ray placement. The generative path gives the sampler -- and any
            GPU-resident reward model -- its own DeviceGroup; the forward paths use the 'model' group.
        tuner_config: source of ``adapters`` when not given explicitly.
        adapters: LoRA checkpoints. One adapter is selected per request (``adapter_path``); the engine
            is configured for LoRA at construction because vLLM cannot enable it later.
        merge_lora: fold the adapters into the base weights first and infer on the merged model.
        max_rows: stop after this many rows, for smoke tests.
        split_dataset_ratio: how much of ``dataset`` becomes the eval split when
            ``DatasetConfig.val_dataset`` is not set. 0 runs over the whole dataset.
        output_path: jsonl destination. None returns the rows without writing a file.
        _shutdown: leave True. False keeps the engine alive for tests that reuse it.

    Returns:
        Result rows. The generative 'all' path yields one row per prompt (``response`` / ``responses``
        / ``labels`` / ``messages``, plus ``scores`` when scored, plus every dataset column); 'dpo'
        yields chosen/rejected pairs; the forward paths yield one value per row.
    """
    from swift.dev.builders import is_pooling_task
    from swift.dev.plugin import PluginRegistry

    # A custom model / dataset / reward lives in a plugin file, so its registration must precede the
    # first name lookup. Importing the sandbox module registers the 'tool' extension point (and its
    # built-in plugin) so a user's external @register('tool', ...) finds the kind already declared; its
    # twinkle env/tool imports are lazy, so this stays cheap when tools are off.
    import swift.dev.rollout.sandbox  # noqa: F401
    PluginRegistry.load_configured(plugin_config)

    adapters = _resolve_adapters(adapters, tuner_config)
    if merge_lora and adapters:
        # run_merge_lora loads through transformers directly, so this needs no twinkle init and runs
        # before either path initializes.
        model_config, adapters = _merge_adapters(model_config, template_config, adapters)

    task_type = model_config.task_type or 'causal_lm'
    if is_pooling_task(task_type) or task_type == 'generative_reranker':
        from swift.dev.builders import load_prompt_rows
        from swift.dev.recipe.assembly import TrainAssembly

        # The forward-pass paths build a twinkle model/engine on the 'model' DeviceGroup, so they
        # initialize through the training assembly; the generative path initializes for sampling
        # (its own 'sampler' group plus any GPU-resident reward models) inside _run_generative.
        TrainAssembly.initialize_twinkle(distributed_config)
        rows = load_prompt_rows(dataset_config, max_rows, split_dataset_ratio)
        if not rows:
            raise ValueError('run_infer got an empty dataset. Set DatasetConfig.dataset or .val_dataset.')
        if is_pooling_task(task_type):
            return _run_pooling(
                model_config, template_config, distributed_config, rows, adapters, quantize_config,
                output_path, infer_config.metric, backend=backend, engine_args=engine_args,
                shutdown=_shutdown, reranker_use_activation=infer_config.reranker_use_activation)
        return _run_generative_reranker(
            model_config, template_config, generation_config, distributed_config, rows, adapters,
            quantize_config, output_path, infer_config.metric, backend=backend, engine_args=engine_args,
            shutdown=_shutdown, reranker_use_activation=infer_config.reranker_use_activation)

    return _run_generative(
        model_config, template_config, dataset_config, infer_config, generation_config,
        rlhf_config=rlhf_config, rollout_config=rollout_config, backend=backend, engine_args=engine_args,
        distributed_config=distributed_config, adapters=adapters, quantize_config=quantize_config,
        max_rows=max_rows, split_dataset_ratio=split_dataset_ratio, output_path=output_path,
        shutdown=_shutdown)


def _run_generative(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    infer_config: InferConfig,
    generation_config: Optional[GenerationConfig],
    *,
    rlhf_config: Optional[RLHFConfig],
    rollout_config: Optional[RolloutConfig],
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    distributed_config: Optional[DistributedConfig],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
    max_rows: Optional[int],
    split_dataset_ratio: float,
    output_path: Optional[str],
    shutdown: bool,
) -> List[Dict[str, Any]]:
    """The causal-LM path: sample a candidate group per prompt, optionally score it, then emit rows.

    Generation goes through one ``RolloutEngine`` (single- and multi-turn alike, a local sampler or a
    remote 'client' teacher), scoring through ``_RewardChannels`` (rule-based ORM/PRM funcs and/or model
    rewards), and emission through the ``output_format`` strategy -- so a new storage shape is a new
    emit function, not a new recipe.
    """
    from swift.dev.builders import (build_device_mesh_if_dp, build_sampler, build_template,
                                    load_model_processor, load_prompt_rows, to_sampling_params)

    num_return = infer_config.num_return_sequences
    output_format = infer_config.output_format
    if output_format == 'dpo':
        if num_return < 2:
            raise ValueError(f"output_format='dpo' needs num_return_sequences >= 2 to have both a chosen "
                             f'and a rejected candidate, got {num_return}.')
        if infer_config.n_best_to_keep >= num_return:
            raise ValueError(f'n_best_to_keep={infer_config.n_best_to_keep} must be < '
                             f'num_return_sequences={num_return}: the lowest-scoring candidate becomes the '
                             'rejected response, so it cannot also be a positive.')
    multi_turn_enabled = bool(rlhf_config and rlhf_config.max_turns is not None)
    if infer_config.score_ground_truth and multi_turn_enabled:
        # The reference answer is a single assistant turn with no tool trajectory, while a multi-turn
        # candidate's reward reads its full messages / rollout_infos. Scoring both on one axis compares
        # different shapes, so this is rejected (STAGE2_GOALS: score_ground_truth is mutually exclusive
        # with a multi-turn / gym rollout, whose reference has no rollout trajectory).
        raise ValueError(
            'score_ground_truth cannot combine with a multi-turn rollout: the reference answer has no '
            "tool trajectory to score against the candidates'. Turn off score_ground_truth, or run "
            'single-turn.')

    # Resolve the model-reward channels BEFORE twinkle init and the sampler build: the specs decide
    # whether a reward model needs its own GPU DeviceGroup (declared at init), and a generative judge
    # reusing the sampler needs its reward LoRA resident in the engine's adapter list at construction
    # (vLLM/SGLang size their adapter slots up front). The channels themselves are built after the
    # sampler, since a judge scores through it.
    sampler_model = model_config.model
    orm_spec = _resolve_reward_model(infer_config.orm_model, infer_config.orm_adapter, sampler_model, backend)
    prm_spec = _resolve_reward_model(infer_config.prm_model, infer_config.prm_adapter, sampler_model, backend)
    reuse_adapters = [spec.adapter for spec in (orm_spec, prm_spec)
                      if spec is not None and spec.kind == 'generative_reuse' and spec.adapter]
    reward_groups = _exclusive_reward_groups(orm_spec, prm_spec)
    if reward_groups and (distributed_config is None or distributed_config.mode != 'ray'):
        raise ValueError(
            f'reward channel(s) {reward_groups} keep their own model resident on GPU and need a dedicated '
            "DeviceGroup, which only DistributedConfig.mode='ray' can place. Run with mode='ray' (and "
            'nproc_per_node), or use a sampler-reusing generative judge / an http(s) endpoint, which need '
            'no extra device.')
    if distributed_config is not None:
        _initialize_for_sampling(distributed_config, reward_groups)

    rows = load_prompt_rows(dataset_config, max_rows, split_dataset_ratio)
    if not rows:
        raise ValueError('run_infer got an empty dataset. Set DatasetConfig.dataset or .val_dataset.')
    logger.info(f'run_infer: {len(rows)} prompts, backend={backend}, num_return_sequences={num_return}, '
                f'output_format={output_format}')

    # A 'dpo' or resumed run uses the checkpointed writer (tmp live-write, per-batch resume snapshot,
    # atomic finalize); a plain 'all' run uses the incremental writer, which also gathers across DP
    # ranks. An existing complete checkpoint output short-circuits the run unless override_exist_file,
    # so re-running a finished job is a no-op.
    use_checkpoint = bool(output_path) and (output_format == 'dpo' or infer_config.resume)
    if use_checkpoint:
        probe = _CheckpointPaths(os.path.dirname(output_path) or 'output', os.path.basename(output_path))
        if os.path.exists(probe.final) and not infer_config.override_exist_file:
            logger.info(f'run_infer: {probe.final} exists and override_exist_file is False; nothing to do.')
            return []

    cache = _CandidateCache(infer_config.cache_files)
    quant_method = getattr(quantize_config, 'quant_method', None)
    if backend in {'client', 'no'} and quant_method is not None:
        raise ValueError(
            f'quant_method={quant_method!r} cannot affect infer_backend={backend!r}, which loads no local '
            'model. Configure quantization on the remote server or omit --quant_method.')

    template = None
    adapter_path = adapters[0] if adapters else None
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
            model_config, backend=backend, engine_args=engine_args,
            device_mesh=build_device_mesh_if_dp(distributed_config), template=template,
            adapters=sampler_adapters or None,
            remote_group=_SAMPLER_GROUP if distributed_config is not None
            and distributed_config.mode == 'ray' else None,
            quantize_config=quantize_config)

    channels = _build_channels(
        infer_config, rlhf_config, orm_spec=orm_spec, prm_spec=prm_spec, sampler=sampler,
        model_config=model_config, template_config=template_config, backend=backend,
        engine_args=engine_args, distributed_config=distributed_config, quantize_config=quantize_config)

    rollout = None
    if sampler is not None:
        # One generation entry for single- and multi-turn alike: a RolloutEngine wrapping the already-built
        # sampler (injected, so this recipe keeps owning its lifecycle and calls close(), not shutdown()).
        # With backend='client' the teacher owns chat formatting and exposes no local template, so
        # template is None and the path runs message-only: no token IDs, just the teacher's text per turn.
        from swift.dev.rollout import RolloutEngine
        rollout = RolloutEngine(sampler=sampler, template=template)
        if multi_turn_enabled:
            # Tools are opt-in: with RolloutConfig.tools set, a sandbox env pool is built and the tool
            # plugins resolved so each episode leases its own env; otherwise the rollout has no tools.
            from swift.dev.rollout.sandbox import build_tool_sandbox
            env_pool, tool_plugins = build_tool_sandbox(rollout_config)
            rollout.configure_multi_turn(
                max_turns=rlhf_config.max_turns, max_trajectory_tokens=rlhf_config.max_trajectory_tokens,
                env_pool=env_pool, tool_plugins=tool_plugins)

    batches = _plan_batches(len(rows), infer_config.batch_size, infer_config.max_batches)
    params = to_sampling_params(generation_config, num_samples=num_return)
    writer = (_CheckpointWriter(output_path, infer_config.resume) if use_checkpoint else
              _IncrementalWriter(output_path, 1 if output_path else None))
    # Rollout-token sidecar: only built when save_rollout_tokens is on, and only a local backend can feed
    # it (a message-only client has no tokens; validate rejects that combination). The NPZ dir sits beside
    # the jsonl and each row embeds a path relative to output_dir, so the pair stays portable.
    recorder = None
    if infer_config.save_rollout_tokens:
        if not output_path:
            raise ValueError('save_rollout_tokens needs an output_path to place the NPZ sidecar beside.')
        recorder = _RolloutRecorder(
            output_path + '.rollout_tokens', base_dir=os.path.dirname(output_path) or 'output', enabled=True)

    results: List[Dict[str, Any]] = []
    # Throughput/token summary, the dev counterpart of legacy's InferStats: prompts and generated tokens
    # are counted as batches complete, then reported once at the end. Token counts come only from a local
    # backend (a message-only 'client'/'no' run has none), so tokens/s is logged only when tokens were seen.
    num_prompts = 0
    num_generated_tokens = 0
    start_time = time.perf_counter()
    try:
        for index, (start, end) in enumerate(batches):
            if writer.skip(index):
                continue
            batch = rows[start:end]
            logger.info(f'run_infer: batch {index + 1}/{len(batches)} ({len(batch)} prompts)')
            trajectories, ground_truths, groups = _sample_candidates(
                sampler, rollout, batch, params, template_config, infer_config, cache, backend,
                adapter_path, multi_turn_enabled)
            emit = _emit_dpo if output_format == 'dpo' else _emit_all
            batch_rows: List[Dict[str, Any]] = []
            for row, trajectory, ground_truth, group in zip(batch, trajectories, ground_truths, groups):
                group = [candidate for candidate in group if candidate.text]
                if not group:
                    continue
                num_prompts += 1
                num_generated_tokens += sum(len(candidate.response_token_ids) for candidate in group)
                scores = _score_group(group, row, trajectory, ground_truth, channels, infer_config)
                batch_rows.extend(emit(row, trajectory, ground_truth, group, scores, infer_config, recorder))
            results.extend(batch_rows)
            writer.write(batch_rows)
            writer.checkpoint(index)
    finally:
        channels.shutdown()
        if rollout is not None:
            rollout.close()
        if shutdown and sampler is not None:
            sampler.shutdown()

    writer.finish(results)
    runtime = time.perf_counter() - start_time
    stats: Dict[str, Any] = {
        'num_prompts': num_prompts,
        'runtime': round(runtime, 3),
        'prompts/s': round(num_prompts / runtime, 3) if runtime > 0 else 0.0,
    }
    if num_generated_tokens:
        stats['num_generated_tokens'] = num_generated_tokens
        stats['tokens/s'] = round(num_generated_tokens / runtime, 1) if runtime > 0 else 0.0
    logger.info(f'run_infer stats: {stats}')
    if infer_config.metric:
        if output_format == 'dpo':
            logger.warning("metric=%r does not apply to output_format='dpo' rows: they carry chosen/rejected "
                           "pairs, not a single response/labels to score. Use output_format='all' for a "
                           'metric.', infer_config.metric)
        else:
            logger.info(f'run_infer metric: {compute_metric(results, infer_config.metric)}')
    return results


def _sample_candidates(
    sampler: Any,
    rollout: Any,
    batch: List[Dict[str, Any]],
    params: Any,
    template_config: TemplateConfig,
    infer_config: InferConfig,
    cache: '_CandidateCache',
    backend: str,
    adapter_path: Optional[str],
    multi_turn_enabled: bool,
) -> Tuple[List[Dict[str, Any]], List[Optional[str]], List[List['_Candidate']]]:
    """One batch of prompts -> ``(trajectories, ground_truths, candidate groups)``, sampling only.

    Scoring and row shaping happen in the caller so both output formats share one sampling call.
    Prompts already covered by ``cache_files`` skip generation entirely -- they are the expensive part,
    and their candidates are reused verbatim.
    """
    from swift.dev.builders import split_prompt_and_reference

    trajectories, ground_truths = split_prompt_and_reference(batch, template_config)
    wanted = infer_config.num_return_sequences
    cached = [cache.lookup(trajectory, wanted, multi_turn=multi_turn_enabled) for trajectory in trajectories]
    # to_sample: the prompts the cache did NOT cover -- only these pay for generation, the rest reuse
    # their cached candidates verbatim. Fresh candidates are written back to their original index below.
    to_sample = [index for index, hit in enumerate(cached) if hit is None]
    candidates: List[List[_Candidate]] = [hit or [] for hit in cached]
    if to_sample:
        if sampler is None:
            raise ValueError("infer_backend='no' requires cache_files to cover every input prompt.")
        if rollout is None:
            raise RuntimeError('sampling was requested but no rollout engine was built.')
        selected_trajectories = [trajectories[i] for i in to_sample]
        prompt_extras = []
        # Per-prompt side inputs for the rollout: the dataset row's non-message columns merged with the
        # trajectory's, so a reward func / tool sees the same extras whether or not the prompt was cached.
        for index, trajectory in zip(to_sample, selected_trajectories):
            extra = {key: copy.deepcopy(value) for key, value in batch[index].items() if key != 'messages'}
            extra.update({key: copy.deepcopy(value) for key, value in trajectory.items() if key != 'messages'})
            prompt_extras.append(extra)
        # One generation call for single- and multi-turn alike: RolloutEngine.generate dispatches on whether
        # a multi-turn engine was configured, and both paths yield the same RolloutSample. logprobs are
        # forced only when the recorder keeps tokens (a plain run skips the extra compute); ``strict`` is a
        # transformers single-turn kwarg. ``adapter_path`` selects the LoRA the engine reserved slots for
        # and is honoured by both paths -- the multi-turn engine threads it into every per-turn sample.
        sample_kwargs: Dict[str, Any] = {}
        if backend == 'transformers':
            sample_kwargs['strict'] = infer_config.strict
        if adapter_path is not None:
            sample_kwargs['adapter_path'] = adapter_path
        samples = rollout.generate(
            [trajectory['messages'] for trajectory in selected_trajectories],
            num_samples=wanted,
            sampling_params=asdict(params),
            prompt_extras=prompt_extras,
            force_logprobs=infer_config.save_rollout_tokens,
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
        logger.info(f'run_infer: reused cached candidates for {len(batch) - len(to_sample)}/{len(batch)} prompts')
    return trajectories, ground_truths, candidates


def _score_group(candidates: List['_Candidate'], row: Dict[str, Any], trajectory: Dict[str, Any],
                 ground_truth: Optional[str], channels: '_RewardChannels',
                 infer_config: InferConfig) -> Optional[List[float]]:
    """Score one prompt's candidates -> one float each, or None when no reward func is configured.

    Shared by both output formats so scoring happens once per group and is orthogonal to storage: a
    single-completion run is scored too. The ground truth, when scored, is appended so per-group
    normalisation sees it as the anchor, then trimmed back off -- it is never emitted as a candidate
    because the model did not produce it.
    """
    if channels.empty:
        return None
    scored = list(candidates)
    if infer_config.score_ground_truth and ground_truth:
        # trajectory['messages'] is the prompt with the reference turn already stripped (see
        # split_prompt_and_reference), so rebuilding prompt + ground_truth-as-assistant is the correct
        # single-turn conversation. _run_generative rejects this in a multi-turn rollout, where the
        # reference would have no tool trajectory to compare against the candidates.
        scored.append(_Candidate(ground_truth, messages=_candidate_messages(_Candidate(ground_truth), trajectory)))
    return channels.score(scored, row, trajectory, infer_config)[:len(candidates)]


def _emit_all(row: Dict[str, Any], trajectory: Dict[str, Any], ground_truth: Optional[str],
              candidates: List['_Candidate'], scores: Optional[List[float]], infer_config: InferConfig,
              recorder: Any = None) -> List[Dict[str, Any]]:
    """'all' format: one row per prompt carrying every candidate (the plain-inference / eval shape).

    ``response`` is the first candidate and ``responses`` all of them, the shape the acc/rouge metric and
    downstream eval read. When a reward func scored the group, ``scores`` rides along per candidate; with
    save_rollout_tokens, ``rollout_tokens`` is the per-candidate NPZ path list.
    """
    prompt_id = _prompt_key(trajectory['messages'])
    out: Dict[str, Any] = {key: value for key, value in row.items() if key != 'messages'}
    out['response'] = candidates[0].text
    out['responses'] = [candidate.text for candidate in candidates]
    out['messages'] = _candidate_messages(candidates[0], trajectory)
    if candidates[0].multi_turn:
        # Each multi-turn candidate is a distinct trajectory (its own tool calls); ``messages`` keeps only
        # the first, so store every candidate's full trajectory rather than losing all but one.
        out['all_messages'] = [_candidate_messages(candidate, trajectory) for candidate in candidates]
    if candidates[0].rollout_logprobs:
        # Populated only when logprobs were requested (generation_config.logprobs / save_rollout_tokens);
        # mirrors legacy's per-row ``logprobs`` for the response this row keeps.
        out['logprobs'] = list(candidates[0].rollout_logprobs)
    if ground_truth is not None:
        out['labels'] = ground_truth
    if scores is not None:
        # A judge that returned nothing numeric scores nan; null it so the row stays json-serialisable.
        out['scores'] = [None if score != score else score for score in scores]
    if recorder is not None:
        out['rollout_tokens'] = [
            recorder.record(prompt_id, index, candidate) for index, candidate in enumerate(candidates)
        ]
    return [out]


def _emit_dpo(row: Dict[str, Any], trajectory: Dict[str, Any], ground_truth: Optional[str],
              candidates: List['_Candidate'], scores: Optional[List[float]], infer_config: InferConfig,
              recorder: Any = None) -> List[Dict[str, Any]]:
    """'dpo' format: best-of-n chosen/rejected pairs for DPO training.

    Without any reward func (``scores is None``) every candidate is emitted as a positive with no rejected
    turn -- there is no ranking, so naming one candidate worse would be a fabrication. With scores,
    candidates at or below ``reward_threshold`` are dropped, an easy prompt (see ``easy_query_threshold``)
    is skipped whole, then the top ``n_best_to_keep`` become positives paired against the lowest scorer.
    """
    prompt_id = _prompt_key(trajectory['messages'])
    keep_tokens = recorder is not None
    record = recorder.record if keep_tokens else None
    if scores is None:
        rows = []
        for index, positive in enumerate(candidates):
            token_path = record(prompt_id, index, positive) if keep_tokens else None
            rows.append(_dpo_row(row, trajectory, positive, None, ground_truth, rollout_tokens=token_path))
        return rows

    threshold = infer_config.reward_threshold
    # An unscored candidate (nan, from a judge/API that returned nothing numeric) is dropped from the
    # ranking whole: it can be neither a positive nor the rejected response, and writing nan into a
    # reward_score would emit invalid JSON. ``scorable`` is the nan-free pool the ranking works on.
    scorable = [i for i in range(len(candidates)) if scores[i] == scores[i]]
    if not scorable:
        return []
    keep = [i for i in scorable if threshold is None or scores[i] > threshold]
    if not keep:
        return []
    if _is_too_easy(len(keep), len(candidates), infer_config.easy_query_threshold):
        return []
    ranked = sorted(scorable, key=lambda i: scores[i], reverse=True)
    negative_index = ranked[-1]
    positives = [i for i in ranked[:infer_config.n_best_to_keep] if i in keep and i != negative_index]
    if not positives:
        return []
    negative = candidates[negative_index]
    logger.debug(f'scores={[round(s, 4) for s in scores]} positives={positives} negative={negative_index}')
    rejected_tokens = record(prompt_id, negative_index, negative) if keep_tokens else None
    rows = []
    for i in positives:
        rows.append(
            _dpo_row(
                row,
                trajectory,
                candidates[i],
                negative,
                ground_truth,
                rollout_tokens=record(prompt_id, i, candidates[i]) if keep_tokens else None,
                reward_score=scores[i] if keep_tokens else None,
                rejected_rollout_tokens=rejected_tokens,
                rejected_reward_score=scores[negative_index] if keep_tokens else None))
    return rows


def _dpo_row(row: Dict[str, Any], trajectory: Dict[str, Any], positive: '_Candidate',
             negative: Optional['_Candidate'], ground_truth: Optional[str], *,
             rollout_tokens: Optional[str] = None, reward_score: Optional[float] = None,
             rejected_rollout_tokens: Optional[str] = None,
             rejected_reward_score: Optional[float] = None) -> Dict[str, Any]:
    """Build one DPO output row while preserving complete multi-turn chosen/rejected trajectories.

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
    if rollout_tokens is not None:
        out['rollout_tokens'] = rollout_tokens
    if reward_score is not None:
        out['reward_score'] = reward_score
    if rejected_rollout_tokens is not None:
        out['rejected_rollout_tokens'] = rejected_rollout_tokens
    if rejected_reward_score is not None:
        out['rejected_reward_score'] = rejected_reward_score
    return out


def _run_pooling(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    distributed_config: Optional[DistributedConfig],
    rows: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
    output_path: Optional[str],
    metric: Optional[str],
    *,
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    shutdown: bool,
    reranker_use_activation: bool = True,
) -> List[Dict[str, Any]]:
    """The forward-pass path for seq_cls / embedding / reranker.

    Two ways to run the same pooling forward, chosen by ``backend``:

    - vLLM / SGLang have a pooling head, so the model is built as a pooling engine and served through
      ``sampler.encode`` -- the same sampler surface the generative path uses, which is what lets one
      backend serve both. This is the throughput path.
    - transformers has no pooling head, so it falls back to ``build_model`` + ``forward_only(task=..)``,
      a plain HF forward. This is the reach path.

    Either way each row yields one plain-Python value (a vector, class logits, or a score), so the
    result assembly (:func:`_finalize_forward_results`) is shared with the generative-reranker path. The
    label comes from the row's own ``label`` column rather than from a trailing assistant turn, because
    there is no completion to strip.
    """
    from swift.dev.builders import to_trajectory

    task_type = model_config.task_type
    trajectories = [to_trajectory(row, list(row['messages']), template_config) for row in rows]

    if backend in ('vllm', 'sglang'):
        per_row = _pooling_via_sampler(
            model_config, template_config, distributed_config, trajectories, adapters, quantize_config, backend,
            engine_args, shutdown, reranker_use_activation)
    else:
        per_row = _forward_via_transformers(
            model_config, template_config, distributed_config, rows, trajectories, adapters, quantize_config,
            max_batch_size=(engine_args or {}).get('max_batch_size'),
            reranker_use_activation=reranker_use_activation)

    return _finalize_forward_results(rows, per_row, task_type, backend, output_path, metric)


def _finalize_forward_results(
    rows: List[Dict[str, Any]],
    per_row: List[Any],
    task_type: Optional[str],
    backend: str,
    output_path: Optional[str],
    metric: Optional[str],
) -> List[Dict[str, Any]]:
    """Assemble, write and score the one-value-per-row output of a forward-pass path.

    Shared by :func:`_run_pooling` and :func:`_run_generative_reranker`: both produce exactly one plain
    value per row (a vector, class logits, or a relevance score), so the row shape is the same. The label
    comes from the row's own ``label`` column rather than a trailing assistant turn, because there is no
    completion to strip.
    """
    logger.info(f'run_infer: {len(rows)} rows, task_type={task_type}, backend={backend}')
    results = []
    for row, output in zip(rows, per_row):
        passthrough = {key: value for key, value in row.items() if key != 'messages'}
        results.append({
            'response': output,
            'responses': [output],
            'labels': row.get('label'),
            'messages': list(row['messages']),
            **passthrough
        })
    if output_path:
        _write_jsonl(output_path, results)
        logger.info(f'run_infer: wrote {len(results)} rows to {output_path}')
    if metric:
        logger.info(f'run_infer metric: {compute_metric(results, metric)}')
    return results


def _sigmoid(score: float) -> float:
    """Numerically stable sigmoid, mapping a reranker's raw score into a [0, 1] relevance."""
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    exp = math.exp(score)
    return exp / (1.0 + exp)


@contextmanager
def _forward_sampler(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    distributed_config: Optional[DistributedConfig],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    shutdown: bool,
):
    """Build a vLLM/SGLang sampler for a forward-pass path, and shut it down afterwards.

    Shared by the pooling (``encode``) and generative-reranker (``sample``) sampler paths: they differ
    only in the call they make, while the model/template/mesh build, the single-adapter selection and the
    teardown are identical. Yields ``(sampler, processor, adapter_path)``.
    """
    from swift.dev.builders import build_device_mesh_if_dp, build_sampler, build_template, load_model_processor

    _, processor = load_model_processor(model_config)
    template = build_template(template_config, processor)
    sampler = build_sampler(
        model_config,
        backend=backend,
        engine_args=engine_args,
        device_mesh=build_device_mesh_if_dp(distributed_config),
        template=template,
        adapters=adapters,
        quantize_config=quantize_config)
    try:
        yield sampler, processor, (adapters[0] if adapters else None)
    finally:
        if shutdown:
            sampler.shutdown()


def _pooling_via_sampler(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    distributed_config: Optional[DistributedConfig],
    trajectories: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    shutdown: bool,
    reranker_use_activation: bool = True,
) -> List[Any]:
    """Run the pooling forward on a vLLM/SGLang sampler's ``encode``, one plain value per row."""
    from swift.dev.builders import pooled_data, to_pooling_params

    with _forward_sampler(model_config, template_config, distributed_config, adapters, quantize_config,
                          backend, engine_args, shutdown) as (sampler, _processor, adapter_path):
        # A cross-encoder reranker's raw score is a logit; ``use_activation`` has twinkle sigmoid it into a
        # [0, 1] relevance comparable across queries (legacy reranker_use_activation). It rides in as a
        # PoolingParams override, the channel to_pooling_params documents for these post-processing knobs.
        overrides = {'use_activation': reranker_use_activation} if model_config.task_type == 'reranker' else {}
        pooling_params = to_pooling_params(model_config.task_type, **overrides)
        kwargs: Dict[str, Any] = {}
        if adapter_path is not None:
            kwargs['adapter_path'] = adapter_path
        return pooled_data(sampler.encode(trajectories, pooling_params, **kwargs))


def _forward_via_transformers(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    distributed_config: Optional[DistributedConfig],
    rows: List[Dict[str, Any]],
    trajectories: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
    max_batch_size: Optional[int] = None,
    reranker_use_activation: bool = True,
) -> List[Any]:
    """Run one HF ``forward_only`` pass (transformers backend), one plain value per row.

    Serves both the pooling tasks (seq_cls / embedding / reranker) and the transformers
    ``generative_reranker`` path, which reads its yes/no logit difference out of the raw logits
    returned here. Requests are chunked by ``max_batch_size`` so a large dataset does not materialise
    every encoded input and every output tensor at once.
    """
    from swift.dev.builders import build_model, build_template, load_model_processor
    from swift.dev.config import DistributedConfig

    task_type = model_config.task_type
    _, processor = load_model_processor(model_config)
    template = build_template(template_config, processor)
    model = build_model(
        model_config, distributed_config or DistributedConfig(), quantize_config=quantize_config)
    if adapters:
        # add_adapter_to_model takes a checkpoint directory as well as a PeftConfig, so a trained
        # adapter is loaded here rather than built -- apply_tuner would create a fresh, untrained one.
        for index, adapter in enumerate(adapters):
            model.add_adapter_to_model(f'adapter_{index}' if index else 'default', adapter)

    # forward_only resolves a generative_reranker's yes/no token ids off the model's template, so it must
    # be attached even though the inputs are already encoded here (the pooling tasks ignore it). Without
    # this the transformers generative-reranker path trips twinkle's "call set_template before forwarding".
    model.set_template(template)

    batch_size = max_batch_size or len(rows) or 1
    per_row: List[Any] = []
    for start in range(0, len(rows), batch_size):
        chunk = trajectories[start:start + batch_size]
        encoded = [template.encode(trajectory) for trajectory in chunk]
        outputs = model.forward_only(inputs=encoded, task=task_type, return_logits=True)
        per_row.extend(_per_row_outputs(outputs, len(chunk)))
    if reranker_use_activation and task_type in ('reranker', 'generative_reranker'):
        # forward_only returns the raw relevance logit (a cross-encoder score, or a generative reranker's
        # yes/no logit difference); sigmoid maps it to [0, 1], matching legacy's activation and the
        # vLLM/SGLang paths. seq_cls / embedding are left untouched.
        per_row = [
            [_sigmoid(value) for value in row] if isinstance(row, (list, tuple)) else _sigmoid(row)
            for row in per_row
        ]
    return per_row


def _per_row_outputs(outputs: Any, num_rows: int) -> List[Any]:
    """Unpack ``forward_only``'s output into one plain-Python value per row.

    The shape depends on the head: seq_cls gives logits per row, embedding gives a vector. Both are
    returned as lists so the result is jsonl-serialisable, which is the whole point of this path.
    """
    tensor = outputs
    if isinstance(outputs, dict):
        for key in ('logits', 'embedding', 'last_hidden_state'):
            if key in outputs:
                tensor = outputs[key]
                break
    if hasattr(tensor, 'tolist'):
        listed = tensor.tolist()
        if isinstance(listed, list) and len(listed) == num_rows:
            return listed
        return [listed]
    return list(tensor) if isinstance(tensor, (list, tuple)) else [tensor] * num_rows


def _run_generative_reranker(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    generation_config: Optional[GenerationConfig],
    distributed_config: Optional[DistributedConfig],
    rows: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
    output_path: Optional[str],
    metric: Optional[str],
    *,
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    shutdown: bool,
    reranker_use_activation: bool = True,
) -> List[Dict[str, Any]]:
    """Score a decoder-only (generative) reranker, one relevance score per row.

    A generative reranker is a causal LM, not a pooling model: its score is the difference between the
    logprob of a positive token (``yes``) and a negative one (``no``) at the first generated position.
    That is exactly what ``swift.utils.torch_utils.get_generative_reranker_logits`` computes from the
    lm_head for the HF path, and what the generation path reproduces here from logprobs -- the two agree
    because the log-softmax normaliser cancels in the difference.

    So the backend split differs from pooling: vLLM/SGLang run it through *generation* (``sample``),
    since neither serves a decoder-only reranker through ``encode`` (sglang's docs are explicit that it
    must not be launched with ``--is-embedding``); transformers runs the same HF forward the pooling
    path uses, with ``task='generative_reranker'``. Either way each row yields one score, so the result
    assembly is shared.
    """
    from swift.dev.builders import to_trajectory

    task_type = model_config.task_type
    trajectories = [to_trajectory(row, list(row['messages']), template_config) for row in rows]

    if backend in ('vllm', 'sglang'):
        per_row = _generative_reranker_via_sampler(
            model_config, template_config, generation_config, distributed_config, trajectories, adapters,
            quantize_config, backend, engine_args, shutdown, reranker_use_activation)
    else:
        per_row = _forward_via_transformers(
            model_config, template_config, distributed_config, rows, trajectories, adapters, quantize_config,
            max_batch_size=(engine_args or {}).get('max_batch_size'),
            reranker_use_activation=reranker_use_activation)

    return _finalize_forward_results(rows, per_row, task_type, backend, output_path, metric)


def _generative_reranker_via_sampler(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    generation_config: Optional[GenerationConfig],
    distributed_config: Optional[DistributedConfig],
    trajectories: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    shutdown: bool,
    reranker_use_activation: bool = True,
) -> List[float]:
    """Score a generative reranker on a vLLM/SGLang generation engine, one score per row."""
    from swift.dev.builders import to_sampling_params

    # Built as a generation engine, not a pooling runner: build_sampler treats ``generative_reranker`` as
    # a generative task_type, so no ``runner='pooling'``/``is_embedding`` is injected.
    with _forward_sampler(model_config, template_config, distributed_config, adapters, quantize_config,
                          backend, engine_args, shutdown) as (sampler, processor, adapter_path):
        tokenizer = _tokenizer_of(processor)
        positive_id, negative_id = _generative_reranker_token_ids(tokenizer)
        # temperature is forced to 1.0 so the reported logprobs are the log-softmax of the raw logits:
        # only then does logprob(yes) - logprob(no) equal the logit difference the HF path returns (any
        # other temperature scales the logits before the softmax and changes the gap). max_tokens=1 --
        # not 0 -- because twinkle's max_tokens==0 logprobs-only path drops logprobs, and we need the
        # first generated token's.
        params = to_sampling_params(
            generation_config, max_tokens=1, temperature=1.0, logprobs=_GENERATIVE_RERANKER_TOP_LOGPROBS)
        kwargs: Dict[str, Any] = {}
        if adapter_path is not None:
            kwargs['adapter_path'] = adapter_path
        responses = sampler.sample(trajectories, params, **kwargs)
        return [
            _score_from_logprobs(response, positive_id, negative_id, reranker_use_activation)
            for response in responses
        ]


def _score_from_logprobs(response: Any, positive_id: int, negative_id: int,
                         use_activation: bool = True) -> float:
    """``logprob(yes) - logprob(no)`` at the first generated position of one SampleResponse.

    With ``use_activation`` the log-ratio is squashed by a sigmoid into ``P(yes) / (P(yes) + P(no))``, a
    [0, 1] relevance comparable across queries -- the same activation legacy applies to a generative
    reranker, and the counterpart of the pooling path's ``PoolingParams.use_activation``.
    """
    sequence = response.sequences[0]
    if not sequence.logprobs:
        raise RuntimeError('generative reranker scoring requested logprobs but the sampler returned none; '
                           'the backend may not honour logprobs on this path.')
    topk = dict(sequence.logprobs[0])
    if positive_id not in topk or negative_id not in topk:
        raise RuntimeError(
            f'generative reranker: the positive/negative tokens ({positive_id}/{negative_id}) are not both '
            f'in the top-{len(topk)} logprobs of the first generated token. Raise '
            '_GENERATIVE_RERANKER_TOP_LOGPROBS (or check GENERATIVE_RERANKER_POSITIVE_TOKEN/'
            'GENERATIVE_RERANKER_NEGATIVE_TOKEN) so both are returned.')
    score = topk[positive_id] - topk[negative_id]
    return _sigmoid(score) if use_activation else score


def _generative_reranker_token_ids(tokenizer: Any) -> Tuple[int, int]:
    """The (positive, negative) token ids, from the same env vars the HF scoring path reads."""
    positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
    negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')
    return tokenizer.convert_tokens_to_ids(positive_token), tokenizer.convert_tokens_to_ids(negative_token)


def _tokenizer_of(processor: Any) -> Any:
    """A tokenizer from whatever ``load_model_processor`` returned (a tokenizer, or a processor wrapping one)."""
    return getattr(processor, 'tokenizer', processor)


class _IncrementalWriter:
    """Append result rows to jsonl as batches complete, gathering across DP ranks first.

    Why gather: with DP > 1 each rank holds only its own slice, so a naive per-rank write either
    interleaves partial files or has every rank overwrite the same path. Legacy solved this with
    ``JsonlWriter(gather_obj=True)``; twinkle's ``gather_object`` is the same idea.

    This is the plain-'all' writer. When ``batch_size`` is falsy it does nothing until :meth:`finish`,
    which writes everything at once. :meth:`skip` / :meth:`checkpoint` are no-ops, so the generative
    loop drives this and :class:`_CheckpointWriter` through one interface.
    """

    def __init__(self, output_path: Optional[str], batch_size: Optional[int]):
        self.output_path = output_path
        self.incremental = bool(output_path and batch_size)
        self._started = bool(output_path and os.path.exists(output_path))

    def skip(self, index: int) -> bool:
        return False

    def write(self, batch: List[Dict[str, Any]]) -> None:
        if not self.incremental:
            return
        rows = _gather_rows(batch)
        if rows is None:
            return  # not the writing rank
        _write_jsonl(self.output_path, rows, append=self._started)
        self._started = True
        logger.info(f'run_infer: flushed {len(rows)} rows to {self.output_path}')

    def checkpoint(self, index: int) -> None:
        pass

    def finish(self, results: List[Dict[str, Any]]) -> None:
        if self.incremental or not self.output_path:
            return
        rows = _gather_rows(results)
        if rows is None:
            return
        _write_jsonl(self.output_path, rows, append=os.path.exists(self.output_path))
        logger.info(f'run_infer: wrote {len(rows)} rows to {self.output_path}')


class _CheckpointWriter:
    """The checkpointed jsonl writer for a 'dpo' or resumed run, over :class:`_CheckpointPaths`.

    ``tmp`` is the live write, ``resume`` the snapshot taken after each completed batch, ``state`` the
    last finished batch index, and the closing move to ``final`` is what marks a run complete -- so a
    multi-hour run that crashes resumes instead of restarting. Rows arrive as dicts and are serialized
    here, so it and :class:`_IncrementalWriter` share one emit surface. Under DP this assumes a single
    driving process (the ray sampler-group layout); the incremental writer is the one that gathers
    across torchrun DP ranks.
    """

    def __init__(self, output_path: str, resume: bool):
        output_dir = os.path.dirname(output_path) or 'output'
        os.makedirs(output_dir, exist_ok=True)
        self.paths = _CheckpointPaths(output_dir, os.path.basename(output_path))
        self._resume_from, write_mode = self.paths.prepare(resume)
        self._f = open(self.paths.tmp, write_mode, encoding='utf-8')

    def skip(self, index: int) -> bool:
        return index <= self._resume_from

    def write(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            self._f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        self._f.flush()

    def checkpoint(self, index: int) -> None:
        self.paths.checkpoint(index)

    def finish(self, results: List[Dict[str, Any]]) -> None:
        self._f.close()
        self.paths.finalize()
        logger.info(f'run_infer: wrote {self.paths.final}')


def _gather_rows(rows: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """All ranks' rows on the writing rank, None elsewhere. A no-op without torch.distributed."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return rows

    from twinkle.utils import framework_util, is_master

    gathered = framework_util.gather_object(rows, device_mesh=None)
    return gathered if is_master() else None


def _resolve_adapters(adapters: Optional[List[str]], tuner_config: Optional[TunerConfig]) -> Optional[List[str]]:
    if adapters:
        return list(adapters)
    if tuner_config is not None and getattr(tuner_config, 'adapters', None):
        return list(tuner_config.adapters)
    return None


def _merge_adapters(model_config: ModelConfig, template_config: TemplateConfig,
                    adapters: List[str]) -> Tuple[ModelConfig, None]:
    """Merge the adapters into the base weights and return a ModelConfig pointing at the result.

    The returned config is a copy: mutating the caller's would make a second run with the same object
    silently infer on the merged directory. Adapters come back as None because they are now part of
    the weights -- passing them again would apply the same delta twice.
    """
    import dataclasses

    from swift.dev.config import TunerConfig
    from swift.dev.recipe.merge_lora import run_merge_lora

    merged = run_merge_lora(
        model_config, TunerConfig(adapters=list(adapters)), template_config=template_config, device_map='cpu')
    logger.info(f'run_infer: merged {len(adapters)} adapter(s) into {merged}')
    return dataclasses.replace(model_config, model=merged), None


def _write_jsonl(path: str, rows: List[Dict[str, Any]], append: bool = False) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'a' if append else 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def compute_metric(results: List[Dict[str, Any]], metric: Literal['acc', 'rouge']) -> Dict[str, float]:
    """Score the first completion of each row against its reference answer.

    ``acc`` is exact string equality, which is what legacy's ``--metric acc`` measured -- NOT the
    token-level ``twinkle.metric.Accuracy``, which scores logits against label ids and would report a
    different (and much higher) number for the same run.

    Only the first completion is scored: with ``num_samples > 1`` a best-of-n score is a different
    measurement (pass@n) and reporting it as accuracy would overstate the model. Rows without a
    reference are skipped rather than counted as wrong.
    """
    from twinkle.metric import ExactMatch, RougeBleu

    pairs = [(r['response'], r['labels']) for r in results if r.get('labels') is not None and r.get('response')]
    if not pairs:
        logger.warning('metric requested but no row has both a response and a reference answer.')
        return {}
    predictions, references = zip(*pairs)
    scorer = ExactMatch() if metric == 'acc' else RougeBleu()
    scorer.accumulate(predictions=list(predictions), references=list(references))
    return scorer.calculate()


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


def plan_sampling_device_groups(ranks_per_group: Optional[int],
                                reward_group_names: List[str]) -> Tuple[List[Tuple[str, List[int]]], int]:
    """Plan the twinkle DeviceGroups for a sampling run. Pure function (no twinkle import).

    ``ranks_per_group`` is the GPU width of ONE role -- the sampler and every GPU-resident reward
    channel each get their own disjoint block of that size. It is ``DistributedConfig.nproc_per_node``,
    NOT twinkle's ``nproc_per_node`` (which is the total rank count, returned as ``total_ranks``). So a
    run with R reward groups needs ``ranks_per_group * (1 + R)`` GPUs. Mirrors ``plan_rl_device_groups``
    in run_grpo: the reward group is sized off the same world size the sampler uses rather than a
    separate GPU-count knob.

    Returns ``(groups, total_ranks)`` where ``groups`` is a list of ``(name, ranks)`` to hand
    twinkle.initialize and ``total_ranks`` is the summed GPU count.
    """
    if ranks_per_group is None or ranks_per_group < 1:
        raise ValueError(f'ranks_per_group must be >= 1 (the sampler GPU count), got {ranks_per_group!r}.')
    groups: List[Tuple[str, List[int]]] = [(_SAMPLER_GROUP, list(range(ranks_per_group)))]
    start = ranks_per_group
    for name in reward_group_names:
        groups.append((name, list(range(start, start + ranks_per_group))))
        start += ranks_per_group
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
        # Local mode reaches here only with no reward groups: _run_generative already rejects a
        # GPU-resident reward channel unless mode='ray' (there is no disjoint DeviceGroup to place it
        # in), so there is nothing left to guard here.
        twinkle.initialize(mode='local')
        return
    # DistributedConfig.nproc_per_node is the SAMPLER's GPU width (one role); twinkle's nproc_per_node
    # below is the TOTAL rank count across every group. Same spelling, two meanings -- the planner maps
    # the former to the latter.
    ranks_per_group = distributed_config.nproc_per_node
    if ranks_per_group is None:
        raise ValueError("DistributedConfig.nproc_per_node is required in mode='ray' (it sizes the sampler's "
                         'Ray DeviceGroup and each reward group). Pass it explicitly -- there is no default.')
    planned, total_ranks = plan_sampling_device_groups(ranks_per_group, reward_groups or [])
    twinkle.initialize(
        mode='ray',
        nproc_per_node=total_ranks,
        groups=[
            DeviceGroup(name=name, ranks=ranks, device_type='GPU', gpus_per_worker=1) for name, ranks in planned
        ])


class _RolloutRecorder:
    """Writes one NPZ per kept candidate under ``token_dir``; a no-op when ``enabled`` is False.

    ``base_dir`` is the directory the jsonl lives in, so :meth:`record` returns a path relative to it --
    that relative path is what gets embedded in the row, keeping the jsonl portable alongside its sidecar.
    NPZ filenames are positionally deterministic (``{prompt_id}_c{candidate_index}.npz``), so replaying a
    batch on resume overwrites the same files instead of appending duplicates.
    """

    def __init__(self, token_dir: str, base_dir: str = '', enabled: bool = False):
        self.token_dir = token_dir
        self.base_dir = base_dir
        self.enabled = enabled
        if enabled:
            os.makedirs(token_dir, exist_ok=True)

    def record(self, prompt_id: str, candidate_index: int, candidate: Any) -> Optional[str]:
        """Persist one candidate's rollout tokens; return the NPZ path relative to ``base_dir``.

        Returns None when disabled, or when the candidate carries no token feature (a message-only
        backend), so a path is embedded only for candidates that actually have tokens. ``candidate`` is
        read by attribute (``encoded`` / ``response_token_ids`` / ``rollout_logprobs`` /
        ``response_loss_mask``), so both a ``_Candidate`` and a ``RolloutSample`` work.
        """
        if not self.enabled:
            return None
        encoded = getattr(candidate, 'encoded', None) or {}
        input_ids = encoded.get('input_ids')
        labels = encoded.get('labels')
        if not input_ids or not labels:
            return None
        arrays = {
            'input_ids': np.asarray(input_ids, dtype=np.int64),
            'labels': np.asarray(labels, dtype=np.int64),
            'response_token_ids': np.asarray(getattr(candidate, 'response_token_ids', None) or [], dtype=np.int64),
            'rollout_logprobs': np.asarray(getattr(candidate, 'rollout_logprobs', None) or [], dtype=np.float32),
            'response_loss_mask': np.asarray(getattr(candidate, 'response_loss_mask', None) or [], dtype=np.int8),
        }
        completion_mask = encoded.get('completion_mask')
        if completion_mask is not None:
            arrays['completion_mask'] = np.asarray(completion_mask, dtype=np.int8)
        name = f'{prompt_id}_c{candidate_index}.npz'
        abs_path = os.path.join(self.token_dir, name)
        np.savez_compressed(abs_path, **arrays)
        return os.path.relpath(abs_path, self.base_dir) if self.base_dir else name


def _build_channels(infer_config: InferConfig,
                    rlhf_config: Optional[RLHFConfig] = None,
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

    # reward_funcs / reward_weights live on RLHFConfig (the shared registry surface GRPO also uses); the
    # rest of the synthesis knobs are on InferConfig. RLHFConfig doubles as the ORM hyperparameter carrier.
    reward_funcs = rlhf_config.reward_funcs if rlhf_config is not None else []
    orm_weights = rlhf_config.reward_weights if rlhf_config is not None else None
    orm_funcs, orm_names = get_reward_funcs(reward_funcs, rlhf_config)
    prm_funcs, prm_names = get_reward_funcs(infer_config.prm_funcs, rlhf_config)
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
                infer_config,
                sampler=sampler,
                model_config=model_config,
                template_config=template_config,
                backend=backend,
                engine_args=engine_args,
                distributed_config=distributed_config,
                quantize_config=quantize_config,
                remote_group=remote_group))
        names.append(spec.display_name)
    channels = _RewardChannels(orm_funcs, orm_names, prm_funcs, prm_names, orm_weights=orm_weights)
    if channels.empty:
        logger.info('run_infer: no reward funcs -- every candidate is emitted as a positive.')
    else:
        logger.info(f'run_infer: orm={orm_names} (x{infer_config.orm_channel_weight}) prm={prm_names}, '
                    f'normalize={infer_config.normalize_rewards}')
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


def _candidate_messages(candidate: _Candidate, trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    if candidate.messages is not None:
        return copy.deepcopy(candidate.messages)
    return copy.deepcopy(trajectory['messages']) + [{'role': 'assistant', 'content': candidate.text}]


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
                 prm_names: List[str],
                 orm_weights: Optional[List[float]] = None):
        self.orm_funcs = orm_funcs
        self.orm_names = orm_names
        self.prm_funcs = prm_funcs
        self.prm_names = prm_names
        # ORM channel weights come from RLHFConfig.reward_weights (captured in _build_channels); keeping
        # them here avoids threading rlhf_config through every score call.
        self.orm_weights = orm_weights

    @property
    def empty(self) -> bool:
        return not self.orm_funcs and not self.prm_funcs

    def score(self, candidates: List[_Candidate], row: Dict[str, Any], trajectory: Dict[str, Any],
              infer_config: InferConfig) -> List[float]:
        """Score one prompt's candidates -> one float each.

        Dataset columns are broadcast across the candidates. ``messages`` is different: each reward
        receives the complete trajectory that produced that candidate, including intermediate turns.
        """
        total = [0.0] * len(candidates)
        if self.orm_funcs:
            orm = self._channel(
                candidates, row, trajectory, self.orm_funcs, self.orm_weights, infer_config)
            total = [t + infer_config.orm_channel_weight * value for t, value in zip(total, orm)]
        if self.prm_funcs:
            prm = self._channel(
                candidates, row, trajectory, self.prm_funcs, infer_config.prm_weights, infer_config)
            total = [t + value for t, value in zip(total, prm)]
        return total

    @staticmethod
    def _channel(candidates: List[_Candidate], row: Dict[str, Any], trajectory: Dict[str, Any], funcs: List[Any],
                 weights: Optional[List[float]], infer_config: InferConfig) -> List[float]:
        from swift.dev.reward import compute_rewards_per_func, weight_rewards

        columns = {key: [value] * len(candidates) for key, value in row.items() if key != 'messages'}
        columns['messages'] = [_candidate_messages(candidate, trajectory) for candidate in candidates]
        columns['rollout_infos'] = [copy.deepcopy(candidate.rollout_infos or {}) for candidate in candidates]
        columns['truncated'] = [candidate.truncated for candidate in candidates]
        rewards_per_func = compute_rewards_per_func([candidate.text for candidate in candidates], funcs, columns)
        scores = weight_rewards(rewards_per_func, weights).tolist()
        # Normalise per channel, before the channels are added: doing it after would let the channel
        # with the larger raw range decide the ranking regardless of the weights.
        return _normalize(scores) if infer_config.normalize_rewards else scores

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

#: Built-in LLM-as-judge prompt, overridable via ``InferConfig.judge_template``. A ``str.format``
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
                        infer_config: InferConfig,
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
            spec.model_id, model=(engine_args or {}).get('model'), judge_template=infer_config.judge_template)
    # Generative judge: greedy decoding so the verdict is deterministic across a run.
    judge_params = to_sampling_params(None, temperature=0.0, max_tokens=_JUDGE_MAX_NEW_TOKENS)
    if spec.kind == 'generative_reuse':
        return _GenerativeJudgeReward(
            sampler, judge_template=infer_config.judge_template, adapter_path=spec.adapter, params=judge_params)
    judge_sampler = _build_judge_sampler(
        spec, model_config, template_config, backend, engine_args, distributed_config, quantize_config,
        remote_group)
    return _GenerativeJudgeReward(
        judge_sampler,
        judge_template=infer_config.judge_template,
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
    # A judge/API that returned nothing numeric scores nan. It must not poison the group's min/max -- a
    # single nan would make both nan and flatten every real score to nan too. Normalise over the real
    # scores and pass nan through untouched, so downstream still sees it as "unscored" (and _emit_dpo /
    # _emit_all can drop or null it).
    valid = [value for value in scores if value == value]
    if not valid:
        return scores
    low, high = min(valid), max(valid)
    if low == high:
        constant = min(1.0, low) if low > 0 else 0.0
        return [value if value != value else constant for value in scores]
    return [value if value != value else (value - low) / (high - low + 1e-5) for value in scores]


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
            logger.info(f'run_infer: cache covers {len(self.by_prompt)} prompts from {len(cache_files)} file(s)')

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


def _plan_batches(n_rows: int, batch_size: int, max_batches: Optional[int]) -> List[Tuple[int, int]]:
    """Fixed ``(start, end)`` slices, computed up front so the resume index means the same thing
    across runs. A trailing partial batch is kept, unlike legacy's ``n // batch_size`` truncation."""
    if batch_size < 1:
        raise ValueError(f'InferConfig.batch_size must be >= 1, got {batch_size}.')
    batches = [(start, min(start + batch_size, n_rows)) for start in range(0, n_rows, batch_size)]
    return batches[:max_batches] if max_batches else batches


class _CheckpointPaths:
    """The four-file resume scheme, kept in one place so the ordering cannot be got wrong.

    ``final`` only ever appears via the closing move, which is what makes its existence mean "this run
    finished" -- the early-return check ``_run_generative`` opens with depends on that.
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
            logger.info(f'run_infer: resuming after batch index {last}')
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
