# Copyright (c) ModelScope Contributors. All rights reserved.
# GRPO's device and cluster assembly: the init halves and the StreamingDriver seams (per-recipe).
from __future__ import annotations

import math
import os
from copy import copy
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from swift.dev.utils.logger import get_logger

from swift.dev.recipe.utils import (
    REF_GROUP, TEACHER_GROUP, auxiliary_mesh, build_rl_sampler, resolve_reward_scorers)

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig, DatasetConfig, DistributedConfig, LoggingConfig, MegatronConfig, ModelConfig,
        MoEConfig, QuantizeConfig, RLConfig, RewardConfig, RolloutConfig, TemplateConfig, TrainConfig,
        TunerConfig)

logger = get_logger()


def greedy_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """argmax decoding for a remax baseline: temperature 0 and no nucleus/top-k."""
    greedy = dict(params)
    greedy['temperature'] = 0.0
    greedy.pop('top_p', None)
    greedy.pop('top_k', None)
    return greedy


def weighted_reward(values: List[Any], weights: Any) -> float:
    """One row's weighted reward sum (nan-aware: an unparseable judge verdict never poisons it)."""
    import torch

    row = torch.tensor([float('nan') if value is None or value != value else float(value) for value in values],
                       dtype=torch.float32)
    if weights is not None and row.numel() == len(weights):
        row = row * weights
    return float(row.nansum())


def group_variance(rewards_block: List[List[Any]], weights: Any) -> float:
    """The weighted-reward std of one group (dynamic_sample's zero-variance test)."""
    import torch

    if len(rewards_block) <= 1:
        return 0.0
    matrix = torch.tensor(
        [[0.0 if value is None or value != value else float(value) for value in row] or [0.0]
         for row in rewards_block], dtype=torch.float32)
    if weights is not None and matrix.shape[1] == len(weights):
        matrix = matrix * weights.unsqueeze(0)
    weighted = matrix.nansum(dim=1)
    return float(torch.std(weighted, unbiased=False)) if weighted.numel() > 1 else 0.0


def grpo_group_requests(rl_config: 'RLConfig', reward_config: 'RewardConfig',
                        distributed_config: 'DistributedConfig',
                        tuner_config: Optional['TunerConfig']) -> List[Tuple[str, Optional[str]]]:
    """The disjoint DeviceGroups GRPO's full-parameter frozen models need -> ``[(name, spec)]``.

    Mirrors the builds in :func:`initialize_cluster` exactly: the KL reference (active iff beta > 0
    and calculate_KL is not False; under sync_ref_model it inherits the policy's sharding for the
    paired EMA), the one RLSD/SDAR teacher, each MODEL id in the orm channel, and the prm channel's
    model. A disable_adapter auxiliary reuses the policy actor and plans NO group.
    """
    from swift.dev.recipe.utils import single_teacher_id
    from swift.dev.rewards.scorer import split_channel

    requests: List[Tuple[str, Optional[str]]] = []
    if rl_config.beta and rl_config.calculate_KL is not False:
        disable = tuner_config is not None and not rl_config.ref_adapters and not rl_config.sync_ref_model
        if rl_config.ref_model and not disable:
            spec = ((rl_config.reference_parallel_spec or distributed_config.parallel_spec)
                    if rl_config.sync_ref_model else rl_config.ref_parallel_spec)
            requests.append((REF_GROUP, spec))
    teacher_id = single_teacher_id(rl_config.teacher_model, algo='RLSD/SDAR')
    if teacher_id and not rl_config._teacher_use_disable_adapter:
        requests.append((TEACHER_GROUP, rl_config.teacher_parallel_spec))
    for index in range(len(split_channel(list(reward_config.orm))[1])):
        requests.append((f'reward_{index}', reward_config.orm_parallel_spec))
    if split_channel(list(reward_config.prm))[1]:
        requests.append(('prm_0', reward_config.prm_parallel_spec))
    return requests


class Cluster(NamedTuple):
    """Every actor and handle GRPO's visible loop drives (a plain handle bundle, not a layer)."""

    model: Any
    sampler: Any
    rollout: Any
    engine: Any
    adapter: Any
    reference: Any
    teacher: Any
    worker: Any
    prm_scorer: Any
    prm_orm_mix: float
    sampling_params: Dict[str, Any]
    colocate: bool
    orm_scorers: List[Any]


def initialize_cluster(*, model_config: 'ModelConfig', template_config: 'TemplateConfig', template: Any,
                       train_config: 'TrainConfig', distributed_config: 'DistributedConfig',
                       checkpoint_config: Optional['CheckpointConfig'],
                       rollout_config: 'RolloutConfig', rl_config: 'RLConfig',
                       reward_config: 'RewardConfig', tuner_config: Optional['TunerConfig'],
                       max_steps: int, quantize_config: Optional['QuantizeConfig'] = None,
                       megatron_config: Optional['MegatronConfig'] = None,
                       moe_config: Optional['MoEConfig'] = None,
                       generation_config: Optional[Any] = None,
                       resource_plan: Any = None, auxiliary_meshes: Optional[Dict[str, Any]] = None,
                       output_dir: str = 'output') -> Cluster:
    """Build every actor GRPO trains with, explicitly: policy, sampler, reference, teacher, rewards.

    Nothing is hidden behind a shared assembly class; PPO / the distillation recipes write their
    own. ``max_steps`` is the resolved step budget (the LR-scheduler horizon), derived by the caller
    from the prompt set BEFORE this stage so the optimizer is fully configured here.
    """
    from twinkle.checkpoint_engine import CheckpointEngineManager

    from swift.dev.builders import (
        apply_full_param_freeze, build_dp_mesh, build_model, build_ref_model)
    from swift.dev.builders.adapter import apply_tuner
    from swift.dev.builders.sandbox import tools_factory
    from swift.dev.builders.optimizer import configure_optimizer
    from swift.dev.loss import configure_rl_loss, configure_rl_metrics
    from swift.dev.processor import InputProcessor
    from twinkle_agentic.rollout import MultiTurnRollout
    from swift.dev.recipe.utils import single_teacher_id
    from swift.dev.rewards.worker import RewardWorker
    from twinkle_agentic.async_rl import REF_ONLY_REMOTE_GROUP
    from twinkle.data_format import SamplingParams

    sp = template_config.sequence_parallel_size
    sp_mesh = build_dp_mesh(distributed_config, sp) if sp > 1 else None
    # Refuse the engine/training combinations a quantized refit cannot serve
    # (DeepSeek-V4 MXFP4 experts): cheaper now than after the engine is built.
    from swift.dev.weight_sync import check_quantized_weight_sync
    model_info = getattr(template, 'model_info', None)
    check_quantized_weight_sync(
        getattr(model_info, 'config', None), model_config.model_type, tuner_config, rollout_config)
    resume_dir = checkpoint_config.resume_from_checkpoint if checkpoint_config is not None else None
    policy_config = model_config
    if resume_dir and tuner_config is None:
        # A full-parameter resume loads weights from the ckpt dir instead of the original model id;
        # a tuned policy keeps the original base and reloads its adapter via resume (run_grpo).
        policy_config = copy(model_config)
        policy_config.model = resume_dir
    # create the actor (the trainable policy) here
    model = build_model(
        policy_config, distributed_config, train_config, tuner_config, device_mesh=sp_mesh,
        quantize_config=quantize_config, megatron_config=megatron_config, moe_config=moe_config,
        enable_router_replay=rl_config.router_replay_mode != 'disabled', torch_dtype='float32')
    if tuner_config is not None:
        apply_tuner(model, tuner_config, gradient_accumulation_steps=train_config.gradient_accumulation_steps,
                    task_type=model_config.task_type)
    else:
        apply_full_param_freeze(model, train_config)
    model.set_processor(
        InputProcessor, padding_free=template_config.padding_free, padding_side=template_config.padding_side,
        cp_partition_mode=distributed_config.cp_partition_mode, task_type=model_config.task_type)
    model.set_template(template)
    configure_rl_loss(model, rl_config)
    # The optimizer reads the tuner's param groups and sizes its LR scheduler from the step budget;
    # the policy metric appends onto the optimizer's metric list (twinkle add_metric contract).
    configure_optimizer(
        model, train_config, num_training_steps=max_steps, distributed_config=distributed_config,
        tuner_config=tuner_config, plugin_config=None)
    configure_rl_metrics(model, rl_config)

    colocate = rollout_config.sampler_mode == 'colocate'
    sampler = build_rl_sampler(
        model_config=model_config, rollout_config=rollout_config, template=template,
        tuner_config=tuner_config, resource_plan=resource_plan, colocate=colocate,
                   enable_sampling_replay=rl_config.enable_sampling_replay)
    # The adaptive runtime consumes the sampler AND the policy, so it is wired here next to the
    # elastic-handle registration below.
    adapter = None
    if resource_plan is not None and rollout_config.resource_mode == 'adaptive':
        from twinkle_agentic.async_rl import ClusterAdapter

        adapter = ClusterAdapter.build(
            model=model, sampler=sampler, plan=resource_plan, rollout_config=rollout_config)
    params: Dict[str, Any] = {'max_tokens': rl_config.max_completion_length}
    if generation_config is not None:
        for key in ('temperature', 'top_p', 'top_k'):
            value = getattr(generation_config, key, None)
            if value is not None:
                params[key] = value
    # One engine for every generation entry (single-turn is the one-turn degenerate): disaggregated
    # places it on the CPU payload group; colocate keeps it local for the visible loop's generate.
    rollout = MultiTurnRollout(
        sampler, template, sampling_params=SamplingParams(**params),
        max_turns=(rollout_config.max_turns or 1),
        max_trajectory_tokens=rollout_config.max_trajectory_tokens,
        tools_factory=tools_factory, tools_config=rollout_config,
        remote_group=(None if colocate else REF_ONLY_REMOTE_GROUP))
    engine = adapter.dedicated_manager if adapter is not None else CheckpointEngineManager(
        model=model, sampler=sampler, platform='GPU', mode=('colocate' if colocate else 'standalone'))

    # create the reference here (a KL anchor): a bare frozen model on its own group, or the policy
    # itself with its adapter off -- the pipeline derives disable_lora from that identity.
    reference = None
    if rl_config.beta and rl_config.calculate_KL is not False:
        ref_spec = ((rl_config.reference_parallel_spec or distributed_config.parallel_spec)
                    if rl_config.sync_ref_model else rl_config.ref_parallel_spec)
        reference = build_ref_model(
            name=REF_GROUP, model_id=rl_config.ref_model, model_config=model_config, template=template,
            distributed_config=distributed_config,
            device_mesh=auxiliary_mesh(REF_GROUP, ref_spec, auxiliary_meshes), parallel_spec=ref_spec,
            adapters=tuple(rl_config.ref_adapters or ()), policy_model=model,
            disable_adapter=(tuner_config is not None and not rl_config.ref_adapters
                             and not rl_config.sync_ref_model))
    # create the teacher here (the RLSD/SDAR reweighting channel), same bare-model shape
    teacher = build_ref_model(
        name=TEACHER_GROUP, model_id=single_teacher_id(rl_config.teacher_model, algo='RLSD/SDAR'),
        model_config=model_config, template=template, distributed_config=distributed_config,
        device_mesh=auxiliary_mesh(TEACHER_GROUP, rl_config.teacher_parallel_spec, auxiliary_meshes),
        parallel_spec=rl_config.teacher_parallel_spec,
        adapters=tuple(rl_config.teacher_adapters or ()), policy_model=model,
        disable_adapter=rl_config._teacher_use_disable_adapter)

    # create prm & orm here -- the shared channel resolver: rules local, each MODEL on its group
    backend = rollout_config.rollout_sampler
    scorer_kwargs = dict(reward_config=reward_config, model_config=model_config,
                         template_config=template_config, distributed_config=distributed_config,
                         rollout_config=rollout_config, backend=backend, sampler=sampler,
                         quantize_config=quantize_config)
    orm_scorers = resolve_reward_scorers('orm', **scorer_kwargs)
    prm_scorers = resolve_reward_scorers('prm', **scorer_kwargs)
    prm_scorer = None
    if prm_scorers:
        from swift.dev.utils.plugin import PluginRegistry

        from swift.dev.rewards import PRM_STEP_SCORER

        plugin_name = reward_config.prm_scorer or (
            'turn' if rollout_config.max_turns is not None else 'delimiter')
        prm_scorer = PluginRegistry.resolve(PRM_STEP_SCORER, plugin_name, config=reward_config)
    worker = RewardWorker(orm_scorers=orm_scorers, prm_scorers=prm_scorers,
                          prm_step_scorer=prm_scorer,
                          prm_weights=reward_config.prm_weights, template=template,
                          remote_group=(None if colocate else REF_ONLY_REMOTE_GROUP))
    # Register the GPU-resident frozen RM handles for elastic scaling; a role without a handle
    # (rules-only channel, judge, local mode) stays static and the adapter can never address it.
    elastic_models = {}
    if adapter is not None:
        for scorer in (*orm_scorers, *prm_scorers):
            if getattr(scorer, 'placement_key', None) is not None and \
                    getattr(scorer, '_worker_factory', None) is not None:
                elastic_models[scorer.placement_key] = scorer
        adapter.elastic_models = elastic_models

    return Cluster(
        model=model, sampler=sampler, rollout=rollout,
        engine=engine, adapter=adapter, reference=reference, teacher=teacher, worker=worker,
        prm_scorer=prm_scorer, prm_orm_mix=reward_config.prm_orm_mix, sampling_params=params,
        colocate=colocate, orm_scorers=orm_scorers)


def build_driver(*, cluster: Cluster, rl_config: 'RLConfig', rollout_config: 'RolloutConfig',
                 reward_config: 'RewardConfig', tuner_config: Optional['TunerConfig'],
                 train_batch_size: int, gradient_accumulation_steps: int, prompt_dataloader: Any,
                 num_train_epochs: float, max_steps: int, output_dir: str,
                 run_id: str = 'run_grpo') -> Any:
    """Assemble twinkle's StreamingDriver over the cluster (algorithm seams as closures, no wrapper class).

    Every defaulted knob (engine seams, pull rule, snapshot lifecycle, step budget, the version
    ledger, the fixed sample-list output, the colocate hand-over) lives in the kernel; what stays
    here is GRPO's own algorithm face -- the two group-completion verdicts.
    """
    from twinkle_agentic.async_rl import StreamingDriver

    from swift.dev.recipe.utils import check_streaming_config
    from swift.dev.rewards.advantage import build_reward_weights

    rollout = cluster.rollout
    worker = cluster.worker
    strategy = rollout_config.weight_sync_strategy
    if rollout_config.async_mode == 'none':
        max_staleness = 0
    elif rollout_config.async_mode == 'one_step_off':
        max_staleness = 1
    else:
        max_staleness = rollout_config.max_staleness
    adapter_name = ('default' if tuner_config is not None else None)
    check_streaming_config(
        adapter_name=adapter_name, max_staleness=max_staleness, weight_sync_strategy=strategy,
        allow_partial_rollout=rollout_config.allow_partial_rollout,
        parameter_sync_step=rollout_config.parameter_sync_step, serialize_generation=False)
    num_generations = rl_config.num_generations
    need_rows = train_batch_size * gradient_accumulation_steps

    # The weighted-orm variance test's weight vector (the only group-completion state left: the
    # ORM scoring itself runs in the kernel via score_worker and lands on the samples).
    orm_weights = (build_reward_weights(reward_config.orm_weights, len(cluster.orm_scorers))
                   if cluster.orm_scorers else None)

    # The two algorithm verdicts stay closures (dynamic_sample, the greedy remax generation, the
    # weight vectors are GRPO's own); PRM scoring is the score_worker collaborator's channel now.
    regenerate = None
    if rl_config.dynamic_sample:
        def regenerate(group_key: str, samples: List[Any]) -> bool:
            return group_variance([s.orm_reward for s in samples], orm_weights) <= 0.0

    baseline_fn = None
    if rl_config.advantage_estimator == 'remax':
        baseline_sampling_params = greedy_params(cluster.sampling_params)
        has_orm = bool(cluster.orm_scorers)
        def baseline_fn(group_key: str, samples: List[Any]) -> float:
            """The greedy completion's weighted reward, generated on the PUBLISHED weights (remax)."""
            greedy = rollout.generate(
                [list(samples[0].prompt or [])], num_samples=1,
                sampling_params=baseline_sampling_params,
                prompt_extras=[dict(samples[0].extra)] if samples[0].extra else None)
            if callable(getattr(greedy, 'result', None)):
                greedy = greedy.result()
            if len(greedy) != 1:
                raise RuntimeError(f'remax greedy baseline returned {len(greedy)} samples, expected 1.')
            greedy[0].prompt = list(samples[0].prompt or [])
            rewards = worker.score_group([greedy[0]]) if has_orm else [[]]
            if callable(getattr(rewards, 'result', None)):
                rewards = rewards.result()
            return weighted_reward(rewards[0], orm_weights)

    # Stream geometry: a publish cycle trains ``parameter_sync_step`` groups, and the buffer must
    # hold one full pull quantum -- lcm(need_rows, num_generations) rows -- or assembly deadlocks.
    groups_per_partition = max(1, rollout_config.parameter_sync_step)
    pull_rows = need_rows * num_generations // math.gcd(need_rows, num_generations)
    buffer_depth = max((max_staleness + 1) * groups_per_partition * num_generations, pull_rows)
    driver = StreamingDriver(
        run_id=run_id,
        prompt_stream=prompt_dataloader,
        num_train_epochs=num_train_epochs,
        max_staleness=max_staleness,
        num_generations=num_generations,
        groups_per_partition=groups_per_partition,
        buffer_depth=buffer_depth,
        rollout=rollout,
        pull_rule='group',
        need_rows=need_rows,
        snapshot_model=(cluster.model if strategy == 'adapter_snapshot' else None),
        snapshot_dir=os.path.join(output_dir, 'async_adapters'),
        adapter_name=adapter_name,
        max_steps=max_steps,
        consume=None,
        score=None,
        score_worker=worker,
        resource_adapter=cluster.adapter,
        allow_partial_rollout=rollout_config.allow_partial_rollout,
        regenerate=regenerate,
        baseline_fn=baseline_fn,
        max_group_regenerations=(rl_config.max_resample_times if rl_config.dynamic_sample else 0))
    return driver
