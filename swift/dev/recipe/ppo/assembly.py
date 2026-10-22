# Copyright (c) ModelScope Contributors. All rights reserved.
# PPO's assembly: the value-critic cluster (policy + critic + reference + reward channel).
from __future__ import annotations

import copy
import math
import os
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple

from swift.dev.utils.logger import get_logger

from swift.dev.recipe.utils import (
    REF_GROUP, auxiliary_mesh, build_rl_sampler, resolve_reward_scorers)

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig, DatasetConfig, DistributedConfig, MegatronConfig, ModelConfig, MoEConfig,
        QuantizeConfig, RLConfig, RewardConfig, RolloutConfig, TemplateConfig, TrainConfig,
        TunerConfig)

logger = get_logger()


def ppo_group_requests(rl_config: 'RLConfig', reward_config: 'RewardConfig',
                       distributed_config: 'DistributedConfig',
                       tuner_config: Optional['TunerConfig']) -> List[Tuple[str, Optional[str]]]:
    """The disjoint DeviceGroups PPO's full-parameter frozen models need -> ``[(name, spec)]``.

    Mirrors the builds in :func:`initialize_ppo_cluster`: PPO ALWAYS anchors the KL -- a
    full-parameter run keeps a frozen ref copy (under sync_ref_model it inherits the policy's
    sharding); only a LoRA run reuses the policy actor and plans NO group. Each orm MODEL gets
    its ``reward_<i>`` group; the critic shares the policy's 'model' group, so it plans none.
    """
    from swift.dev.rewards.scorer import split_channel

    requests: List[Tuple[str, Optional[str]]] = []
    if not (tuner_config is not None and not rl_config.ref_adapters and not rl_config.sync_ref_model):
        spec = ((rl_config.reference_parallel_spec or distributed_config.parallel_spec)
                if rl_config.sync_ref_model else rl_config.ref_parallel_spec)
        requests.append((REF_GROUP, spec))
    for index in range(len(split_channel(list(reward_config.orm))[1])):
        requests.append((f'reward_{index}', reward_config.orm_parallel_spec))
    return requests


class Cluster(NamedTuple):
    """Every actor PPO's visible loop drives (a plain handle bundle, not a layer)."""

    model: Any
    value_model: Any
    sampler: Any
    rollout: Any
    engine: Any
    adapter: Any
    reference: Any
    worker: Any
    sampling_params: Dict[str, Any]
    colocate: bool


def initialize_ppo_cluster(*, model_config: 'ModelConfig', template_config: 'TemplateConfig',
                           template: Any, train_config: 'TrainConfig',
                           distributed_config: 'DistributedConfig',
                           checkpoint_config: Optional['CheckpointConfig'],
                           rollout_config: 'RolloutConfig', rl_config: 'RLConfig',
                           reward_config: 'RewardConfig', tuner_config: Optional['TunerConfig'],
                           max_steps: int, quantize_config: Optional['QuantizeConfig'] = None,
                           megatron_config: Optional['MegatronConfig'] = None,
                           moe_config: Optional['MoEConfig'] = None,
                           generation_config: Optional[Any] = None,
                           resource_plan: Any = None,
                           auxiliary_meshes: Optional[Dict[str, Any]] = None,
                           output_dir: str = 'output') -> Cluster:
    """Build PPO's actors explicitly: the policy, the value critic, the frozen reference, the rewards.

    The critic is a trainable seq_cls ``num_labels=1`` model on the policy's OWN 'model' group (a
    distinct ``instance_id``) -- policy and critic MUST slice_dp at the same data_world_size,
    which is also why PPO plans no SP mesh. ``max_steps`` is the resolved step budget, so both
    optimizers' LR schedulers are fully configured here.
    """
    from twinkle.checkpoint_engine import CheckpointEngineManager

    from swift.dev.builders import (
        apply_full_param_freeze, build_dp_mesh, build_model, build_ref_model)
    from swift.dev.builders.adapter import apply_tuner
    from swift.dev.builders.optimizer import configure_optimizer
    from swift.dev.loss import (
        configure_ppo_value_loss, configure_ppo_value_metric, configure_rl_loss, configure_rl_metrics)
    from swift.dev.processor import InputProcessor
    from swift.dev.rewards.worker import RewardWorker
    from twinkle_agentic.rollout import MultiTurnRollout
    from twinkle_agentic.async_rl import REF_ONLY_REMOTE_GROUP
    from twinkle.data_format import SamplingParams

    sp = template_config.sequence_parallel_size
    sp_mesh = build_dp_mesh(distributed_config, sp) if sp > 1 else None
    resume_dir = checkpoint_config.resume_from_checkpoint if checkpoint_config is not None else None
    policy_config = model_config
    if resume_dir and tuner_config is None:
        # A full-parameter resume loads weights from the ckpt dir instead of the original model id;
        # a tuned policy keeps the original base and reloads its adapter via resume (run_ppo).
        policy_config = copy(model_config)
        policy_config.model = resume_dir
    model = build_model(
        policy_config, distributed_config, train_config, tuner_config, device_mesh=sp_mesh,
        quantize_config=quantize_config, megatron_config=megatron_config, moe_config=moe_config,
        torch_dtype='float32')
    if tuner_config is not None:
        apply_tuner(model, tuner_config, gradient_accumulation_steps=train_config.gradient_accumulation_steps,
                    task_type=model_config.task_type)
    else:
        apply_full_param_freeze(model, train_config)
    model.set_processor(
        InputProcessor, padding_free=template_config.padding_free, padding_side=template_config.padding_side,
        cp_partition_mode=distributed_config.cp_partition_mode, task_type=model_config.task_type)
    model.set_template(template)
    # the policy's loss is the shared clipped surrogate ('ppo' maps to it in configure_rl_loss)
    configure_rl_loss(model, rl_config)
    configure_optimizer(
        model, train_config, num_training_steps=max_steps, distributed_config=distributed_config,
        tuner_config=tuner_config, plugin_config=None)
    configure_rl_metrics(model, rl_config)

    # The critic: a trainable seq_cls num_labels=1 value model, initialized from the first ORM
    # reward MODEL (TRL's choice) or the policy base; full-param, so no tuner_config reaches it.
    value_config = copy(model_config)
    value_config.task_type = 'seq_cls'
    value_config.num_labels = 1
    from swift.dev.rewards.scorer import split_channel

    orm_models = split_channel(list(reward_config.orm))[1] if reward_config is not None else []
    value_config.model = orm_models[0] if orm_models else model_config.model
    value_model = build_model(
        value_config, distributed_config, train_config, megatron_config=megatron_config,
        moe_config=moe_config, instance_id='value', torch_dtype='float32')
    value_model.set_processor(
        InputProcessor, padding_free=template_config.padding_free, padding_side=template_config.padding_side,
        cp_partition_mode=distributed_config.cp_partition_mode)
    value_model.set_template(template)
    configure_ppo_value_loss(value_model, rl_config)
    configure_optimizer(
        value_model, train_config, num_training_steps=max_steps, distributed_config=distributed_config,
        tuner_config=None, plugin_config=None)
    configure_ppo_value_metric(value_model, rl_config)

    colocate = rollout_config.sampler_mode == 'colocate'
    sampler = build_rl_sampler(
        model_config=model_config, rollout_config=rollout_config, template=template,
        tuner_config=tuner_config, resource_plan=resource_plan, colocate=colocate,
                   enable_sampling_replay=rl_config.enable_sampling_replay)
    adapter = None
    if resource_plan is not None and rollout_config.resource_mode == 'adaptive':
        from twinkle_agentic.async_rl import ClusterAdapter

        # the critic is never published, but a colocate-borrowing adaptive runtime must hand the
        # device over with BOTH training models offloaded, so it rides the adapter's handover.
        adapter = ClusterAdapter.build(
            model=model, sampler=sampler, plan=resource_plan, rollout_config=rollout_config,
            additional_models=(value_model, ))
    params: Dict[str, Any] = {'max_tokens': rl_config.max_completion_length}
    if generation_config is not None:
        for key in ('temperature', 'top_p', 'top_k'):
            value = getattr(generation_config, key, None)
            if value is not None:
                params[key] = value
    # One engine for every generation entry (PPO rolls out single-turn): disaggregated places it
    # on the CPU payload group; colocate keeps it local for the visible loop's generate.
    rollout = MultiTurnRollout(
        sampler, template, sampling_params=SamplingParams(**params),
        allow_partial_rollout=rollout_config.allow_partial_rollout,
        max_turns=1,
        max_trajectory_tokens=rollout_config.max_trajectory_tokens,
        remote_group=(None if colocate else REF_ONLY_REMOTE_GROUP))
    engine = adapter.dedicated_manager if adapter is not None else CheckpointEngineManager(
        model=model, sampler=sampler, platform='GPU', mode=('colocate' if colocate else 'standalone'))

    # PPO ALWAYS anchors the KL: the full-parameter default is a frozen policy-init copy on the ref
    # group; a LoRA run reuses the policy actor itself with its adapter off (identity-derived).
    reference = build_ref_model(
        name=REF_GROUP, model_id=(rl_config.ref_model or model_config.model), model_config=model_config,
        template=template, distributed_config=distributed_config,
        device_mesh=auxiliary_mesh(REF_GROUP, (
            (rl_config.reference_parallel_spec or distributed_config.parallel_spec)
            if rl_config.sync_ref_model else rl_config.ref_parallel_spec), auxiliary_meshes),
        parallel_spec=((rl_config.reference_parallel_spec or distributed_config.parallel_spec)
                       if rl_config.sync_ref_model else rl_config.ref_parallel_spec),
        adapters=tuple(rl_config.ref_adapters or ()), policy_model=model,
        disable_adapter=(tuner_config is not None and not rl_config.ref_adapters
                         and not rl_config.sync_ref_model),
        model_type=rl_config.ref_model_type, model_revision=rl_config.ref_model_revision)

    backend = rollout_config.rollout_sampler
    orm_scorers = resolve_reward_scorers(
        'orm', reward_config=reward_config, model_config=model_config, template_config=template_config,
        distributed_config=distributed_config, rollout_config=rollout_config, backend=backend,
        sampler=sampler, quantize_config=quantize_config)
    worker = RewardWorker(orm_scorers=orm_scorers, template=template)
    return Cluster(
        model=model, value_model=value_model, sampler=sampler, rollout=rollout, engine=engine,
        adapter=adapter, reference=reference, worker=worker, sampling_params=params,
        colocate=colocate)


def build_ppo_driver(*, cluster: Cluster, rl_config: 'RLConfig', rollout_config: 'RolloutConfig',
                     tuner_config: Optional['TunerConfig'], train_batch_size: int,
                     gradient_accumulation_steps: int, prompt_dataloader: Any,
                     num_train_epochs: float, max_steps: int,
                     output_dir: str) -> Any:
    """Assemble twinkle's StreamingDriver for PPO: the per-sample pull stream, batch-scored.

    PPO has no group contract (per-token GAE scores each trajectory on its own), so the pull rule
    is group_pull=False with no group-completion verdict -- the driver batches the ORM scoring at
    the pull boundary via the remote ``score`` shape, and the recipe lifts each pull through the GAE
    pipeline. No sync-only feature exists either: every async_mode rides this one stream.
    """
    from twinkle_agentic.async_rl import StreamingDriver

    from swift.dev.recipe.utils import check_streaming_config

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
    num_generations = max(1, rl_config.num_generations)
    need_rows = train_batch_size * gradient_accumulation_steps
    # Stream geometry: the buffer must hold one full pull quantum (lcm with the admission width)
    # or assembly could deadlock under the backpressure cap.
    groups_per_partition = max(1, rollout_config.parameter_sync_step)
    pull_rows = need_rows * num_generations // math.gcd(need_rows, num_generations)
    buffer_depth = max((max_staleness + 1) * groups_per_partition * num_generations, pull_rows)
    return StreamingDriver(
        version_control=True,
        prompt_stream=prompt_dataloader,
        num_train_epochs=num_train_epochs,
        max_staleness=max_staleness,
        num_generations=num_generations,
        groups_per_partition=groups_per_partition,
        buffer_depth=buffer_depth,
        rollout=cluster.rollout,
        group_pull=False,
        need_rows=need_rows,
        snapshot_model=(cluster.model if strategy == 'adapter_snapshot' else None),
        snapshot_dir=os.path.join(output_dir, 'async_adapters'),
        adapter_name=adapter_name,
        max_steps=max_steps,
        consume=None,
        score=cluster.worker,
        resource_adapter=cluster.adapter,
        regenerate=None,
        baseline_fn=None,
        max_group_regenerations=0)
