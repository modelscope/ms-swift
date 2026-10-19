# Copyright (c) ModelScope Contributors. All rights reserved.
# Distillation's assembly: teacher group requests, the dataset rows, and the explicit cluster build.
from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from swift.dev.utils.logger import get_logger

from swift.dev.recipe.distill.worker import DistillRow, DistillWorker
from swift.dev.recipe.utils import (
    TEACHER_GROUP, auxiliary_mesh, build_rl_sampler, teacher_group)

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig, DatasetConfig, DistributedConfig, MegatronConfig, ModelConfig, MoEConfig,
        QuantizeConfig, RLConfig, RolloutConfig, TemplateConfig, TrainConfig, TunerConfig)

logger = get_logger()

#: Each signal form's teacher count and GKD-only knobs, keyed by the recipe's rl_type.
DISTILL_SIGNALS = {'gkd': 'logits', 'opsd': 'logps', 'mopd': 'multi_logps'}


def distill_dataset_rows(template: Any, raw_rows: Sequence[Dict[str, Any]]) -> List[DistillRow]:
    """The off-policy training rows: dataset rows that carry their own assistant completion.

    Each is encoded once with the student's template (its labels mark the completion's trainable
    positions) and keeps the full conversation plus passthrough extras for the teacher view.
    """
    dataset_rows: List[DistillRow] = []
    for row in raw_rows:
        messages = row.get('messages') if isinstance(row, dict) else None
        if not messages or messages[-1].get('role') != 'assistant':
            continue
        extra = {key: copy.deepcopy(value) for key, value in row.items() if key != 'messages'}
        dataset_rows.append(DistillRow(template.encode(row), copy.deepcopy(messages), extra))
    return dataset_rows


def distill_sampling_params(rl_config: Any, generation_config: Any) -> Dict[str, Any]:
    """SamplingParams dict for the on-policy distillation rollout (shared by gkd/opsd/mopd).

    The rollout uses ``sampling_temperature``, NOT the distillation ``distill_temperature``: those
    are separate roles. ``sampling_temperature=None`` falls back to ``distill_temperature`` so
    pre-split runs are byte-identical; setting it decouples how hot the student samples from how
    softly the divergence scores.
    """
    sampling_temperature = rl_config.sampling_temperature
    if sampling_temperature is None:
        sampling_temperature = rl_config.distill_temperature
    params: Dict[str, Any] = {
        'max_tokens': rl_config.max_completion_length,
        'temperature': sampling_temperature,
    }
    if generation_config is not None:
        if generation_config.top_p is not None:
            params['top_p'] = generation_config.top_p
        if generation_config.top_k is not None:
            params['top_k'] = generation_config.top_k
    return params


def distill_group_requests(rl_config: 'RLConfig') -> List[Tuple[str, Optional[str]]]:
    """The frozen-teacher DeviceGroups this run needs -> ``[(name, teacher_parallel_spec)]``.

    Mirrors the teacher builds in :func:`initialize_distill_cluster` exactly: a disable_adapter /
    dynamic-self teacher reuses the policy actor and plans NO group; MOPD's ``spread`` placement
    gives each of its K teachers its own ``teacher_<i>`` group, ``pack`` (the default) shares ONE.
    """
    if rl_config.rl_type == 'mopd':
        spread = rl_config.auxiliary_placement == 'spread'
        return ([(teacher_group(index), rl_config.teacher_parallel_spec)
                 for index in range(len(rl_config.teacher_model or []))] if spread
                else [(TEACHER_GROUP, rl_config.teacher_parallel_spec)])
    from swift.dev.recipe.utils import single_teacher_id

    separate = (not rl_config._teacher_use_disable_adapter
                and single_teacher_id(rl_config.teacher_model, algo='distill') is not None)
    return [(TEACHER_GROUP, rl_config.teacher_parallel_spec)] if separate else []


class Cluster(NamedTuple):
    """Every actor and handle the distillation loop drives (a plain handle bundle, not a layer)."""

    model: Any
    sampler: Any
    rollout: Any
    engine: Any
    adapter: Any
    teachers: List[Any]
    worker: Any
    sampling_params: Dict[str, Any]
    colocate: bool


def initialize_distill_cluster(*, model_config: 'ModelConfig', template_config: 'TemplateConfig',
                               template: Any, train_config: 'TrainConfig',
                               distributed_config: 'DistributedConfig',
                               checkpoint_config: Optional['CheckpointConfig'],
                               rollout_config: 'RolloutConfig', rl_config: 'RLConfig',
                               tuner_config: Optional['TunerConfig'], max_steps: int,
                               signal: str, train_batch_size: int, gradient_accumulation_steps: int,
                               quantize_config: Optional['QuantizeConfig'] = None,
                               megatron_config: Optional['MegatronConfig'] = None,
                               moe_config: Optional['MoEConfig'] = None,
                               generation_config: Optional[Any] = None,
                               resource_plan: Any = None, auxiliary_meshes: Optional[Dict[str, Any]] = None,
                               output_dir: str = 'output') -> Cluster:
    """Build every actor distillation trains with, explicitly: policy, sampler, teachers, worker.

    Nothing is hidden behind a shared assembly class (deep_refactor's initialize_cluster note).
    ``max_steps`` is the resolved step budget derived by the caller BEFORE this stage, so the
    optimizer's LR scheduler is fully configured here.
    """
    from twinkle.checkpoint_engine import CheckpointEngineManager

    from swift.dev.builders import (
        apply_full_param_freeze, build_dp_mesh, build_model, build_ref_model)
    from swift.dev.builders.adapter import apply_tuner
    from swift.dev.builders.optimizer import configure_optimizer
    from swift.dev.loss import configure_rl_loss
    from swift.dev.processor import InputProcessor
    from twinkle_agentic.rollout import MultiTurnRollout
    from twinkle_agentic.async_rl import REF_ONLY_REMOTE_GROUP
    from twinkle.data_format import SamplingParams

    sp = template_config.sequence_parallel_size
    sp_mesh = build_dp_mesh(distributed_config, sp) if sp > 1 else None
    resume_dir = checkpoint_config.resume_from_checkpoint if checkpoint_config is not None else None
    policy_config = model_config
    if resume_dir and tuner_config is None:
        # A full-parameter resume loads weights from the ckpt dir instead of the original model id;
        # a tuned policy keeps the original base and reloads its adapter via resume (run_distill).
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
    configure_rl_loss(model, rl_config)
    # The optimizer reads the tuner's param groups and sizes its LR scheduler from the step budget;
    # the policy metric appends onto the optimizer's metric list (twinkle add_metric contract).
    configure_optimizer(
        model, train_config, num_training_steps=max_steps, distributed_config=distributed_config,
        tuner_config=tuner_config, plugin_config=None)

    colocate = rollout_config.sampler_mode == 'colocate'
    sampler = build_rl_sampler(
        model_config=model_config, rollout_config=rollout_config, template=template,
        tuner_config=tuner_config, resource_plan=resource_plan, colocate=colocate)
    adapter = None
    if resource_plan is not None and rollout_config.resource_mode == 'adaptive':
        from twinkle_agentic.async_rl import ClusterAdapter

        adapter = ClusterAdapter.build(
            model=model, sampler=sampler, plan=resource_plan, rollout_config=rollout_config)
    params = distill_sampling_params(rl_config, generation_config)
    # One engine for every generation entry (distillation rolls out single-turn): disaggregated
    # places it on the CPU payload group; colocate keeps it local for the visible loop's generate.
    rollout = MultiTurnRollout(
        sampler, template, sampling_params=SamplingParams(**params),
        allow_partial_rollout=rollout_config.allow_partial_rollout,
        max_turns=1,
        max_trajectory_tokens=rollout_config.max_trajectory_tokens,
        remote_group=(None if colocate else REF_ONLY_REMOTE_GROUP))
    engine = adapter.dedicated_manager if adapter is not None else CheckpointEngineManager(
        model=model, sampler=sampler, platform='GPU', mode=('colocate' if colocate else 'standalone'))

    teachers, teacher_options = _build_teachers(
        model=model, model_config=model_config, template=template, rl_config=rl_config,
        tuner_config=tuner_config, distributed_config=distributed_config, signal=signal,
        auxiliary_meshes=auxiliary_meshes)
    worker = DistillWorker(
        teachers=teachers, signal=signal, template=template,
        gkd_logits_topk=rl_config.gkd_logits_topk, sft_alpha=rl_config.sft_alpha,
        train_batch_size=train_batch_size, gradient_accumulation_steps=gradient_accumulation_steps,
        **teacher_options)
    return Cluster(
        model=model, sampler=sampler, rollout=rollout, engine=engine, adapter=adapter,
        teachers=teachers, worker=worker, sampling_params=params, colocate=colocate)


def _build_teachers(*, model: Any, model_config: 'ModelConfig', template: Any, rl_config: 'RLConfig',
                    tuner_config: Optional['TunerConfig'], distributed_config: 'DistributedConfig',
                    signal: str, auxiliary_meshes: Optional[Dict[str, Any]]) -> Tuple[List[Any], Dict[str, Any]]:
    """Resolve the bare frozen teachers plus the plain worker options that describe them.

    GKD/OPSD build ONE teacher: the LoRA student's adapter-disabled base (the policy itself, so the
    worker forwards it with ``disable_lora``), a separate frozen model (``teacher_model``), or --
    OPSD's default -- the student's CURRENT weights (dynamic self-distillation). MOPD builds K.
    """
    from swift.dev.builders import build_ref_model

    from swift.dev.recipe.utils import single_teacher_id

    if signal == 'multi_logps':
        if rl_config._teacher_use_disable_adapter:
            raise ValueError('MOPD distils from K separate domain teachers; _teacher_use_disable_adapter '
                             '(a single adapter-disabled base) is an OPSD/GKD self-distillation mode and '
                             'does not apply here.')
        if rl_config.teacher_adapters:
            raise ValueError('MOPD builds each teacher as a distinct full model and does not apply '
                             'per-teacher adapters; teacher_adapters would be ambiguous. Remove it, or use '
                             'OPSD for a single adapter-based teacher.')
        if rl_config.offload_teacher_model:
            raise ValueError('offload_teacher_model is not supported for MOPD: it keeps K teachers resident '
                             'on their own DeviceGroup to score every trajectory. Leave it off.')
        teacher_models = rl_config.teacher_model
        if not teacher_models:
            raise ValueError('MOPD needs at least one teacher: set RLConfig.teacher_model to a list of '
                             'model ids (one per domain expert).')
        spread = rl_config.auxiliary_placement == 'spread'
        spec = rl_config.teacher_parallel_spec
        teachers = [
            build_ref_model(
                name=(teacher_group(index) if spread else TEACHER_GROUP), model_id=model_id,
                model_config=model_config, template=template, distributed_config=distributed_config,
                device_mesh=auxiliary_mesh((teacher_group(index) if spread else TEACHER_GROUP), spec,
                                           auxiliary_meshes),
                parallel_spec=spec, policy_model=model,
                model_type=rl_config.teacher_model_type, model_revision=rl_config.teacher_model_revision,
                instance_id=f'teacher{index}', deepspeed=rl_config.teacher_deepspeed)
            for index, model_id in enumerate(teacher_models)]
        weights = None if rl_config.teacher_weights is None else list(rl_config.teacher_weights)
        if weights is not None and len(weights) != len(teachers):
            raise ValueError(f'teacher_weights has {len(weights)} entries but {len(teachers)} teachers were '
                             'built; they must match, or leave teacher_weights unset for uniform weights.')
        # pack shares the ONE teacher group (serial forwards); spread gives each its own (parallel).
        placements = [(teacher_group(index) if spread else TEACHER_GROUP) for index in range(len(teachers))]
        return teachers, {'placements': placements, 'teacher_weights': weights}

    # GKD / OPSD: one teacher, three shapes.
    if signal == 'logits' and rl_config.offload_teacher_model and rl_config._teacher_use_disable_adapter:
        raise ValueError('offload_teacher_model requires a distinct frozen teacher model, not the '
                         'adapter-disabled base of the student (which has no separate weights to offload).')
    if signal == 'logps' and rl_config.offload_teacher_model:
        raise ValueError('offload_teacher_model is GKD-only: its full-vocab teacher forward wraps '
                         'reload/offload, while OPSD scores response-only teacher_logps via a chunked '
                         'forward_only with no offload hook. Leave it off.')
    if rl_config._teacher_use_disable_adapter:
        if tuner_config is None:
            raise ValueError('rl_config._teacher_use_disable_adapter=True requires a LoRA student '
                             '(tuner_config), since it distils from the adapter-disabled base of that '
                             'same model.')
        # the teacher IS the policy actor with its adapter off -- the worker forwards it disable_lora
        return [model], {'disable_lora': True}
    teacher_model = single_teacher_id(rl_config.teacher_model, algo='distill')
    if teacher_model is None:
        if signal == 'logits':
            raise ValueError('GKD needs a teacher: set RLConfig.teacher_model, or use a LoRA student '
                             'with _teacher_use_disable_adapter=True to distil from its own frozen base.')
        # OPSD's default: dynamic self-distillation -- the student's CURRENT weights score the
        # privileged view on the policy actor (no separate group, adapters ON).
        return [model], {}
    offload = bool(signal == 'logits' and rl_config.offload_teacher_model)
    teacher = build_ref_model(
        name=TEACHER_GROUP, model_id=teacher_model, model_config=model_config, template=template,
        distributed_config=distributed_config,
        device_mesh=auxiliary_mesh(TEACHER_GROUP, rl_config.teacher_parallel_spec, auxiliary_meshes),
        parallel_spec=rl_config.teacher_parallel_spec,
        adapters=tuple(rl_config.teacher_adapters or ()), policy_model=model,
        model_type=rl_config.teacher_model_type, model_revision=rl_config.teacher_model_revision,
        deepspeed=rl_config.teacher_deepspeed)
    if offload:
        # GKD's full-vocab memory hook: keep the teacher on CPU between forwards.
        teacher.offload_to_cpu()
    return [teacher], {'offload': offload}


def build_distill_driver(*, cluster: Cluster, rl_config: 'RLConfig', rollout_config: 'RolloutConfig',
                         tuner_config: Optional['TunerConfig'], train_batch_size: int,
                         gradient_accumulation_steps: int, prompt_dataloader: Any,
                         num_train_epochs: float, max_steps: int,
                         output_dir: str, run_id: str = 'run_distill') -> Any:
    """Assemble twinkle's StreamingDriver over the cluster for a purely on-policy run (lmbda == 1).

    Distillation scores no outcome channel, so no score_worker rides the barrier and no group
    verdict exists -- the driver streams samples one by one (``unit_size=1``), the recipe lifts each
    pull into rows, and the teacher scoring happens in the DistillWorker per mini-batch.
    """
    import os

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
        run_id=run_id,
        prompt_stream=prompt_dataloader,
        num_train_epochs=num_train_epochs,
        max_staleness=max_staleness,
        num_generations=num_generations,
        groups_per_partition=groups_per_partition,
        buffer_depth=buffer_depth,
        rollout=cluster.rollout,
        pull_rule='prefix',
        need_rows=need_rows,
        snapshot_model=(cluster.model if strategy == 'adapter_snapshot' else None),
        snapshot_dir=os.path.join(output_dir, 'async_adapters'),
        adapter_name=adapter_name,
        max_steps=max_steps,
        consume=None,
        score=None,
        score_worker=None,
        resource_adapter=cluster.adapter,
        regenerate=None,
        baseline_fn=None,
        max_group_regenerations=0)
