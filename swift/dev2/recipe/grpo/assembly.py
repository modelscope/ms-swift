# Copyright (c) ModelScope Contributors. All rights reserved.
# GRPO's device and cluster assembly: prompt loading, the init halves, the StreamingDriver seams.
# Per-recipe by design (the visible training loop lives in run_grpo.py).
from __future__ import annotations

import math
import os
from copy import copy
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from swift.dev.utils.logger import get_logger  # TODO(dev2): bridge until the utils batch

if TYPE_CHECKING:
    from swift.dev2.config import (
        CheckpointConfig, DatasetConfig, DistributedConfig, LoggingConfig, MegatronConfig, ModelConfig,
        MoEConfig, QuantizeConfig, RLConfig, RewardConfig, RolloutConfig, TemplateConfig, TrainConfig,
        TunerConfig)

logger = get_logger()

MODEL_GROUP = 'model'
SAMPLER_GROUP = 'sampler'
#: A separate frozen teacher (GRPO's RLSD/SDAR channel) is its own Ray DeviceGroup; a
#: disable_adapter teacher reuses the policy actor and plans no group.
TEACHER_GROUP = 'teacher'
#: A FULL-PARAMETER frozen reference is an independent weight copy: its own DeviceGroup. A LoRA /
#: disable_adapter reference reuses the policy's base and plans no group.
REF_GROUP = 'ref'


def rollout_step_budget(*, num_prompts: int, num_generations: int, train_batch_size: int,
                        gradient_accumulation_steps: int, num_train_epochs: float,
                        generation_batch_size: Optional[int] = None, num_iterations: int = 1) -> int:
    """Optimizer steps for ``num_train_epochs`` passes over the prompt set (the scheduler horizon).

    Each generation batch rolls out ``batch_size * num_generations`` completions, splits into full
    ``train_batch_size`` mini-batches (an undersized tail is dropped), and replays them
    ``num_iterations`` times. Returns 0 when a generation batch cannot fill even one mini-batch.
    """
    # TODO(dev2): bridge until the recipe/common batch migrates num_optimizer_steps
    from swift.dev.recipe.common.loop_utils import num_optimizer_steps

    batch_size = min(generation_batch_size or num_prompts, num_prompts)
    batches_per_epoch = math.ceil(num_prompts / batch_size)
    total_batches = max(1, math.ceil(num_train_epochs * batches_per_epoch))
    mini_batches = (batch_size * num_generations) // max(1, train_batch_size)
    micro_per_batch = mini_batches * max(1, num_iterations)
    return num_optimizer_steps(micro_per_batch * total_batches, gradient_accumulation_steps)


def resolve_rollout_max_steps(explicit_max_steps: int, budget: int, *, recipe: str) -> int:
    """Explicit ``--max_steps`` wins; a non-positive budget fails loudly instead of a 0-step run."""
    if explicit_max_steps and explicit_max_steps > 0:
        return explicit_max_steps
    if budget <= 0:
        raise ValueError(
            f'{recipe}: derived {budget} training steps -- a generation batch cannot fill one '
            'train_batch_size (per_device_train_batch_size * dp_size) mini-batch. Enlarge the prompt '
            'dataset or num_generations, lower per_device_train_batch_size / the DP world size, or set '
            '--max_steps explicitly.')
    return budget


def prompt_rows_from_dataset(dataset_config: 'DatasetConfig') -> Tuple[List[Dict[str, Any]], List[List[dict]], List[Dict[str, Any]]]:
    """Load prompt rows, the prompt message views, and the non-message columns for rewards.

    Returns ``(rows, prompts, extras)``: ``rows`` is the raw loaded rows (the resource estimator reads
    the prompt lengths from them), ``prompts`` the message lists with any trailing assistant turn
    stripped, ``extras`` everything else (solution/target/teacher_prompt/...) keyed per prompt.
    """
    from swift.dev2.builders import load_prompt_only_rows

    rows = load_prompt_only_rows(dataset_config, None, split_dataset_ratio=0.0)
    if not rows:
        raise ValueError('online RL got an empty dataset. Set DatasetConfig.dataset with prompts to roll out on.')
    prompts: List[List[dict]] = []
    extras: List[Dict[str, Any]] = []
    for row in rows:
        messages = row.get('messages') if isinstance(row, dict) else None
        if not messages:
            continue
        if messages[-1].get('role') == 'assistant':
            messages = messages[:-1]
        prompts.append(list(messages))
        extras.append({key: value for key, value in row.items() if key != 'messages'})
    if not prompts:
        raise ValueError('online RL found no prompt messages in the dataset rows (expected a `messages` column).')
    return rows, prompts, extras


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


def _rl_tools(config: Any) -> Tuple[Any, List[Any], Any]:
    """(env_pool, tool_plugins, tool_manager_for) for the rollout engine, built engine-side."""
    from swift.dev2.builders.sandbox import build_tool_sandbox, tool_manager_for

    env_pool, plugins = build_tool_sandbox(config)
    return env_pool, plugins, tool_manager_for


def grpo_group_requests(rl_config: 'RLConfig', reward_config: 'RewardConfig',
                        distributed_config: 'DistributedConfig',
                        tuner_config: Optional['TunerConfig']) -> List[Tuple[str, Optional[str]]]:
    """The disjoint DeviceGroups GRPO's full-parameter frozen models need -> ``[(name, spec)]``.

    Mirrors the builds in :func:`initialize_cluster` exactly: the KL reference (active iff beta > 0
    and calculate_KL is not False; under sync_ref_model it inherits the policy's sharding for the
    paired EMA), the one RLSD/SDAR teacher, each MODEL id in the orm channel, and the prm channel's
    model. A disable_adapter auxiliary reuses the policy actor and plans NO group.
    """
    # TODO(dev2): bridge until the recipe/common batch migrates single_teacher_id
    from swift.dev.recipe.common.assembly import single_teacher_id
    from swift.dev2.rewards.scorer import split_channel

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


def plan_device_groups(nproc_per_node: int, sampler_mode: Optional[str], sampler_world_size: int,
                       auxiliary: Sequence[Tuple[str, int]]) -> Tuple[List[Tuple[str, List[int]]], bool]:
    """Plan the twinkle DeviceGroups for an online-RL run (pure: no twinkle import).

    Colocate overlays trainer + sampler in ONE group over the trainer's GPUs (independent rank
    spaces); anything else appends a disjoint sampler group, and each auxiliary entry is appended
    last, disjoint, in order. Returns ``(groups, colocate)``.
    """
    if nproc_per_node is None or nproc_per_node < 1:
        raise ValueError(f'nproc_per_node must be >= 1 (the trainer GPU count), got {nproc_per_node!r}.')
    if sampler_world_size < 1:
        raise ValueError(f'sampler_world_size must be >= 1, got {sampler_world_size}.')
    if sampler_mode == 'colocate':
        if sampler_world_size > nproc_per_node:
            raise ValueError(f'colocate needs the sampler ({sampler_world_size} GPUs) to fit within the trainer '
                             f'GPUs ({nproc_per_node}). Use sampler_mode="disaggregated" for its own GPUs.')
        # The overlay has its own worker-rank space; reusing MODEL_GROUP would inherit the trainer's
        # worker count and DeviceMesh, wrong as soon as sampler TP/PP is wider than one GPU.
        groups = [(MODEL_GROUP, list(range(nproc_per_node))), (SAMPLER_GROUP, list(range(sampler_world_size)))]
        colocate, next_rank = True, nproc_per_node
    else:
        total = nproc_per_node + sampler_world_size
        groups = [
            (MODEL_GROUP, list(range(nproc_per_node))),
            (SAMPLER_GROUP, list(range(nproc_per_node, total))),
        ]
        colocate, next_rank = False, total
    reserved = {MODEL_GROUP, SAMPLER_GROUP}
    for name, world_size in auxiliary:
        if world_size <= 0:
            continue
        if name in reserved:
            raise ValueError(f'auxiliary DeviceGroup name {name!r} collides with a reserved group '
                             f'({sorted(reserved)}); each frozen model needs a unique name.')
        groups.append((name, list(range(next_rank, next_rank + world_size))))
        next_rank += world_size
    return groups, colocate


def initialize_devices(*, model_config: 'ModelConfig', template_config: 'TemplateConfig',
                       dataset_config: 'DatasetConfig', train_config: 'TrainConfig',
                       distributed_config: 'DistributedConfig', checkpoint_config: 'CheckpointConfig',
                       rollout_config: 'RolloutConfig', rl_config: 'RLConfig',
                       reward_config: 'RewardConfig', tuner_config: Optional['TunerConfig'],
                       logging_config: Optional['LoggingConfig'] = None,
                       quantize_config: Optional['QuantizeConfig'] = None,
                       megatron_config: Optional['MegatronConfig'] = None,
                       moe_config: Optional['MoEConfig'] = None, output_dir: str = 'output',
                       prompt_rows: Sequence[Dict[str, Any]] = ()) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """Prepare the configs, build the template, and initialize twinkle + perf.

    Returns ``(template, processor, resource_plan, auxiliary_meshes)``: the plan is the
    ClusterAdapter's (already APPLIED -- derived parallel sizes written back into the configs
    before twinkle.initialize reserves the GPUs) or ``None``; the meshes map each auxiliary group
    to the DeviceMesh its model shards with (empty under an estimate plan).
    """
    import twinkle
    from twinkle import DeviceGroup, DeviceMesh
    from twinkle.perf import Perf

    # TODO(dev2): bridge until the recipe/common batch migrates TrainAssembly's static helper
    from swift.dev.recipe.common.assembly import TrainAssembly
    from swift.dev2.config import process_configs, validate_configs
    from swift.dev2.builders import build_dp_mesh, build_template, load_model_processor
    from twinkle_agentic.async_rl import REF_ONLY_REMOTE_GROUP, estimate_plan, resource_device_groups

    process_configs(
        model_config, template_config, dataset_config, train_config, distributed_config,
        checkpoint_config, tuner_config, rl_config, rollout_config=rollout_config,
        megatron_config=megatron_config, quantize_config=quantize_config, plugin_config=None)
    validate_configs(
        model_config, template_config, dataset_config, train_config, distributed_config,
        checkpoint_config, tuner_config, rl_config, reward_config=reward_config,
        rollout_config=rollout_config, logging_config=logging_config, quantize_config=quantize_config,
        megatron_config=megatron_config, moe_config=moe_config)
    _, processor = load_model_processor(model_config)
    task_type = model_config.task_type or 'causal_lm'
    template = build_template(template_config, processor, task_type=task_type)
    TrainAssembly.write_ckpt_args_json(
        output_dir, processor, model_config, template_config, tuner_config, quantize_config,
        task_type=task_type)

    backend = rollout_config.rollout_sampler
    if backend not in ('vllm', 'sglang'):
        raise ValueError(f"rollout_sampler={backend!r} cannot back an online-RL rollout: weight sync into the "
                         "sampler needs a CheckpointEngineMixin engine. Use 'vllm' or 'sglang'.")
    requests = grpo_group_requests(rl_config, reward_config, distributed_config, tuner_config)
    # One mesh per auxiliary, built once: the group's rank count and the model's own sharding both
    # read this object, so they cannot drift apart. The estimator takes the ORIGINAL spec sizes as
    # hints; an estimate plan rewrites the specs, so the dict is dropped there and the cluster build
    # re-derives each mesh from the rewritten spec.
    auxiliary_meshes: Dict[str, Any] = {
        name: (DeviceMesh.from_spec(spec) if spec else None) for name, spec in requests}
    resource_plan = estimate_plan(
        model_config=model_config, distributed_config=distributed_config, rollout_config=rollout_config,
        template=template, prompt_rows=prompt_rows,
        auxiliary_groups=[(name, (mesh.world_size if mesh else 1)) for name, mesh in auxiliary_meshes.items()],
        max_completion_length=rl_config.max_completion_length)
    if backend == 'sglang':
        sampler_width = rollout_config.sglang_tp_size * rollout_config.sglang_pp_size
        sampler_replicas = rollout_config.sglang_dp_size
    else:
        sampler_width = rollout_config.vllm_tensor_parallel_size * rollout_config.vllm_pipeline_parallel_size
        sampler_replicas = rollout_config.vllm_data_parallel_size
    if resource_plan is None:
        groups, colocate = plan_device_groups(
            distributed_config.nproc_per_node, rollout_config.sampler_mode,
            sampler_width * sampler_replicas,
            [(name, (mesh.world_size if mesh else 1)) for name, mesh in auxiliary_meshes.items()])
    else:
        groups = resource_device_groups(resource_plan, adaptive=rollout_config.resource_mode == 'adaptive')
        colocate = False
        auxiliary_meshes = {}
        logger.info('RL resource plan:\n%s', resource_plan.describe())
    if distributed_config.mode != 'ray':
        raise ValueError("online RL requires DistributedConfig.mode='ray': the trainer and the rollout sampler "
                         'are separate Ray actors that the driver drives and syncs weights between.')
    allocated = set()
    device_groups = []
    for group_name, ranks in groups:
        rank_set = set(ranks)
        shared = group_name != MODEL_GROUP and bool(rank_set) and rank_set.issubset(allocated)
        # A sampler group's workers each span the engine's parallel width; every other group is one
        # GPU per rank.
        sampler_group = (group_name in (SAMPLER_GROUP, 'shared_sampler') or group_name.startswith('sampler_')
                         or group_name.startswith('shared_sampler_') or group_name.startswith('flex_sampler_'))
        width = sampler_width if sampler_group else 1
        device_groups.append(
            DeviceGroup(name=group_name, ranks=ranks, device_type='GPU', gpus_per_worker=width, shared=shared))
        if not shared:
            allocated.update(rank_set)
    total = max(allocated) + 1 if allocated else 0
    twinkle.initialize(
        mode='ray',
        nproc_per_node=total,
        extra_torch_kwargs={'seed': train_config.seed, 'full_determinism': train_config.full_determinism,
                            'tf32': train_config.tf32},
        global_device_mesh=build_dp_mesh(distributed_config, template_config.sequence_parallel_size),
        name=distributed_config.ray_exp_name,
        groups=[*device_groups, DeviceGroup(name=REF_ONLY_REMOTE_GROUP, ranks=1, device_type='CPU')])
    logger.info('twinkle initialized with DeviceGroups: %s', [(g.name, g.ranks) for g in device_groups])
    Perf.initialize({})  # the cluster's actors register themselves once built (see run_grpo)
    return template, processor, resource_plan, auxiliary_meshes


class Cluster(NamedTuple):
    """Every actor and handle GRPO's visible loop drives (a plain handle bundle, not a layer)."""

    template: Any
    processor: Any
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
                       processor: Any,
                       train_config: 'TrainConfig', distributed_config: 'DistributedConfig',
                       checkpoint_config: Optional['CheckpointConfig'],
                       rollout_config: 'RolloutConfig', rl_config: 'RLConfig',
                       reward_config: 'RewardConfig', tuner_config: Optional['TunerConfig'],
                       max_steps: int, quantize_config: Optional['QuantizeConfig'] = None,
                       megatron_config: Optional['MegatronConfig'] = None,
                       moe_config: Optional['MoEConfig'] = None,
                       generation_config: Optional[Any] = None, engine_args: Optional[Dict[str, Any]] = None,
                       resource_plan: Any = None, auxiliary_meshes: Optional[Dict[str, Any]] = None,
                       output_dir: str = 'output') -> Cluster:
    """Build every actor GRPO trains with, explicitly: policy, sampler, reference, teacher, rewards.

    Nothing is hidden behind a shared assembly class; PPO / the distillation recipes write their
    own. ``max_steps`` is the resolved step budget (the LR-scheduler horizon), derived by the caller
    from the prompt set BEFORE this stage so the optimizer is fully configured here.
    """
    from twinkle import DeviceMesh
    from twinkle.checkpoint_engine import CheckpointEngineManager

    from swift.dev2.builders import (
        apply_full_param_freeze, build_dp_mesh, build_engine_args, build_model, build_ref_model,
        build_reward_model, build_sampler)
    from swift.dev2.builders.adapter import apply_tuner
    from swift.dev2.builders.optimizer import configure_optimizer
    from swift.dev2.loss import configure_rl_loss, configure_rl_metrics
    from swift.dev2.processor import InputProcessor
    from twinkle_agentic.rollout import MultiTurnRollout
    # TODO(dev2): bridge until the recipe/common batch migrates single_teacher_id
    from swift.dev.recipe.common.assembly import single_teacher_id
    from swift.dev2.rewards.worker import RewardWorker
    from twinkle_agentic.async_rl import REF_ONLY_REMOTE_GROUP
    from twinkle.data_format import SamplingParams

    sp = template_config.sequence_parallel_size
    sp_mesh = build_dp_mesh(distributed_config, sp) if sp > 1 else None
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
        enable_router_replay=rl_config.router_replay_mode != 'disabled')
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
    # The optimizer reads the tuner's param groups and sizes its LR scheduler from the step budget,
    # so it runs after the tuner; the policy metric appends onto the optimizer's metric list and
    # MUST come after it (twinkle add_metric contract).
    configure_optimizer(
        model, train_config, num_training_steps=max_steps, distributed_config=distributed_config,
        tuner_config=tuner_config, plugin_config=None)
    configure_rl_metrics(model, rl_config)

    backend = rollout_config.rollout_sampler
    colocate = rollout_config.sampler_mode == 'colocate'
    sampler_engine_args = build_engine_args(backend, None, rollout_config)
    sampler_engine_args.update(engine_args or {})
    if colocate:
        sampler_engine_args['enable_sleep_mode' if backend == 'vllm' else 'enable_memory_saver'] = True
    if (backend == 'vllm' and tuner_config is not None and tuner_config.tuner == 'lora'
            and rollout_config.weight_sync_strategy == 'adapter_snapshot'):
        sampler_engine_args.setdefault('enable_lora', True)
        sampler_engine_args.setdefault('max_lora_rank', tuner_config.lora_rank)
        sampler_engine_args.setdefault('max_loras', 2)
    if rl_config.enable_sampling_replay:
        sampler_engine_args['enable_sampling_replay'] = True
    # create the samplers here -- a planned layout (estimate/adaptive) sizes the replica pool and
    # stamps the stable instance id the elastic runtime addresses; the static shape reads the
    # rollout_config replica knobs (estimate's apply_plan already rewrote them, so it agrees).
    if resource_plan is not None:
        sampler = build_sampler(
            model_config, backend=backend, engine_args=dict(sampler_engine_args),
            device_mesh=DeviceMesh.from_sizes(dp_size=resource_plan.sampler.replica_count), template=template,
            remote_group=SAMPLER_GROUP, instance_id='adaptive-sampler')
    else:
        if backend == 'sglang':
            replicas = rollout_config.sglang_dp_size
        else:
            replicas = rollout_config.vllm_data_parallel_size
        sampler = build_sampler(
            model_config, backend=backend, engine_args=sampler_engine_args,
            device_mesh=(None if colocate else DeviceMesh.from_sizes(dp_size=replicas)), template=template,
            remote_group=(MODEL_GROUP if colocate else SAMPLER_GROUP))
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
    # One engine for every generation entry: single-turn is the one-turn degenerate. A disaggregated
    # run places it on the CPU payload group (episode refs never enter the driver early); colocate
    # keeps it local so the visible loop's batch generate drives the sampler directly.
    rollout = MultiTurnRollout(
        sampler, template, sampling_params=SamplingParams(**params),
        allow_partial_rollout=rollout_config.allow_partial_rollout,
        max_turns=(rollout_config.max_turns or 1),
        max_trajectory_tokens=rollout_config.max_trajectory_tokens,
        tools_factory=_rl_tools, tools_config=rollout_config,
        remote_group=(None if colocate else REF_ONLY_REMOTE_GROUP))
    engine = adapter.dedicated_manager if adapter is not None else CheckpointEngineManager(
        model=model, sampler=sampler, platform='GPU', mode=('colocate' if colocate else 'standalone'))

    # The mesh each auxiliary builds with: the planned one when the plan laid one out, else
    # re-derived from the (estimate-rewritten) spec. Pure layout resolution, local to the recipe.
    def auxiliary_mesh(group: str, spec: Optional[str]) -> Optional[Any]:
        mesh = (auxiliary_meshes or {}).get(group)
        return mesh if mesh is not None else (DeviceMesh.from_spec(spec) if spec else None)

    # create the reference here (a KL anchor) -- disable_adapter reuses the policy actor
    reference = None
    if rl_config.beta and rl_config.calculate_KL is not False:
        ref_spec = ((rl_config.reference_parallel_spec or distributed_config.parallel_spec)
                    if rl_config.sync_ref_model else rl_config.ref_parallel_spec)
        reference = build_ref_model(
            name=REF_GROUP, model_id=rl_config.ref_model, model_config=model_config, template=template,
            distributed_config=distributed_config, device_mesh=auxiliary_mesh(REF_GROUP, ref_spec),
            parallel_spec=ref_spec, adapters=tuple(rl_config.ref_adapters or ()), policy_model=model,
            disable_adapter=(tuner_config is not None and not rl_config.ref_adapters
                             and not rl_config.sync_ref_model))
    # create the teacher here (the RLSD/SDAR reweighting channel)
    teacher = build_ref_model(
        name=TEACHER_GROUP, model_id=single_teacher_id(rl_config.teacher_model, algo='RLSD/SDAR'),
        model_config=model_config, template=template, distributed_config=distributed_config,
        device_mesh=auxiliary_mesh(TEACHER_GROUP, rl_config.teacher_parallel_spec),
        parallel_spec=rl_config.teacher_parallel_spec,
        adapters=tuple(rl_config.teacher_adapters or ()), policy_model=model,
        disable_adapter=rl_config._teacher_use_disable_adapter)

    # create prm & orm here -- rules resolve locally, each MODEL goes through the shared builder
    def _scorers(channel: str, prefix: str) -> List[Any]:
        from swift.dev2.rewards.reward import get_reward_funcs
        from swift.dev2.rewards.scorer import ChannelCompanions, resolve_channel

        plan = resolve_channel(
            list(getattr(reward_config, channel)),
            ChannelCompanions(
                adapters=getattr(reward_config, f'{channel}_adapters'),
                model_types=getattr(reward_config, f'{channel}_model_type'),
                model_revisions=getattr(reward_config, f'{channel}_model_revision'),
                template_names=getattr(reward_config, f'{channel}_template'),
                parallel_spec=getattr(reward_config, f'{channel}_parallel_spec')),
            channel=channel, sampler_model=model_config.model, backend=backend)
        scorers = list(get_reward_funcs(plan.rules, reward_config)[0])
        for index, spec in enumerate(plan.specs):
            if channel == 'prm' and spec.kind != 'scalar':
                raise ValueError(f'--prm model {spec.model_id!r} resolves to a {spec.kind!r} reward, but online RL '
                                 'places only a SCALAR (seq_cls/reranker) frozen PRM today.')
            scorers.append(build_reward_model(
                spec, model_config=model_config, template_config=template_config,
                distributed_config=distributed_config, rollout_config=rollout_config, backend=backend,
                sampler=sampler, engine_args=engine_args, remote_group=f'{prefix}_{index}',
                judge_template=reward_config.judge_template, quantize_config=quantize_config))
        return scorers

    orm_scorers = _scorers('orm', 'reward')
    prm_scorers = _scorers('prm', 'prm')
    prm_scorer = None
    if prm_scorers:
        from swift.dev.utils.plugin import PluginRegistry

        from swift.dev2.rewards import PRM_STEP_SCORER

        plugin_name = reward_config.prm_scorer or (
            'turn' if rollout_config.max_turns is not None else 'delimiter')
        prm_scorer = PluginRegistry.resolve(PRM_STEP_SCORER, plugin_name, config=reward_config)
    worker = RewardWorker(orm_scorers=orm_scorers, prm_scorers=prm_scorers,
                          prm_step_scorer=prm_scorer,
                          prm_weights=reward_config.prm_weights, template=template)
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
        template=template, processor=processor, model=model, sampler=sampler, rollout=rollout,
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

    # TODO(dev2): bridge until the recipe/common batch migrates check_streaming_config
    from swift.dev.recipe.rl.streaming import check_streaming_config
    from swift.dev2.rewards.advantage import build_reward_weights

    rollout = cluster.rollout
    worker = cluster.worker
    strategy = rollout_config.weight_sync_strategy
    max_staleness = (0 if rollout_config.async_mode == 'none' else 1
                     if rollout_config.async_mode == 'one_step_off' else rollout_config.max_staleness)
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

    # The two algorithm verdicts stay closures (their knobs -- dynamic_sample, the greedy remax
    # generation, the weight vectors -- are GRPO's own); the PRM scoring that used to share this
    # hook is the score_worker collaborator's own channel now.
    regenerate = None
    if rl_config.dynamic_sample:
        def regenerate(group_key: str, samples: List[Any]) -> bool:
            return group_variance([s.orm_reward for s in samples], orm_weights) <= 0.0

    baseline_fn = None
    if rl_config.advantage_estimator == 'remax':
        def baseline_fn(group_key: str, samples: List[Any]) -> float:
            """The greedy completion's weighted reward, generated on the PUBLISHED weights (remax)."""
            greedy = rollout.generate(
                [list(samples[0].prompt or [])], num_samples=1,
                sampling_params=greedy_params(cluster.sampling_params),
                prompt_extras=[dict(samples[0].extra)] if samples[0].extra else None)
            if len(greedy) != 1:
                raise RuntimeError(f'remax greedy baseline returned {len(greedy)} samples, expected 1.')
            greedy[0].prompt = list(samples[0].prompt or [])
            values = (worker.score_group([greedy[0]])[0] if cluster.orm_scorers else [])
            return weighted_reward(values, orm_weights)

    # Stream geometry: one publish cycle is expected to train ``parameter_sync_step`` groups, and the
    # buffer must hold at least one full pull quantum -- lcm(need_rows, num_generations) rows -- or
    # assembly could deadlock under the backpressure cap.
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
        regenerate=regenerate,
        baseline_fn=baseline_fn,
        max_group_regenerations=(rl_config.max_resample_times if rl_config.dynamic_sample else 0))
    return driver
