# Copyright (c) ModelScope Contributors. All rights reserved.
# ruff: noqa: C901
# GRPO's device and cluster assembly: the init halves and the StreamingDriver seams (per-recipe).
from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from swift.dev.recipe.utils import REF_GROUP, TEACHER_GROUP, auxiliary_mesh, build_rl_sampler, resolve_reward_scorers
from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DistributedConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        QuantizeConfig,
        RewardConfig,
        RLConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

logger = get_logger()


def greedy_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """argmax decoding for a remax baseline: temperature 0 and no nucleus/top-k."""
    return dict(params, temperature=0.0, top_p=1.0, top_k=-1)


def group_variance(rewards_block: List[List[Any]], weights: Any) -> float:
    """The weighted-reward std of one group (dynamic_sample's zero-variance test)."""
    import torch

    if len(rewards_block) <= 1:
        return 0.0
    matrix = torch.tensor(
        [[0.0 if value is None or value != value else float(value) for value in row] or [0.0] for row in rewards_block],
        dtype=torch.float32,
    )
    if weights is not None and matrix.shape[1] == len(weights):
        matrix = matrix * weights.unsqueeze(0)
    weighted = matrix.nansum(dim=1)
    return float(torch.std(weighted, unbiased=False)) if weighted.numel() > 1 else 0.0


def grpo_group_requests(
    rl_config: "RLConfig",
    reward_config: "RewardConfig",
    distributed_config: "DistributedConfig",
    tuner_config: Optional["TunerConfig"],
) -> List[Tuple[str, Optional[str]]]:
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
            spec = (
                (rl_config.reference_parallel_spec or distributed_config.parallel_spec)
                if rl_config.sync_ref_model
                else rl_config.ref_parallel_spec
            )
            requests.append((REF_GROUP, spec))
    teacher_id = single_teacher_id(rl_config.teacher_model, algo="RLSD/SDAR")
    if teacher_id and not rl_config._teacher_use_disable_adapter:
        requests.append((TEACHER_GROUP, rl_config.teacher_parallel_spec))
    for index in range(len(split_channel(list(reward_config.orm))[1])):
        requests.append((f"reward_{index}", reward_config.orm_parallel_spec))
    if split_channel(list(reward_config.prm))[1]:
        requests.append(("prm_0", reward_config.prm_parallel_spec))
    return requests


class Cluster(NamedTuple):
    """Every actor and handle GRPO's visible loop drives (a plain handle bundle, not a layer).

    The signal components (``scores`` ... ``routing``) read the trajectories where they live; an
    option that is off leaves its component ``None``.
    """

    model: Any
    sampler: Any
    rollout: Any
    engine: Any
    adapter: Any
    reference: Any
    teacher: Any
    orm_scorers: List[Any]
    orm_names: List[str]
    prm_scorers: List[Any]
    sampling_params: Dict[str, Any]
    colocate: bool
    scores: Any
    kl: Any
    process_reward: Any
    combine: Any
    teacher_reweight: Any
    privileged: Any
    correction: Any
    routing: Any


def initialize_cluster(
    *,
    model_config: "ModelConfig",
    template_config: "TemplateConfig",
    template: Any,
    train_config: "TrainConfig",
    distributed_config: "DistributedConfig",
    checkpoint_config: Optional["CheckpointConfig"],
    rollout_config: "RolloutConfig",
    rl_config: "RLConfig",
    reward_config: "RewardConfig",
    tuner_config: Optional["TunerConfig"],
    max_steps: int,
    batch_size: int,
    quantize_config: Optional["QuantizeConfig"] = None,
    megatron_config: Optional["MegatronConfig"] = None,
    moe_config: Optional["MoEConfig"] = None,
    generation_config: Optional[Any] = None,
    resource_plan: Any = None,
    auxiliary_meshes: Optional[Dict[str, Any]] = None,
    teacher_prompt: bool = False,
) -> Cluster:
    """Build every actor GRPO trains with, explicitly: policy, sampler, rollout, reference, teacher, rewards.

    Nothing is hidden behind a shared assembly class; PPO / the distillation recipes write their
    own. ``max_steps`` is the resolved step budget (the LR-scheduler horizon), derived by the caller
    from the prompt set BEFORE this stage so the optimizer is fully configured here; ``batch_size``
    is the rows of one driver batch (it bounds the episodes in flight); ``teacher_prompt`` says the
    dataset carries a privileged ``teacher_prompt`` column.
    """
    from swift.dev.builders import apply_full_param_freeze, build_dp_mesh, build_model, build_ref_model
    from swift.dev.builders.adapter import apply_tuner
    from swift.dev.builders.optimizer import configure_optimizer
    from swift.dev.builders.streaming import (
        CONTROL_GROUP,
        build_checkpoint_manager,
        build_cluster_adapter,
        build_rollout,
        inflight_rows,
        rollout_sampling_params,
    )
    from swift.dev.loss import configure_rl_loss, configure_rl_metrics
    from swift.dev.processor import InputProcessor, processor_kwargs
    from swift.dev.recipe.utils import single_teacher_id
    from swift.dev.utils.scoring import PrivilegedView
    from twinkle.advantage import TeacherReweight
    from twinkle.metric import RolloutCorrection
    from twinkle.processor import RoutingReplay
    from twinkle.signals import KLEstimator, ResponseScores, SignalCombine
    from twinkle_agentic.async_rl import ProcessReward

    sp = template_config.sequence_parallel_size
    sp_mesh = build_dp_mesh(distributed_config, sp) if sp > 1 else None
    # Refuse the engine/training combinations a quantized refit cannot serve
    # (DeepSeek-V4 MXFP4 experts): cheaper now than after the engine is built.
    from swift.dev.weight_sync import check_quantized_weight_sync

    model_info = getattr(template, "model_info", None)
    check_quantized_weight_sync(
        getattr(model_info, "config", None), model_config.model_type, tuner_config, rollout_config
    )
    resume_dir = checkpoint_config.resume_from_checkpoint if checkpoint_config is not None else None
    policy_config = model_config
    if resume_dir and tuner_config is None:
        # A full-parameter resume loads weights from the ckpt dir instead of the original model id;
        # a tuned policy keeps the original base and reloads its adapter via resume (run_grpo).
        policy_config = copy.copy(model_config)
        policy_config.model = resume_dir
    # create the actor (the trainable policy) here
    model = build_model(
        policy_config,
        distributed_config,
        train_config,
        tuner_config,
        device_mesh=sp_mesh,
        quantize_config=quantize_config,
        megatron_config=megatron_config,
        moe_config=moe_config,
        enable_router_replay=rl_config.router_replay_mode != "disabled",
        torch_dtype="float32",
    )
    if tuner_config is not None:
        apply_tuner(
            model,
            tuner_config,
            gradient_accumulation_steps=1,
            task_type=model_config.task_type,
        )
    else:
        apply_full_param_freeze(model, train_config)
    model.set_processor(
        InputProcessor,
        padding_free=template_config.padding_free,
        padding_side=template_config.padding_side,
        cp_partition_mode=distributed_config.cp_partition_mode,
        **processor_kwargs(template),
    )
    model.set_template(template)
    configure_rl_loss(model, rl_config)
    # The optimizer reads the tuner's param groups and sizes its LR scheduler from the step budget;
    # the policy metric appends onto the optimizer's metric list (twinkle add_metric contract).
    configure_optimizer(
        model,
        train_config,
        num_training_steps=max_steps,
        distributed_config=distributed_config,
        tuner_config=tuner_config,
        plugin_config=None,
    )
    # The metric's temperature knob must be the SAMPLING temperature (1.0 unless the user set one):
    # it recovers the T=1 report scale from the sampler-temperature logps the forwards produce.
    sampling_temperature = 1.0
    if generation_config is not None and generation_config.temperature is not None:
        sampling_temperature = float(generation_config.temperature)
    configure_rl_metrics(model, rl_config, sampling_temperature=sampling_temperature)

    colocate = rollout_config.sampler_mode == "colocate"
    sampler = build_rl_sampler(
        model_config=model_config,
        rollout_config=rollout_config,
        template=template,
        tuner_config=tuner_config,
        resource_plan=resource_plan,
        colocate=colocate,
        enable_sampling_replay=rl_config.enable_sampling_replay,
    )
    engine = build_checkpoint_manager(model, sampler, rollout_config, remote=True)
    adapter = None
    if resource_plan is not None and rollout_config.resource_mode == "adaptive":
        adapter = build_cluster_adapter(
            resource_plan, rollout_config, model=model, sampler=sampler, checkpoint_manager=engine)
    params = rollout_sampling_params(rl_config.max_completion_length, generation_config)
    # One rollout for every generation (single-turn is the one-turn degenerate), on its CPU group.
    rollout = build_rollout(
        sampler, template, rollout_config, sampling_params=params, remote=True,
        inflight=inflight_rows(rollout_config, batch_size, rl_config.num_generations))

    # create the reference here (a KL anchor): a bare frozen model on its own group, or the policy
    # itself with its adapter off -- the pipeline derives disable_lora from that identity.
    reference = None
    if rl_config.beta and rl_config.calculate_KL is not False:
        ref_spec = (
            (rl_config.reference_parallel_spec or distributed_config.parallel_spec)
            if rl_config.sync_ref_model
            else rl_config.ref_parallel_spec
        )
        reference = build_ref_model(
            name=REF_GROUP,
            model_id=rl_config.ref_model,
            model_config=model_config,
            template=template,
            distributed_config=distributed_config,
            device_mesh=auxiliary_mesh(REF_GROUP, ref_spec, auxiliary_meshes),
            parallel_spec=ref_spec,
            adapters=tuple(rl_config.ref_adapters or ()),
            policy_model=model,
            disable_adapter=(tuner_config is not None and not rl_config.ref_adapters and not rl_config.sync_ref_model),
            model_type=rl_config.ref_model_type,
            model_revision=rl_config.ref_model_revision,
        )
    # create the teacher here (the RLSD/SDAR reweighting channel), same bare-model shape
    teacher = build_ref_model(
        name=TEACHER_GROUP,
        model_id=single_teacher_id(rl_config.teacher_model, algo="RLSD/SDAR"),
        model_config=model_config,
        template=template,
        distributed_config=distributed_config,
        device_mesh=auxiliary_mesh(TEACHER_GROUP, rl_config.teacher_parallel_spec, auxiliary_meshes),
        parallel_spec=rl_config.teacher_parallel_spec,
        adapters=tuple(rl_config.teacher_adapters or ()),
        policy_model=model,
        disable_adapter=rl_config._teacher_use_disable_adapter,
        model_type=rl_config.teacher_model_type,
        model_revision=rl_config.teacher_model_revision,
        deepspeed=rl_config.teacher_deepspeed,
    )

    # create prm & orm here -- the shared channel resolver: one scorer worker per entry, each
    # reward MODEL on its own group; the driver runs them on every finished episode / segment.
    scorer_kwargs = {
        "reward_config": reward_config,
        "model_config": model_config,
        "template_config": template_config,
        "distributed_config": distributed_config,
        "rollout_config": rollout_config,
        "backend": rollout_config.rollout_sampler,
        "sampler": sampler,
        "worker_group": CONTROL_GROUP,
        "quantize_config": quantize_config,
    }
    orm_scorers, orm_names = resolve_reward_scorers("orm", **scorer_kwargs)
    prm_scorers, _ = resolve_reward_scorers("prm", **scorer_kwargs)

    # the signal components, each on the control group and only when its option is on: the
    # per-token lanes they read stay references in the driver.
    group = {"remote_group": CONTROL_GROUP}
    has_teacher = teacher is not None or teacher_prompt
    preserve = bool(rl_config.rollout_importance_sampling_mode or rl_config.log_rollout_offpolicy_metrics)
    kl_in_reward = bool(rl_config.kl_in_reward and rl_config.beta)
    teacher_mode = has_teacher and not prm_scorers and (
        rl_config.advantage_reweight == "rlsd" or rl_config.sdar_loss_coef <= 0)

    return Cluster(
        model=model,
        sampler=sampler,
        rollout=rollout,
        engine=engine,
        adapter=adapter,
        reference=reference,
        teacher=teacher,
        orm_scorers=orm_scorers,
        orm_names=orm_names,
        prm_scorers=prm_scorers,
        sampling_params=params,
        colocate=colocate,
        scores=ResponseScores(**group),
        kl=KLEstimator(**group) if kl_in_reward else None,
        process_reward=ProcessReward(**group) if prm_scorers else None,
        combine=SignalCombine(**group) if prm_scorers else None,
        teacher_reweight=TeacherReweight(**group) if teacher_mode else None,
        privileged=PrivilegedView(template=template, **group) if teacher_prompt else None,
        correction=RolloutCorrection(**group) if preserve else None,
        routing=RoutingReplay(**group) if rl_config.router_replay_mode == "R2" else None,
    )


def grpo_on_group(rl_config: "RLConfig", reward_config: "RewardConfig", sampling_params: Dict[str, Any],
                  num_orms: int) -> Optional[Callable[[Any, List[Any], int], List[Any]]]:
    """GRPO's group verdict for the driver: dynamic_sample regenerates, remax adds a greedy baseline.

    A zero-variance group (its advantage is degenerate) is replaced by ``num_generations`` fresh
    trajectories of the same row, up to ``max_resample_times`` (the driver's regeneration cap);
    remax returns one greedy baseline of the row, run once the group is final. ``None`` when
    neither is on.
    """
    dynamic = bool(rl_config.dynamic_sample)
    remax = rl_config.advantage_estimator == "remax"
    if not dynamic and not remax:
        return None
    from twinkle.data_format import attach_user_data

    from swift.dev.utils.scoring import build_reward_weights

    weights = build_reward_weights(reward_config.orm_weights, num_orms) if num_orms else None
    num_generations = rl_config.num_generations
    greedy = greedy_params(sampling_params)

    def on_group(row: Any, samples: List[Any], attempt: int) -> List[Any]:
        trajectories: List[Any] = []
        if dynamic and group_variance([sample.orm_reward for sample in samples], weights) <= 0.0:
            trajectories = [copy.deepcopy(row) for _ in range(num_generations)]
        if remax:
            trajectories.append(attach_user_data(copy.deepcopy(row), role="baseline", sampling_params=greedy))
        return trajectories

    return on_group


def build_driver(*, cluster: Cluster, rl_config: "RLConfig", rollout_config: "RolloutConfig",
                 reward_config: "RewardConfig", prompt_dataloader: Any, batch_size: int,
                 num_train_epochs: float, max_steps: Optional[int]) -> Any:
    """Assemble twinkle's StreamingDriver over the cluster; GRPO's own face is the group verdict."""
    from swift.dev.builders.streaming import build_streaming_driver

    return build_streaming_driver(
        dataloader=prompt_dataloader,
        rollout=cluster.rollout,
        rollout_config=rollout_config,
        batch_size=batch_size,
        num_generations=rl_config.num_generations,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        checkpoint_manager=cluster.engine,
        orms=cluster.orm_scorers,
        prms=cluster.prm_scorers,
        on_group=grpo_on_group(rl_config, reward_config, cluster.sampling_params, len(cluster.orm_scorers)),
        max_group_regenerations=(rl_config.max_resample_times if rl_config.dynamic_sample else 0),
        cluster_adapter=cluster.adapter,
        remote=True,
    )
