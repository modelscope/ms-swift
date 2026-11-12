# Copyright (c) ModelScope Contributors. All rights reserved.
# PPO's assembly: the value-critic cluster (policy + critic + reference + reward channel).
from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple

from swift.dev.recipe.utils import REF_GROUP, auxiliary_mesh, build_rl_sampler, resolve_reward_scorers
from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
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


def ppo_group_requests(
    rl_config: "RLConfig",
    reward_config: "RewardConfig",
    distributed_config: "DistributedConfig",
    tuner_config: Optional["TunerConfig"],
) -> List[Tuple[str, Optional[str]]]:
    """The disjoint DeviceGroups PPO's full-parameter frozen models need -> ``[(name, spec)]``.

    Mirrors the builds in :func:`initialize_ppo_cluster`: PPO ALWAYS anchors the KL -- a
    full-parameter run keeps a frozen ref copy (under sync_ref_model it inherits the policy's
    sharding); only a LoRA run reuses the policy actor and plans NO group. Each orm MODEL gets
    its ``reward_<i>`` group; the critic shares the policy's 'model' group, so it plans none.
    """
    from swift.dev.rewards.scorer import split_channel

    requests: List[Tuple[str, Optional[str]]] = []
    if not (tuner_config is not None and not rl_config.ref_adapters and not rl_config.sync_ref_model):
        spec = (
            (rl_config.reference_parallel_spec or distributed_config.parallel_spec)
            if rl_config.sync_ref_model
            else rl_config.ref_parallel_spec
        )
        requests.append((REF_GROUP, spec))
    for index in range(len(split_channel(list(reward_config.orm))[1])):
        requests.append((f"reward_{index}", reward_config.orm_parallel_spec))
    return requests


class Cluster(NamedTuple):
    """Every actor PPO's visible loop drives (a plain handle bundle, not a layer).

    The signal components (``scores`` ... ``reduce``) read the per-token lanes where they live.
    """

    model: Any
    value_model: Any
    sampler: Any
    rollout: Any
    engine: Any
    adapter: Any
    reference: Any
    orm_scorers: List[Any]
    orm_names: List[str]
    sampling_params: Dict[str, Any]
    colocate: bool
    scores: Any
    kl: Any
    reward_shape: Any
    gae: Any
    reduce: Any


def initialize_ppo_cluster(
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
) -> Cluster:
    """Build PPO's actors explicitly: the policy, the value critic, the frozen reference, the rewards.

    The critic is a trainable seq_cls ``num_labels=1`` model on the policy's OWN 'model' group (a
    distinct ``instance_id``) -- policy and critic MUST slice_dp at the same data_world_size,
    which is also why PPO plans no SP mesh. ``max_steps`` is the resolved step budget, so both
    optimizers' LR schedulers are fully configured here; ``batch_size`` is the rows of one driver
    batch (it bounds the episodes in flight).
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
    from swift.dev.loss import (
        configure_ppo_value_loss,
        configure_ppo_value_metric,
        configure_rl_loss,
        configure_rl_metrics,
    )
    from swift.dev.processor import InputProcessor, processor_kwargs
    from twinkle.advantage import GAEAdvantage
    from twinkle.reward import RewardShape
    from twinkle.signals import KLEstimator, Reduce, ResponseScores

    sp = template_config.sequence_parallel_size
    sp_mesh = build_dp_mesh(distributed_config, sp) if sp > 1 else None
    resume_dir = checkpoint_config.resume_from_checkpoint if checkpoint_config is not None else None
    policy_config = model_config
    if resume_dir and tuner_config is None:
        # A full-parameter resume loads weights from the ckpt dir instead of the original model id;
        # a tuned policy keeps the original base and reloads its adapter via resume (run_ppo).
        policy_config = copy.deepcopy(model_config)
        policy_config.model = resume_dir
    model = build_model(
        policy_config,
        distributed_config,
        train_config,
        tuner_config,
        device_mesh=sp_mesh,
        quantize_config=quantize_config,
        megatron_config=megatron_config,
        moe_config=moe_config,
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
    # the policy's loss is the shared clipped surrogate ('ppo' maps to it in configure_rl_loss)
    configure_rl_loss(model, rl_config)
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

    # The critic: a trainable seq_cls num_labels=1 value model, initialized from the first ORM
    # reward MODEL (TRL's choice) or the policy base; full-param, so no tuner_config reaches it.
    value_config = copy.deepcopy(model_config)
    value_config.task_type = "seq_cls"
    value_config.num_labels = 1
    from swift.dev.rewards.scorer import split_channel

    orm_models = split_channel(list(reward_config.orm))[1] if reward_config is not None else []
    value_config.model = orm_models[0] if orm_models else model_config.model
    value_model = build_model(
        value_config,
        distributed_config,
        train_config,
        megatron_config=megatron_config,
        moe_config=moe_config,
        instance_id="value",
        torch_dtype="float32",
    )
    value_model.set_processor(
        InputProcessor,
        padding_free=template_config.padding_free,
        padding_side=template_config.padding_side,
        cp_partition_mode=distributed_config.cp_partition_mode,
        **processor_kwargs(template),
    )
    value_model.set_template(template)
    configure_ppo_value_loss(value_model, rl_config)
    configure_optimizer(
        value_model,
        train_config,
        num_training_steps=max_steps,
        distributed_config=distributed_config,
        tuner_config=None,
        plugin_config=None,
    )
    configure_ppo_value_metric(value_model, rl_config)

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
        # Only the flex ranks: the trainer's ranks also hold the critic, which lending would leave
        # resident next to the borrowed replicas.
        adapter = build_cluster_adapter(
            resource_plan, rollout_config, model=model, sampler=sampler, checkpoint_manager=engine, lend=False)
    params = rollout_sampling_params(rl_config.max_completion_length, generation_config)
    # PPO rolls out single-turn, on the rollout CPU group.
    rollout = build_rollout(
        sampler, template, rollout_config, sampling_params=params, remote=True,
        inflight=inflight_rows(rollout_config, batch_size, max(1, rl_config.num_generations)))

    # PPO ALWAYS anchors the KL: the full-parameter default is a frozen policy-init copy on the ref
    # group; a LoRA run reuses the policy actor itself with its adapter off (identity-derived).
    reference = build_ref_model(
        name=REF_GROUP,
        model_id=(rl_config.ref_model or model_config.model),
        model_config=model_config,
        template=template,
        distributed_config=distributed_config,
        device_mesh=auxiliary_mesh(
            REF_GROUP,
            (
                (rl_config.reference_parallel_spec or distributed_config.parallel_spec)
                if rl_config.sync_ref_model
                else rl_config.ref_parallel_spec
            ),
            auxiliary_meshes,
        ),
        parallel_spec=(
            (rl_config.reference_parallel_spec or distributed_config.parallel_spec)
            if rl_config.sync_ref_model
            else rl_config.ref_parallel_spec
        ),
        adapters=tuple(rl_config.ref_adapters or ()),
        policy_model=model,
        disable_adapter=(tuner_config is not None and not rl_config.ref_adapters and not rl_config.sync_ref_model),
        model_type=rl_config.ref_model_type,
        model_revision=rl_config.ref_model_revision,
    )

    orm_scorers, orm_names = resolve_reward_scorers(
        "orm",
        reward_config=reward_config,
        model_config=model_config,
        template_config=template_config,
        distributed_config=distributed_config,
        rollout_config=rollout_config,
        backend=rollout_config.rollout_sampler,
        sampler=sampler,
        worker_group=CONTROL_GROUP,
        quantize_config=quantize_config,
    )
    group = {"remote_group": CONTROL_GROUP}
    return Cluster(
        model=model,
        value_model=value_model,
        sampler=sampler,
        rollout=rollout,
        engine=engine,
        adapter=adapter,
        reference=reference,
        orm_scorers=orm_scorers,
        orm_names=orm_names,
        sampling_params=params,
        colocate=colocate,
        scores=ResponseScores(**group),
        kl=KLEstimator(**group),
        reward_shape=RewardShape(**group),
        gae=GAEAdvantage(gamma=rl_config.gamma, gae_lambda=rl_config.lam, normalize=rl_config.whiten_rewards, **group),
        reduce=Reduce(**group),
    )


def build_ppo_driver(*, cluster: Cluster, rl_config: 'RLConfig', rollout_config: 'RolloutConfig',
                     prompt_dataloader: Any, batch_size: int, num_train_epochs: float,
                     max_steps: Optional[int]) -> Any:
    """Assemble twinkle's StreamingDriver for PPO: ORM-scored episodes, no group verdict.

    PPO has no group contract (per-token GAE scores each trajectory on its own); every async_mode
    rides this one stream.
    """
    from swift.dev.builders.streaming import build_streaming_driver

    return build_streaming_driver(
        dataloader=prompt_dataloader,
        rollout=cluster.rollout,
        rollout_config=rollout_config,
        batch_size=batch_size,
        num_generations=max(1, rl_config.num_generations),
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        checkpoint_manager=cluster.engine,
        orms=cluster.orm_scorers,
        cluster_adapter=cluster.adapter,
        remote=True,
    )
