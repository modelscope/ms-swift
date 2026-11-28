# Copyright (c) ModelScope Contributors. All rights reserved.
# ruff: noqa: C901
# RFT's device and cluster assembly: the frozen-reward groups and the explicit actor build.
from __future__ import annotations
from copy import copy
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple

from swift.dev.config.adapter_config import has_adapter
from swift.dev.recipe.utils import build_rl_sampler, effective_colocate, resolve_reward_scorers
from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
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


def rft_group_requests(reward_config: "RewardConfig") -> List[Tuple[str, Optional[str]]]:
    """The disjoint DeviceGroups RFT's frozen reward models need -> ``[(name, spec)]``.

    RFT has no reference and no teacher (it is cross-entropy SFT on filtered completions), so each
    MODEL in the orm channel is its only frozen auxiliary -- one ``reward_<i>`` group per model,
    exactly the names :func:`resolve_reward_scorers` builds below (both sides split the channel).
    """
    from swift.dev.reward.scorer import split_channel

    return [
        (f"reward_{index}", reward_config.orm_parallel_spec)
        for index in range(len(split_channel(list(reward_config.orm))[1]))
    ]


class RFTCluster(NamedTuple):
    """Every actor RFT's visible loop drives (a plain handle bundle, not a layer)."""

    model: Any
    sampler: Any
    rollout: Any
    engine: Any
    sampling_params: Dict[str, Any]
    colocate: bool
    orm_scorers: List[Any]
    #: One metric-column name per orm scorer, in the scorer order (rules then models).
    orm_scorer_names: List[str]


def initialize_rft_cluster(
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
    quantize_config: Optional["QuantizeConfig"] = None,
    megatron_config: Optional["MegatronConfig"] = None,
    moe_config: Optional["MoEConfig"] = None,
    generation_config: Optional[Any] = None,
    resource_plan: Any = None,
    inflight: Optional[int] = None,
) -> RFTCluster:
    """Build every actor RFT trains with: the policy, the sampler, the orm reward channel.

    GRPO's :func:`~swift.dev.recipe.grpo.assembly.initialize_cluster` mirrored minus the
    auxiliaries RFT does not have (no reference, no teacher, no PRM, no elastic adapter) and with
    the SFT cross-entropy loss instead of an RL one; ``max_steps`` is the resolved step budget
    (the LR-scheduler horizon), derived by the caller BEFORE this stage. ``inflight`` is the
    episodes a round runs at once (see :func:`~swift.dev.builder.streaming.inflight_rows`).
    """
    from swift.dev.builder import apply_full_param_freeze, build_dp_mesh, build_model
    from swift.dev.builder.adapter import apply_tuner
    from swift.dev.builder.optimizer import configure_optimizer
    from swift.dev.builder.streaming import (
        CONTROL_GROUP,
        build_checkpoint_manager,
        build_rollout,
        rollout_sampling_params,
    )
    from swift.dev.loss import configure_loss
    from swift.dev.processor import InputProcessor, input_processor_kwargs

    if resource_plan is not None and rollout_config.resource_mode == "adaptive":
        raise ValueError(
            "RFT does not support resource_mode='adaptive' (no runtime ClusterAdapter); "
            "use 'manual' or 'estimate' for a static plan."
        )
    sp = template_config.sequence_parallel_size
    # mode='remote' runs ONE client process (the sft assembly's remote shape for the model;
    # the sampler / weight sync / rollout take their remote object sources below).
    remote_run = distributed_config.mode == 'remote'
    # build_dp_mesh is the only seam that reads and validates parallel_spec on the local path, so
    # it must not be gated behind SP>1; ray keeps the gate (see its docstring) so no ray chain moves.
    sp_mesh = (None if remote_run or (distributed_config.mode == 'ray' and sp <= 1)
               else build_dp_mesh(distributed_config, sp))
    resume_dir = checkpoint_config.resume_from_checkpoint if checkpoint_config is not None else None
    policy_config = model_config
    if resume_dir and not has_adapter(tuner_config) and not remote_run:
        # A full-parameter resume loads weights from the ckpt dir instead of the original model id;
        # a tuned policy keeps the original base and reloads its adapter via resume (run_grpo).
        policy_config = copy(model_config)
        policy_config.model = resume_dir
    # The model / tuner / freeze ride the ONE builders entry in every mode (mode='remote'
    # self-dispatches inside build_model / apply_tuner / apply_full_param_freeze to the client
    # handle); what genuinely differs by mode is the processor / loss source below.
    model = build_model(
        policy_config,
        distributed_config,
        train_config,
        tuner_config,
        device_mesh=sp_mesh,
        quantize_config=quantize_config,
        megatron_config=megatron_config,
        moe_config=moe_config,
    )
    if has_adapter(tuner_config):
        apply_tuner(
            model,
            tuner_config,
            gradient_accumulation_steps=1,
            task_type=model_config.task_type,
            model_config=model_config,
        )
    else:
        apply_full_param_freeze(model, train_config)
    if remote_run:
        from swift.dev.builder.remote import (
            configure_remote_loss,
            configure_remote_processor,
        )

        configure_remote_processor(model, template_config, model_config, distributed_config)
        # RFT trains the kept completions with plain cross-entropy SFT -- the same causal_lm
        # cross-entropy the remote supervised path sets by name (reduction='sum', no DFT).
        configure_remote_loss(model, train_config, task_type=model_config.task_type,
                              model_config=model_config)
    else:
        model.set_processor(
            InputProcessor,
            **input_processor_kwargs(template, template_config, distributed_config, model_config),
        )
        model.set_template(template)
        # RFT trains the kept completions with plain cross-entropy SFT -- NOT an RL loss.
        configure_loss(model, loss_type="cross_entropy", reduction="sum")
    configure_optimizer(
        model,
        train_config,
        num_training_steps=max_steps,
        distributed_config=distributed_config,
        tuner_config=tuner_config,
        plugin_config=None,
    )
    colocate = effective_colocate(rollout_config, resource_plan)
    sampler = build_rl_sampler(
        model_config=model_config,
        rollout_config=rollout_config,
        template=template,
        tuner_config=tuner_config,
        resource_plan=resource_plan,
        colocate=colocate,
        enable_sampling_replay=rl_config.enable_sampling_replay,
        distributed_config=distributed_config,
        template_config=template_config,
    )
    params = rollout_sampling_params(rl_config.max_completion_length, generation_config)
    # One rollout for every generation (single-turn is the one-turn degenerate); under ray it is
    # an actor on the CPU rollout group, remotely a plain in-process object over the client
    # sampler handle.
    rollout = build_rollout(sampler, template, rollout_config, sampling_params=params,
                            remote=not remote_run, inflight=inflight)
    if remote_run:
        from swift.dev.builder.remote import RemoteWeightSync

        engine = RemoteWeightSync(model, sampler)
    else:
        engine = build_checkpoint_manager(model, sampler, rollout_config, remote=True, resource_plan=resource_plan)
    orm_scorers, orm_scorer_names = resolve_reward_scorers(
        "orm",
        reward_config=reward_config,
        model_config=model_config,
        template_config=template_config,
        distributed_config=distributed_config,
        rollout_config=rollout_config,
        backend=rollout_config.rollout_sampler,
        sampler=sampler,
        worker_group=None if remote_run else CONTROL_GROUP,
        quantize_config=quantize_config,
    )
    if not orm_scorers:
        raise ValueError(
            "RFT needs at least one --orm scorer: rejection sampling selects completions BY reward, so "
            "an empty outcome channel would silently train on random samples. Register a rule or place "
            "a scalar reward model in --orm."
        )
    if list(reward_config.prm):
        raise ValueError(
            "RFT does not consume a --prm channel (it trains on whole completions); put the "
            "reward in --orm, or use run_grpo for per-token process rewards."
        )
    return RFTCluster(
        model=model,
        sampler=sampler,
        rollout=rollout,
        engine=engine,
        sampling_params=params,
        colocate=colocate,
        orm_scorers=orm_scorers,
        orm_scorer_names=orm_scorer_names,
    )


def build_rft_driver(*, cluster: RFTCluster, rollout_config: "RolloutConfig", prompt_dataloader: Any,
                     num_samples: int, window: int) -> Any:
    """One round's StreamingDriver: every prompt once, one group of ``num_samples`` per batch.

    No checkpoint manager: the policy is frozen for the round (synced once before it), so nothing is
    stale and every group is handed out, the last one included; about ``window`` groups run ahead.
    The orm channel scores every finished episode; ``orm_scorer_names`` name the scores the
    batches' summaries carry.
    """
    import math

    from swift.dev.builder.streaming import build_streaming_driver

    return build_streaming_driver(
        dataloader=prompt_dataloader,
        rollout=cluster.rollout,
        rollout_config=rollout_config,
        batch_size=num_samples,
        num_generations=num_samples,
        num_train_epochs=1,
        orms=cluster.orm_scorers,
        orm_names=cluster.orm_scorer_names,
        max_staleness=max(0, math.ceil(window / rollout_config.parameter_sync_step) - 1),
        remote=True,
    )
