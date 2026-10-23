# Copyright (c) ModelScope Contributors. All rights reserved.
# ruff: noqa: C901
# RFT's device and cluster assembly: the frozen-reward groups and the explicit actor build.
from __future__ import annotations
from copy import copy
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple

from swift.dev.recipe.utils import build_rl_sampler, resolve_reward_scorers
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
    from swift.dev.rewards.scorer import split_channel

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
    worker: Any
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
    output_dir: str = "output",
) -> RFTCluster:
    """Build every actor RFT trains with: the policy, the sampler, the orm reward channel.

    GRPO's :func:`~swift.dev.recipe.grpo.assembly.initialize_cluster` mirrored minus the
    auxiliaries RFT does not have (no reference, no teacher, no PRM, no elastic adapter) and with
    the SFT cross-entropy loss instead of an RL one; ``max_steps`` is the resolved step budget
    (the LR-scheduler horizon), derived by the caller BEFORE this stage.
    """
    from twinkle.checkpoint_engine import CheckpointEngineManager
    from twinkle.data_format import SamplingParams
    from twinkle_agentic.async_rl import REF_ONLY_REMOTE_GROUP
    from twinkle_agentic.rollout import MultiTurnRollout

    from swift.dev.builders import apply_full_param_freeze, build_dp_mesh, build_model
    from swift.dev.builders.adapter import apply_tuner
    from swift.dev.builders.optimizer import configure_optimizer
    from swift.dev.builders.sandbox import tools_factory
    from swift.dev.loss import configure_loss
    from swift.dev.processor import InputProcessor, processor_kwargs
    from swift.dev.rewards.scorer import ApiJudgeReward, GenerativeJudgeReward, RewardModel
    from swift.dev.rewards.worker import RewardWorker
    from swift.dev.utils.plugin import PluginRegistry

    if resource_plan is not None and rollout_config.resource_mode == "adaptive":
        raise ValueError(
            "RFT does not support resource_mode='adaptive' (no runtime ClusterAdapter); "
            "use 'manual' or 'estimate' for a static plan."
        )
    sp = template_config.sequence_parallel_size
    sp_mesh = build_dp_mesh(distributed_config, sp) if sp > 1 else None
    resume_dir = checkpoint_config.resume_from_checkpoint if checkpoint_config is not None else None
    policy_config = model_config
    if resume_dir and tuner_config is None:
        # A full-parameter resume loads weights from the ckpt dir instead of the original model id;
        # a tuned policy keeps the original base and reloads its adapter via resume (run_grpo).
        policy_config = copy(model_config)
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
    )
    if tuner_config is not None:
        apply_tuner(
            model,
            tuner_config,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
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
    params: Dict[str, Any] = {"max_tokens": rl_config.max_completion_length}
    if generation_config is not None:
        for key in ("temperature", "top_p", "top_k"):
            value = getattr(generation_config, key, None)
            if value is not None:
                params[key] = value
    # One engine for every generation entry (single-turn is the one-turn degenerate): disaggregated
    # places it on the CPU payload group; colocate keeps it local for the visible loop's generate.
    rollout = MultiTurnRollout(
        sampler,
        template,
        sampling_params=SamplingParams(**params),
        allow_partial_rollout=rollout_config.allow_partial_rollout,
        max_turns=(rollout_config.max_turns or 1),
        max_trajectory_tokens=rollout_config.max_trajectory_tokens,
        tools_factory=tools_factory,
        tools_config=rollout_config,
        remote_group=(None if colocate else REF_ONLY_REMOTE_GROUP),
    )
    engine = CheckpointEngineManager(
        model=model, sampler=sampler, mode=("colocate" if colocate else "standalone"),
        sync_backend=rollout_config.weight_sync_backend,
    )
    orm_scorers = resolve_reward_scorers(
        "orm",
        reward_config=reward_config,
        model_config=model_config,
        template_config=template_config,
        distributed_config=distributed_config,
        rollout_config=rollout_config,
        backend=rollout_config.rollout_sampler,
        sampler=sampler,
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
    for scorer in orm_scorers:
        if isinstance(scorer, (ApiJudgeReward, GenerativeJudgeReward)):
            raise ValueError(
                f"--orm judge {type(scorer).__name__} is not placeable in RFT: rejection sampling needs a "
                "SCALAR reward column to filter on. Use a scalar seq_cls/reranker reward model or a rule."
            )
    orm_scorer_names = [
        scorer.display_name if isinstance(scorer, RewardModel) else PluginRegistry.display_name(scorer)
        for scorer in orm_scorers
    ]
    worker = RewardWorker(orm_scorers=orm_scorers, template=template)
    return RFTCluster(
        model=model,
        sampler=sampler,
        rollout=rollout,
        engine=engine,
        worker=worker,
        sampling_params=params,
        colocate=colocate,
        orm_scorers=orm_scorers,
        orm_scorer_names=orm_scorer_names,
    )
