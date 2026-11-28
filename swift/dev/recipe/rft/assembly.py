# Copyright (c) ModelScope Contributors. All rights reserved.
# ruff: noqa: C901
# RFT's device and cluster assembly: the frozen-reward groups and the explicit actor build.
from __future__ import annotations
from copy import copy
from typing import TYPE_CHECKING, List, Optional, Tuple

from swift.dev.config.adapter_config import has_adapter
from swift.dev.recipe.configs import RecipeConfigs
from swift.dev.recipe.preparation import prepare_training_model
from swift.dev.recipe.resources import RFTResources
from swift.dev.recipe.utils import build_rl_sampler, effective_colocate, resolve_reward_scorers
from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from twinkle.dataloader import DataLoader
    from twinkle_agentic.async_rl import StreamingDriver
    from twinkle_agentic.async_rl.cluster_adapter import RLResourcePlan

    from swift.dev.config import (
        RewardConfig,
    )
    from swift.dev.template import Template

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


def initialize_rft_cluster(
    configs: RecipeConfigs,
    *,
    template: Template,
    max_steps: int,
    resource_plan: Optional[RLResourcePlan] = None,
    inflight: Optional[int] = None,
) -> RFTResources:
    """Assemble the supervised policy and reward scorers for fixed-round rejection sampling."""
    model_config = configs.model
    template_config = configs.template
    train_config = configs.train
    distributed_config = configs.distributed
    checkpoint_config = configs.checkpoint
    rollout_config = configs.rollout
    rl_config = configs.rl
    reward_config = configs.reward
    tuner_config = configs.tuner
    quantize_config = configs.quantize
    generation_config = configs.generation
    assert rl_config is not None and rollout_config is not None and reward_config is not None
    from swift.dev.builder import build_dp_mesh
    from swift.dev.builder.streaming import (
        CONTROL_GROUP,
        build_rollout,
        rollout_sampling_params,
    )
    from swift.dev.loss import configure_loss

    if resource_plan is not None and rollout_config.resource_mode == "adaptive":
        raise ValueError(
            "RFT does not support resource_mode='adaptive' (no runtime ClusterAdapter); "
            "use 'manual' or 'estimate' for a static plan."
        )
    sp = template_config.sequence_parallel_size
    # mode='remote' runs ONE client process (the sft assembly's remote shape for the model;
    # the sampler / weight sync / rollout take their remote object sources below).
    remote_run = distributed_config.mode == "remote"
    # build_dp_mesh is the only seam that reads and validates parallel_spec on the local path, so
    # it must not be gated behind SP>1; ray keeps the gate (see its docstring) so no ray chain moves.
    sp_mesh = (
        None if remote_run or (distributed_config.mode == "ray" and sp <= 1) else build_dp_mesh(distributed_config, sp)
    )

    def sft_loss(model):
        if remote_run:
            from swift.dev.builder.remote import configure_remote_loss

            configure_remote_loss(model, train_config, task_type=model_config.task_type, model_config=model_config)
        else:
            configure_loss(model, loss_type="cross_entropy", reduction="sum")

    build_config = model_config
    resume = checkpoint_config.resume_from_checkpoint if checkpoint_config is not None else None
    if resume and not has_adapter(tuner_config) and distributed_config.mode != "remote":
        build_config = copy(model_config)
        build_config.model = resume
    model = prepare_training_model(
        configs,
        template,
        num_training_steps=max_steps,
        device_mesh=sp_mesh,
        configure_loss=sft_loss,
        build_config=build_config,
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
    rollout = build_rollout(
        sampler, template, rollout_config, sampling_params=params, remote=not remote_run, inflight=inflight
    )
    from swift.dev.recipe.online import build_policy_stream

    engine, _ = build_policy_stream(model, sampler, rollout_config, distributed_config, resource_plan)
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
            "reward in --orm, or use the policy recipe (rl_type=grpo/ppo) for per-token process rewards."
        )
    return RFTResources(
        model=model,
        sampler=sampler,
        rollout=rollout,
        engine=engine,
        sampling_params=params,
        colocate=colocate,
        orm_scorers=orm_scorers,
        orm_scorer_names=orm_scorer_names,
    )


def build_rft_driver(
    configs: RecipeConfigs, *, cluster: RFTResources, prompt_dataloader: DataLoader, num_samples: int, window: int
) -> StreamingDriver:
    """One round's StreamingDriver: every prompt once, one group of ``num_samples`` per batch.

    No checkpoint manager: the policy is frozen for the round (synced once before it), so nothing is
    stale and every group is handed out, the last one included; about ``window`` groups run ahead.
    The orm channel scores every finished episode; ``orm_scorer_names`` name the scores the
    batches' summaries carry.
    """
    rollout_config = configs.rollout
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
