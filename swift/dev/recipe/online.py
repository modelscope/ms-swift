# Copyright (c) ModelScope Contributors. All rights reserved.
"""Online training cadence over the existing driver and weight channel."""

from __future__ import annotations
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from swift.dev.recipe.utils import RolloutStepState, StepState, close_driver_preserving_error

if TYPE_CHECKING:
    from ray import ObjectRef
    from twinkle.checkpoint_engine import CheckpointEngineManager
    from twinkle.data_format import InputFeature, Trajectory
    from twinkle.sampler import Sampler
    from twinkle_agentic.async_rl import ClusterAdapter, StreamingDriver
    from twinkle_agentic.async_rl.cluster_adapter import RLResourcePlan
    from twinkle_client.rollout import HTTPSampler

    from swift.dev.builder.remote import RemoteWeightSync
    from swift.dev.config import DistributedConfig, RolloutConfig
    from swift.dev.recipe.preparation import TrainingModel

    WeightSync = CheckpointEngineManager | RemoteWeightSync
    RolloutSampler = Sampler | HTTPSampler
    TrainingRows = list[Trajectory | InputFeature | ObjectRef]


def build_policy_stream(
    model: TrainingModel,
    sampler: RolloutSampler,
    rollout_config: RolloutConfig,
    distributed_config: DistributedConfig,
    resource_plan: Optional[RLResourcePlan],
    *,
    lend: bool = True,
) -> tuple[WeightSync, Optional[ClusterAdapter]]:
    """Build the weight channel and optional adaptive adapter without changing rollout ownership."""
    from swift.dev.builder.streaming import build_checkpoint_manager, build_cluster_adapter

    if distributed_config.mode == "remote":
        from swift.dev.builder.remote import RemoteWeightSync

        engine = RemoteWeightSync(model, sampler)
    else:
        engine = build_checkpoint_manager(model, sampler, rollout_config, remote=True, resource_plan=resource_plan)
    adapter = None
    if resource_plan is not None and rollout_config.resource_mode == "adaptive":
        adapter = build_cluster_adapter(
            resource_plan,
            rollout_config,
            model=model,
            sampler=sampler,
            checkpoint_manager=engine,
            **({"lend": False} if not lend else {}),
        )
    return engine, adapter


def publish_policy(
    engine: WeightSync,
    rollout_config: RolloutConfig,
    *,
    colocate: bool,
    companion_models: Sequence[TrainingModel] = (),
    on_lend: Optional[Callable[[], None]] = None,
) -> None:
    """Offload companion models before publishing only the policy's weights."""
    from swift.dev.builder.streaming import offload_kwargs

    if colocate and companion_models:
        for model in companion_models:
            model.offload_to_cpu(**offload_kwargs(rollout_config))
        if on_lend is not None:
            on_lend()
    engine.sync_weights()
    if colocate and on_lend is not None:
        on_lend()


def reclaim_training_devices(
    model: TrainingModel,
    sampler: RolloutSampler,
    rollout_config: RolloutConfig,
    *,
    companion_models: Sequence[TrainingModel] = (),
) -> None:
    """Sleep the sampler before restoring the policy and its trainable companions."""
    from swift.dev.recipe.offload import sleep_for_rollout

    sleep_for_rollout(sampler, rollout_config.sleep_level)
    model.reload_to_gpu()
    for companion in companion_models:
        companion.reload_to_gpu()


def fit_rollout(  # noqa: C901
    state: StepState,
    driver: StreamingDriver,
    sampler: RolloutSampler,
    engine: WeightSync,
    rollout_config: RolloutConfig,
    *,
    colocate: bool,
    train_batch: Callable[[TrainingRows, dict], None],
    companion_models: Sequence[TrainingModel] = (),
    before_rollout: Optional[Callable[[], bool]] = None,
    sync_every: Optional[int] = None,
) -> None:
    """Train delivered batches and publish by delivery count, not optimizer-step count."""
    from twinkle.perf import span as perf_span

    lent = False
    rollout_state = state if isinstance(state, RolloutStepState) else None

    def mark_lent() -> None:
        nonlocal lent
        lent = True

    def sync(*, measured: bool = False) -> None:
        span = (
            perf_span("update_weights")
            if measured and rollout_state is not None and not companion_models
            else nullcontext()
        )
        with span:
            publish_policy(
                engine,
                rollout_config,
                colocate=colocate,
                companion_models=companion_models,
                on_lend=mark_lent,
            )

    def reclaim() -> None:
        nonlocal lent
        reclaim_training_devices(state.model, sampler, rollout_config, companion_models=companion_models)
        lent = False

    # GRPO publishes before its loop guard; PPO protects the companion offload, Distill the source draw.
    initial_outside = not companion_models and before_rollout is None
    if initial_outside:
        sync()
    batches = 0
    try:
        if before_rollout is not None and not before_rollout():
            return
        if not initial_outside:
            sync()
        if rollout_state is not None:
            rollout_state.begin_rollout_iteration()
        for rows, facts in driver:
            if colocate:
                reclaim()
            if rollout_state is not None:
                rollout_state.observe_rollout_batch(facts)
            train_batch(rows, facts)
            batches += 1
            if before_rollout is not None and not before_rollout():
                break
            finished = state.reached_max() or state.should_stop()
            interval = rollout_config.parameter_sync_step if sync_every is None else sync_every
            publish = not finished and batches % interval == 0
            if publish:
                sync(measured=True)
            if rollout_state is not None:
                rollout_state.fold_iteration_metrics(publications=int(publish), terminal=finished)
            if finished:
                break
            if rollout_state is not None:
                rollout_state.begin_rollout_iteration()
    finally:
        stats = close_driver_preserving_error(driver)
        if rollout_state is not None:
            state.stream_stats = stats
        if lent:
            reclaim()
