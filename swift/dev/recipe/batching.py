# Copyright (c) ModelScope Contributors. All rights reserved.
"""Controller-side DP balancing for post-rollout training batches."""

from __future__ import annotations
import math
from typing import Any


def streaming_batch_geometry(
    *, need_rows: int, num_generations: int, parameter_sync_step: int, max_staleness: int
) -> tuple[int, int]:
    """Size admission windows in prompt groups and prefetch capacity in trajectories.

    One optimizer update consumes ``need_rows`` trajectories, not one prompt group.
    Round a publication cycle up to complete optimizer batches and generation groups,
    then reserve a cycle for the current version and each permitted overlapping version.
    The iterator driver waits for training's commit, so extra in-flight capacity must
    exist before it hands off a batch for generation to continue during training.
    """
    pull_rows = math.lcm(need_rows, num_generations)
    cycle_rows = ((need_rows * parameter_sync_step + pull_rows - 1) // pull_rows) * pull_rows
    return cycle_rows // num_generations, (max_staleness + 1) * cycle_rows


def plan_rollout_batch(
    state: Any, inputs, *, samples=None, model=None, micro_batch_kwargs=None, metric_prefix="dp_balance"
):
    """Plan one complete optimizer batch after generation has revealed sequence lengths.

    Callers apply the same permutation to every per-sample training signal, or reorder whole
    records before building teacher signals. Dataset completions and auxiliary SFT bypass this.
    Streaming samples carry token_lengths alongside remote encoded features. Use that metadata
    to plan without fetching the features from Ray; local/synthesized samples retain the fallback.
    """
    if not getattr(state, "balance_batch", False) or not inputs:
        return None
    from ray import ObjectRef
    from twinkle.model import MicroBatchConfig
    from twinkle.model.dp_balance import DPBalancePlan, plan_dp_balance_for_lengths
    from twinkle.model.micro_batch import sequence_length

    model = state.model if model is None else model
    config = MicroBatchConfig.from_kwargs(dict(micro_batch_kwargs or {}))
    if samples is not None and len(samples) != len(inputs):
        raise ValueError("Rollout samples and encoded inputs must have the same length for batch balancing.")
    lengths = []
    for index, feature in enumerate(inputs):
        counts = getattr(samples[index], "token_lengths", None) if samples is not None else None
        if counts is None and isinstance(feature, ObjectRef):
            raise ValueError(
                "balance_batch needs token_lengths on streaming samples with Ray ObjectRef features "
                f"(missing metadata at batch index {index})."
            )
        lengths.append(sum(counts) if counts is not None else sequence_length(feature))
    plan = plan_dp_balance_for_lengths(
        lengths, model.device_mesh.data_world_size, padding_free=state.padding_free, micro_batch_config=config
    )
    metrics = {key.replace("dp_balance/", metric_prefix + "/", 1): value for key, value in plan.metrics.items()}
    return DPBalancePlan(order=plan.order, strategy=plan.strategy, metrics=metrics)


def record_rollout_step(state, *plans):
    """Record independent policy/value plans without carrying metrics across updates."""
    metrics = {key: value for plan in plans if plan is not None for key, value in plan.metrics.items()}
    return state.record(extra_metrics=metrics) if metrics else state.record()


def publish_rollout_weights(cluster):
    """Keep partial rollouts parked until the new in-place weights have arrived.

    An abort alone allows requests to retry on their old weights while the sync RPC
    waits for those same requests to drain. Pause closes the retry gate; resume only
    after a successful sync. The adaptive adapter owns this transaction itself.
    """
    if cluster.adapter is not None:
        cluster.adapter.publish()
        return
    cluster.sampler.pause_generation()
    # Leave generation paused if transfer fails; never decode partially updated weights.
    cluster.engine.sync_weights()
    cluster.sampler.resume_generation()


def apply_rollout_plan(plan, values):
    return values if plan is None else plan.apply(values)
