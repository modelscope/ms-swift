# Copyright (c) ModelScope Contributors. All rights reserved.
"""Controller-side DP balancing for post-rollout training batches."""

from __future__ import annotations
from typing import Any


def plan_rollout_batch(
    state: Any, inputs, *, lengths=None, model=None, micro_batch_kwargs=None, metric_prefix="dp_balance"
):
    """Plan one complete optimizer batch after generation has revealed sequence lengths.

    Callers apply the same permutation to every per-sample training signal, or reorder whole
    records before building teacher signals. Dataset completions and auxiliary SFT bypass this.
    The row lengths come from the driver's light batch summary (prompt + response, per row) so
    reference trajectories are never materialised for their lengths; local features fall back
    to their own sequence_length.
    """
    if not getattr(state, "balance_batch", False) or not inputs:
        return None
    from twinkle.model import MicroBatchConfig
    from twinkle.model.dp_balance import DPBalancePlan, plan_dp_balance_for_lengths
    from twinkle.model.micro_batch import sequence_length

    model = state.model if model is None else model
    config = MicroBatchConfig.from_kwargs(dict(micro_batch_kwargs or {}))
    if lengths is not None and len(lengths) != len(inputs):
        raise ValueError("Rollout row lengths and encoded inputs must have the same length for batch balancing.")
    row_lengths = []
    for index, feature in enumerate(inputs):
        count = lengths[index] if lengths is not None else None
        if count is None and not isinstance(feature, dict):
            raise ValueError(
                "balance_batch needs the batch summary's row lengths for reference trajectories "
                f"(missing at batch index {index})."
            )
        row_lengths.append(count if count is not None else sequence_length(feature))
    plan = plan_dp_balance_for_lengths(
        row_lengths, model.device_mesh.data_world_size, padding_free=state.padding_free, micro_batch_config=config
    )
    metrics = {key.replace("dp_balance/", metric_prefix + "/", 1): value for key, value in plan.metrics.items()}
    return DPBalancePlan(order=plan.order, strategy=plan.strategy, metrics=metrics)


def record_rollout_step(state, *plans):
    """Record independent policy/value plans without carrying metrics across updates."""
    metrics = {key: value for plan in plans if plan is not None for key, value in plan.metrics.items()}
    return state.record(extra_metrics=metrics) if metrics else state.record()


def apply_rollout_plan(plan, values):
    return values if plan is None else plan.apply(values)
