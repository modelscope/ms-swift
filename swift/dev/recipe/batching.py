# Copyright (c) ModelScope Contributors. All rights reserved.
"""Controller-side DP balancing for post-rollout training batches."""

from __future__ import annotations
from typing import Any


def plan_rollout_batch(state: Any, inputs, *, model=None, micro_batch_kwargs=None, metric_prefix="dp_balance"):
    """Plan one complete optimizer batch after generation has revealed sequence lengths.

    Callers apply the same permutation to every per-sample training signal, or reorder whole
    records before building teacher signals. Dataset completions and auxiliary SFT bypass this.
    """
    if not getattr(state, "balance_batch", False) or not inputs:
        return None
    from twinkle.model import MicroBatchConfig, plan_dp_balance
    from twinkle.model.dp_balance import DPBalancePlan

    model = state.model if model is None else model
    config = MicroBatchConfig.from_kwargs(dict(micro_batch_kwargs or {}))
    plan = plan_dp_balance(
        inputs, model.device_mesh.data_world_size, padding_free=state.padding_free, micro_batch_config=config
    )
    metrics = {key.replace("dp_balance/", metric_prefix + "/", 1): value for key, value in plan.metrics.items()}
    return DPBalancePlan(order=plan.order, strategy=plan.strategy, metrics=metrics)


def record_rollout_step(state, *plans):
    """Record independent policy/value plans without carrying metrics across updates."""
    metrics = {key: value for plan in plans if plan is not None for key, value in plan.metrics.items()}
    return state.record(extra_metrics=metrics) if metrics else state.record()


def apply_rollout_plan(plan, values):
    return values if plan is None else plan.apply(values)
