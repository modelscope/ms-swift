# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared controller-side DP balancing, with explicit loss-layout constraints."""

from __future__ import annotations
from typing import Any


def _token_count(value):
    if value is None:
        return 0
    if hasattr(value, "numel"):
        return value.numel()
    if not value:
        return 0
    if value[0] is None or isinstance(value[0], (list, tuple)) or hasattr(value[0], "__len__"):
        return sum(_token_count(part) for part in value)
    return len(value)


def _planning_features(inputs):
    """Keep grouped embedding/ranking records atomic; use their total tokens as a cost proxy.

    One ordinary feature is one sequence. Grouped records expand inside the collator, so their
    concatenated length is only a workload proxy, not an exact estimate of collator padding.
    """
    features, grouped = [], False
    for row in inputs:
        if isinstance(row, (list, tuple)):
            # PackingDataset's outer item is an indivisible list of encoded sequences.
            children, _ = _planning_features(row)
            features.append({"input_ids": range(sum(_token_count(child["input_ids"]) for child in children))})
            grouped = True
        elif "input_ids" in row:
            features.append(row)
        else:
            lengths = [_token_count(value) for key, value in row.items() if key.endswith("_input_ids")]
            if not lengths:
                raise ValueError("balance_batch requires encoded input_ids or grouped *_input_ids.")
            features.append({"input_ids": range(sum(lengths))})
            grouped = True
    return features, grouped


def _microstep_plan(features, dp_size, config, padding_free):
    """Redistribute within each existing global micro-batch, retaining its negative pool."""
    from twinkle.model.dp_balance import DPBalancePlan, micro_batch_workload, plan_dp_balance
    from twinkle.model.micro_batch import plan_micro_batches_for_lengths, sequence_length

    if config.dynamic_batching:
        raise ValueError("Batch-dependent losses require fixed micro-batches when balance_batch is enabled.")
    size = len(features) // dp_size
    ranks = [[] for _ in range(dp_size)]
    for start in range(0, size, config.micro_batch_size):
        width = min(config.micro_batch_size, size - start)
        indices = [rank * size + index for rank in range(dp_size) for index in range(start, start + width)]
        plan = plan_dp_balance(
            [features[index] for index in indices], dp_size, padding_free=padding_free, micro_batch_config=config
        )
        order = plan.apply(indices)
        for rank in range(dp_size):
            ranks[rank].extend(order[rank * width : (rank + 1) * width])
    order = [index for rank in ranks for index in rank]
    lengths = [sequence_length(feature) for feature in features]
    metrics = {"dp_balance/rank_tokens_mean": sum(lengths) / dp_size, "dp_balance/preserve_microbatch_pool": 1.0}
    for phase, permutation in [("before", list(range(len(features)))), ("after", order)]:
        rank_tokens, costs, computed = [], [], []
        for rank in range(dp_size):
            values = [lengths[index] for index in permutation[rank * size : (rank + 1) * size]]
            batches = plan_micro_batches_for_lengths(values, config, padding_free=padding_free)
            groups = [[values[index] for index in batch] for batch in batches]
            rank_tokens.append(sum(values))
            costs.append(sum(micro_batch_workload(group, padding_free=padding_free) for group in groups))
            computed.append(sum(sum(group) if padding_free else len(group) * max(group) for group in groups))
        metrics[f"dp_balance/rank_tokens_max_{phase}"] = max(rank_tokens)
        metrics[f"dp_balance/rank_tokens_min_{phase}"] = min(rank_tokens)
        metrics[f"dp_balance/workload_imbalance_{phase}"] = max(costs) / (sum(costs) / dp_size) if sum(costs) else 1.0
        metrics[f"dp_balance/micro_batches_{phase}"] = (size + config.micro_batch_size - 1) // config.micro_batch_size
        if not padding_free:
            metrics[f"dp_balance/padding_ratio_{phase}"] = 1 - sum(lengths) / sum(computed) if sum(computed) else 0.0
    return DPBalancePlan(order=order, strategy="microbatch_pool", metrics=metrics)


def plan_training_batch(
    state: Any,
    inputs,
    *,
    model=None,
    micro_batch_kwargs=None,
    group_size=1,
    keep_microbatch_pool=False,
    metric_prefix="dp_balance",
):
    """Return a shared Twinkle permutation, or None when disabled.

    Callers apply it to whole records BEFORE producing teacher signals, or to every explicit
    per-sample field AFTER producing signals. Never infer aligned fields by a coincidental length.
    ``group_size`` protects chosen/rejected pairs and local hard-mining batches. InfoNCE instead
    keeps each global micro-batch's membership, since its negative pool spans the DP ranks.
    """
    if not getattr(state, "balance_batch", False) or not inputs:
        return None
    from twinkle.model import MicroBatchConfig, plan_dp_balance
    from twinkle.model.dp_balance import DPBalancePlan

    model = state.model if model is None else model
    dp_size = model.device_mesh.data_world_size
    config = MicroBatchConfig.from_kwargs(dict(micro_batch_kwargs or {}))
    padding_free = state.padding_free
    features, grouped = _planning_features(inputs)
    if keep_microbatch_pool and dp_size > 1 and len(inputs) % dp_size == 0:
        if config is None:
            raise ValueError("Preserving an in-batch negative pool requires micro_batch_size.")
        plan = _microstep_plan(features, dp_size, config, padding_free)
    else:
        plan = plan_dp_balance(
            features, dp_size, padding_free=padding_free, micro_batch_config=config, group_size=group_size
        )
    metrics = dict(plan.metrics)
    if grouped:
        metrics["dp_balance/grouped_input_cost_proxy"] = 1.0
        # Logical grouped-token proxies do not describe the collator's physical padding.
        metrics = {key: value for key, value in metrics.items() if "padding_ratio" not in key}
    metrics = {key.replace("dp_balance/", metric_prefix + "/", 1): value for key, value in metrics.items()}
    return DPBalancePlan(order=plan.order, strategy=plan.strategy, metrics=metrics)


def record_training_step(state, *plans):
    """Merge this update's independent model plans, without carrying metrics across updates."""
    metrics = {key: value for plan in plans if plan is not None for key, value in plan.metrics.items()}
    return state.record(extra_metrics=metrics) if metrics else state.record()


def apply_training_plan(plan, values):
    return values if plan is None else plan.apply(values)


def prepare_supervised_batch(state, samples, encoder, *, micro_batch_size, task_type=None, loss=None):
    """Encode once, then keep pair layouts, detached reference signals and loss pools aligned."""
    kwargs = encoder(samples, micro_batch_size=micro_batch_size)
    embedding_loss = (loss or "infonce") if task_type == "embedding" else None
    group_size = getattr(encoder, "balance_group_size", 1)
    if embedding_loss == "online_contrastive" or task_type in ("reranker", "generative_reranker"):
        rank_size = len(kwargs["inputs"]) // state.model.device_mesh.data_world_size
        micro = kwargs["micro_batch_size"]
        # Keep local hard-mining / candidate normalization batches intact, including short tails.
        group_size = micro if rank_size % micro == 0 else max(1, rank_size)
    plan = plan_training_batch(
        state,
        kwargs["inputs"],
        micro_batch_kwargs=kwargs,
        group_size=group_size,
        keep_microbatch_pool=embedding_loss == "infonce",
    )
    if plan is not None:
        kwargs = dict(kwargs, inputs=plan.apply(kwargs["inputs"]))
        for key in getattr(encoder, "balance_fields", ()):
            if kwargs.get(key) is not None:
                kwargs[key] = plan.apply(kwargs[key])
    return kwargs, plan
