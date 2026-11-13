# Copyright (c) ModelScope Contributors. All rights reserved.
"""Controller-side DP balancing for post-rollout training batches."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple


def balance_update(state: Any, update: List[Any], lengths: Optional[List[int]],
                   *, micro_batch_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[List[Any], Dict[str, float]]:
    """Reorder one optimizer batch's trajectories so ``slice_dp`` splits the workload evenly.

    Every per-sample training signal rides the trajectories, so reordering the list is the
    whole permutation: callers hand back the returned list and every field stays aligned.
    Balancing is skipped (the batch keeps its order, no metrics) when ``state.dp_balance``
    is off, when ``lengths`` is None (dataset rows whose reference features are never
    materialised), or by the planner itself for DP=1 / uneven batches. The PPO value model
    is never reordered: its rank-local token mean is not permutation-invariant.
    """
    if not getattr(state, 'dp_balance', False) or lengths is None:
        return update, {}
    from twinkle.model import MicroBatchConfig
    from twinkle.model.dp_balance import plan_dp_balance_for_lengths

    if len(lengths) != len(update):
        raise ValueError(f'batch balancing needs one row length per trajectory ({len(lengths)} for {len(update)}).')
    plan = plan_dp_balance_for_lengths(
        lengths, state.model.device_mesh.data_world_size, padding_free=state.padding_free,
        micro_batch_config=MicroBatchConfig.from_kwargs(dict(micro_batch_kwargs or {})))
    return [update[index] for index in plan.order], dict(plan.metrics)
