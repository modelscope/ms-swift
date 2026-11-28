# Copyright (c) ModelScope Contributors. All rights reserved.
"""Controller-side DP balancing for post-rollout training batches."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from swift.dev.recipe.utils import StepState

from swift.dev.utils.logger import get_logger

logger = get_logger()


def balance_update(
    state: StepState,
    update: List[Any],
    lengths: Optional[List[int]],
    *,
    micro_batch_kwargs: Optional[Dict[str, Any]] = None,
    group_size: int = 1,
) -> Tuple[List[Any], Dict[str, float]]:
    """Reorder one optimizer batch's trajectories so ``slice_dp`` splits the workload evenly.

    Every per-sample training signal rides the trajectories, so reordering the list is the
    whole permutation: callers hand back the returned list and every field stays aligned.
    Balancing is skipped (the batch keeps its order, no metrics) when ``state.dp_balance``
    is off, when ``lengths`` is None (dataset rows whose reference features are never
    materialised), or by the planner itself for DP=1 / uneven batches. The PPO value model
    is never reordered: its rank-local token mean is not permutation-invariant.
    """
    if not getattr(state, "dp_balance", False) or lengths is None:
        return update, {}
    from twinkle.model import MicroBatchConfig
    from twinkle.model.dp_balance import plan_dp_balance_for_lengths

    if len(lengths) != len(update):
        raise ValueError(f"batch balancing needs one row length per trajectory ({len(lengths)} for {len(update)}).")
    plan = plan_dp_balance_for_lengths(
        lengths,
        state.model.device_mesh.data_world_size,
        padding_free=state.padding_free,
        micro_batch_config=MicroBatchConfig.from_kwargs(dict(micro_batch_kwargs or {})),
        **({"group_size": group_size} if group_size > 1 else {}),
    )
    return [update[index] for index in plan.order], dict(plan.metrics)


def _episodes_span_multiple_rows(episode_ids: List[Any]) -> bool:
    """Whether any episode is delivered as several rows (collected segments).

    The DP planner moves ``group_size`` consecutive ROWS as one block, which stays group-aligned only
    when each episode is one row (a prompt group is ``group_size`` episodes). An episode's collected
    segments are delivered consecutively, so counting runs of equal ``episode_id`` gives the episode
    count; more rows than episodes means some episode spans several rows and a prompt group covers
    more than ``group_size`` rows -- the fixed stride would cut it mid-group (defect76). Reads the
    light per-row ``episode_ids`` facts, never the trajectory rows (references under ray).
    """
    if not episode_ids:
        return False
    runs = 1
    for index in range(1, len(episode_ids)):
        if episode_ids[index] != episode_ids[index - 1]:
            runs += 1
    return len(episode_ids) != runs


def balance_lanes(
    state: StepState,
    trajs: List[Any],
    lengths: Optional[List[int]],
    *,
    group_size: int,
    micro_batch_kwargs: Optional[Dict[str, Any]] = None,
    episode_ids: Optional[List[Any]] = None,
) -> Tuple[List[Any], Optional[List[int]], Optional[List[int]]]:
    """Plan the DP-balanced order for the whole-batch forwards (reference/teacher/old-logps lanes).

    Returns ``(permuted, order, restore)``: feed the lanes ``permuted``, permute every row-aligned
    argument their consumer reads alongside them (the batch's score rows) with ``order``, and
    row-align its returned rows back to the caller's order with ``restore``. Whole groups of
    ``group_size`` consecutive rows move as blocks, so positional group math stays correct.
    ``(trajs, None, None)`` keeps the caller's order when balancing is off, when the batch has no
    row lengths, or when the shape cannot keep groups whole on equal ranks (warned, not raised).
    """
    if not getattr(state, "dp_balance", False) or not lengths:
        return trajs, None, None
    if len(lengths) != len(trajs):
        raise ValueError(f"lane balancing needs one row length per trajectory ({len(lengths)} for {len(trajs)}).")
    if group_size > 1:
        ids = episode_ids if episode_ids is not None else [t.get("episode_id") for t in trajs]
        if _episodes_span_multiple_rows(ids):
            # The planner moves group_size consecutive ROWS as one block, but a prompt group is
            # num_generations EPISODES. With segment_policy='collect' an episode spans several rows, so a
            # group covers a variable row count and the fixed-stride blocks cut it mid-group, permuting the
            # pieces apart and breaking the contiguous-whole-group contract advantage._check_groups enforces
            # (defect76). Keep the caller's contiguous delivery order for such a batch instead.
            logger.warning(
                "DP lane balancing skipped: collected segments make prompt groups span a variable "
                "number of rows (not group_size=%d); keeping the contiguous delivery order so whole "
                "groups stay together (defect76).",
                group_size,
            )
            return trajs, None, None
    from twinkle.model import MicroBatchConfig
    from twinkle.model.dp_balance import plan_dp_balance_for_lengths

    try:
        plan = plan_dp_balance_for_lengths(
            lengths,
            state.model.device_mesh.data_world_size,
            padding_free=state.padding_free,
            micro_batch_config=MicroBatchConfig.from_kwargs(dict(micro_batch_kwargs or {})),
            group_size=group_size,
        )
    except ValueError as exc:
        logger.warning("DP lane balancing skipped for this batch (%s); the forwards keep the caller order.", exc)
        return trajs, None, None
    restore = [0] * len(plan.order)
    for position, index in enumerate(plan.order):
        restore[index] = position
    return [trajs[index] for index in plan.order], plan.order, restore
