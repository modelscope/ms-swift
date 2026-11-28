# Copyright (c) ModelScope Contributors. All rights reserved.
"""Controller-side DP balancing for post-rollout training batches."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import TrainConfig

logger = get_logger()


def micro_batch_config_from(train_config: "TrainConfig") -> Dict[str, Any]:
    """The model's micro-batching options, derived from ``train_config`` in one place.

    A config field the recipe leaves out of these kwargs is a knob nobody reads, so the recipe does
    not pick a subset by hand: ``micro_batch_order`` applies to planned micro-batches whether or not
    ``dynamic_batching`` is on, and the dynamic fields ride along only when it is (validate confines
    them to the GRPO recipe on the HF backend, like the order itself).
    """
    options: Dict[str, Any] = {
        "micro_batch_size": train_config.per_device_train_batch_size,
        "micro_batch_order": train_config.micro_batch_order,
    }
    if train_config.dynamic_batching:
        options.update(
            dynamic_batching=True,
            max_tokens_per_micro_batch=train_config.max_tokens_per_micro_batch,
            packing_algorithm=train_config.packing_algorithm,
        )
    return options


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


def _group_rows(episode_ids: List[Any], group_size: int) -> Optional[List[int]]:
    """The row count of every prompt group, or ``None`` when every episode is one row.

    An episode's collected segments are delivered consecutively, so a run of equal ``episode_id`` is one
    episode and ``group_size`` consecutive episodes are one prompt group. One row per episode means the
    planner's fixed ``group_size`` stride already matches the groups. Otherwise a group covers a variable
    number of rows (segment_policy='collect', defect76) and the planner needs the groups' row counts to
    keep each whole group on one rank. Reads the light per-row ``episode_ids`` facts, never the
    trajectory rows (references under ray).
    """
    episodes: List[int] = []
    for index, episode_id in enumerate(episode_ids):
        if index and episode_id == episode_ids[index - 1]:
            episodes[-1] += 1
        else:
            episodes.append(1)
    if len(episodes) % group_size:
        raise ValueError(f'{len(episodes)} episodes cannot form whole prompt groups of {group_size}')
    if all(count == 1 for count in episodes):
        return None
    return [sum(episodes[start:start + group_size]) for start in range(0, len(episodes), group_size)]


def balance_lanes(state: Any, trajs: List[Any], lengths: Optional[List[int]], *,
                  group_size: int, micro_batch_kwargs: Optional[Dict[str, Any]] = None,
                  episode_ids: Optional[List[Any]] = None
                  ) -> Tuple[List[Any], Optional[List[int]], Optional[List[int]]]:
    """Plan the DP-balanced order for the whole-batch forwards (reference/teacher/old-logps lanes).

    Returns ``(permuted, order, restore)``: feed the lanes ``permuted``, permute every row-aligned
    argument their consumer reads alongside them (the batch's score rows) with ``order``, and
    row-align its returned rows back to the caller's order with ``restore``. Whole prompt groups of
    ``group_size`` episodes move as blocks, so positional group math stays correct.
    ``(trajs, None, None)`` keeps the caller's order when balancing is off or when the batch has no
    row lengths; a shape that cannot keep groups whole on equal ranks keeps the caller's order too
    (fixed-size groups: warned; collected groups of a variable row count: the planner's identity plan).
    """
    if not getattr(state, 'dp_balance', False) or not lengths:
        return trajs, None, None
    if len(lengths) != len(trajs):
        raise ValueError(f'lane balancing needs one row length per trajectory ({len(lengths)} for {len(trajs)}).')
    rows = None
    if group_size > 1:
        ids = episode_ids if episode_ids is not None else [t.get('episode_id') for t in trajs]
        rows = _group_rows(ids, group_size)
    from twinkle.model import MicroBatchConfig
    from twinkle.model.dp_balance import plan_dp_balance_for_lengths

    try:
        plan = plan_dp_balance_for_lengths(
            lengths, state.model.device_mesh.data_world_size, padding_free=state.padding_free,
            micro_batch_config=MicroBatchConfig.from_kwargs(dict(micro_batch_kwargs or {})),
            group_size=group_size, group_rows=rows)
    except ValueError as exc:
        logger.warning('DP lane balancing skipped for this batch (%s); the forwards keep the caller order.', exc)
        return trajs, None, None
    restore = [0] * len(plan.order)
    for position, index in enumerate(plan.order):
        restore[index] = position
    return [trajs[index] for index in plan.order], plan.order, restore
