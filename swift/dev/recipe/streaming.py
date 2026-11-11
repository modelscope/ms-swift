# Copyright (c) ModelScope Contributors. All rights reserved.
"""Recipe policy for Twinkle's streaming driver: batch sizing and weight publication.

Swift chooses optimizer batch sizes and publication cadence. Twinkle owns admission,
version accounting, bounded pulls and sampler pause/resume. Keep those responsibilities
separate from DP balancing, which only permutes an already assembled training batch.
"""

from __future__ import annotations
import math
from typing import Any, NamedTuple


class StreamingBatchPlan(NamedTuple):
    """Coupled driver settings; partition size is in groups, the other fields in trajectories."""

    groups_per_partition: int
    buffer_depth: int
    max_rows_per_pull: int


def plan_streaming_batches(
    *, need_rows: int, num_generations: int, parameter_sync_step: int, max_staleness: int, group_pull: bool
) -> StreamingBatchPlan:
    """Reserve whole publication cycles while keeping each handoff at one training quantum.

    A generation group cannot straddle admission partitions. Round a publication cycle
    to complete optimizer batches AND generation groups, then reserve one cycle for
    the current version and each permitted overlapping version. Group-relative recipes
    also pull whole groups; per-sample recipes can hand off exactly ``need_rows``.

    Capacity and pull size deliberately differ: draining all completed prefetch into
    one handoff would move publication past the next optimizer boundary.
    """
    for name, value, minimum in (
        ("need_rows", need_rows, 1),
        ("num_generations", num_generations, 1),
        ("parameter_sync_step", parameter_sync_step, 1),
        ("max_staleness", max_staleness, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer, got {value!r}")
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if not isinstance(group_pull, bool):
        raise TypeError("group_pull must be a bool")
    quantum = math.lcm(need_rows, num_generations)
    cycle_rows = ((need_rows * parameter_sync_step + quantum - 1) // quantum) * quantum
    return StreamingBatchPlan(
        groups_per_partition=cycle_rows // num_generations,
        buffer_depth=(max_staleness + 1) * cycle_rows,
        max_rows_per_pull=quantum if group_pull else need_rows,
    )


def publish_rollout_weights(*, sampler: Any, engine: Any, adapter: Any = None) -> None:
    """Publish in-place weights with generation parked across the entire transfer.

    The adaptive adapter owns its resource and publication transaction. For static
    placement, close the sampler's retry gate before transferring weights. Aborting
    alone lets partial rollouts retry on old weights while transfer waits for them.
    A failed pause or transfer propagates; never resume partially updated weights.
    Adapter-snapshot publication remains the driver's separate immutable-weight path.
    """
    if adapter is not None:
        adapter.publish()
        return
    sampler.pause_generation()
    engine.sync_weights()
    sampler.resume_generation()
