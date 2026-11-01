# Copyright (c) ModelScope Contributors. All rights reserved.
"""The streaming control plane over already-built resources.

``build_streaming_driver`` assembles twinkle's StreamingDriver from a built
generation engine (the sampler resource), a prompt dataloader and the
scheduling configs. It is the algorithm-free half of the recipe assemblies:
scoring, regeneration verdicts, baselines and adapter snapshots stay with the
callers (a recipe closes over its estimators; the canvas wires them as
separate stages), so this builder never branches on an algorithm name.
"""
from __future__ import annotations

import math
from typing import Any, Optional


def build_streaming_driver(
        *,
        engine: Any,
        prompt_dataloader: Any,
        rollout_config: Any,
        rl_config: Any,
        generation_config: Optional[Any] = None,
        train_batch_size: int,
        gradient_accumulation_steps: int,
        num_train_epochs: float = 1.0,
        max_steps: int = 0,
) -> Any:
    """One StreamingDriver over a built engine, ready to iterate.

    The engine carries its template (set at sampler build time); the rollout
    wraps it with the sampling params derived from the rl/generation configs.
    The stream geometry (staleness, pull quantum, buffer depth) follows the
    same derivation the recipe assemblies use, so a canvas graph and a recipe
    reach the same driver shape for the same configs. Scoring, regeneration
    and snapshotting are left to the caller: the driver stays a pure control
    plane (admission, staleness, version bookkeeping, batch handover).
    """
    from twinkle.data_format import SamplingParams
    from twinkle_agentic.async_rl import StreamingDriver
    from twinkle_agentic.rollout import MultiTurnRollout

    if rollout_config.async_mode == 'none':
        max_staleness = 0
    elif rollout_config.async_mode == 'one_step_off':
        max_staleness = 1
    else:
        max_staleness = rollout_config.max_staleness
    num_generations = rl_config.num_generations
    need_rows = train_batch_size * gradient_accumulation_steps
    # The buffer must hold one full pull quantum -- lcm(need_rows,
    # num_generations) rows -- or the assembly deadlocks; groups_per_partition
    # bounds a publish cycle's group count.
    groups_per_partition = max(1, rollout_config.parameter_sync_step)
    pull_rows = need_rows * num_generations // math.gcd(need_rows, num_generations)
    buffer_depth = max((max_staleness + 1) * groups_per_partition * num_generations, pull_rows)

    params: dict = {'max_tokens': rl_config.max_completion_length}
    for key in ('temperature', 'top_p', 'top_k'):
        value = getattr(generation_config, key, None)
        if value is not None:
            params[key] = value
    rollout = MultiTurnRollout(
        engine,
        getattr(engine, 'template', None),
        sampling_params=SamplingParams(**params),
        max_turns=(rollout_config.max_turns or 1),
        max_trajectory_tokens=rollout_config.max_trajectory_tokens,
        allow_partial_rollout=rollout_config.allow_partial_rollout,
    )
    return StreamingDriver(
        version_control=True,
        prompt_stream=prompt_dataloader,
        num_train_epochs=num_train_epochs,
        max_staleness=max_staleness,
        num_generations=num_generations,
        groups_per_partition=groups_per_partition,
        buffer_depth=buffer_depth,
        rollout=rollout,
        group_pull=True,
        need_rows=need_rows,
        max_steps=(max_steps or None),
    )
