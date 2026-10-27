# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pass colocate trainer-offload and vLLM sleep switches through."""

from typing import Any


def offload_for_rollout(model: Any, rollout_config: Any) -> None:
    """Move the trainer pieces ``rollout_config`` asked to move.

    Both false leaves parameters and optimizer state resident, and still releases the trainer's
    caching allocator so a colocated vLLM process can map its KV cache. vLLM sleep is
    :func:`sleep_for_rollout`. ``reload_to_gpu`` later restores only what this call actually moved.
    """
    model.offload_to_cpu(
        offload_model=bool(rollout_config.offload_model),
        offload_optimizer=bool(rollout_config.offload_optimizer),
    )


def sleep_for_rollout(sampler: Any, sleep_level: int, **kwargs: Any) -> Any:
    """Sleep a colocated sampler the way swift v4 does.

    ``0`` does not call ``sleep``: the engine stays resident. ``1`` offloads vLLM weights to
    CPU and discards the KV cache. ``2`` discards both. Extra keyword arguments (adaptive
    passes ``_worker_ids``) are forwarded, and the sampler's return value is preserved so
    the caller can wait on a remote handle.
    """
    level = int(sleep_level)
    if level <= 0:
        return None
    return sampler.sleep(level=level, **kwargs)
