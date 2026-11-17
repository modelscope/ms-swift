# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pass the colocate vLLM sleep switch through."""

from typing import Any


def sleep_for_rollout(sampler: Any, sleep_level: int, **kwargs: Any) -> Any:
    """Sleep a colocated sampler the way swift v4 does.

    ``0`` does not call ``sleep``: the engine stays resident. ``1`` offloads vLLM weights to
    CPU and discards the KV cache. ``2`` discards both. Extra keyword arguments are forwarded,
    and the sampler's return value is preserved so the caller can wait on a remote handle.
    """
    level = int(sleep_level)
    if level <= 0:
        return None
    return sampler.sleep(level=level, **kwargs)
