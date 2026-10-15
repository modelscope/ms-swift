# Copyright (c) ModelScope Contributors. All rights reserved.
"""Process-wide runtime options shared by non-training CLI commands."""
from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    """Side-effectful run settings that do not belong to a business recipe."""

    seed: int = 42
    #: Data-parallel workers a streaming weight transform (merge / load-time quantize) splits its
    #: keyspace across. 1 runs in-process; >1 starts that many Ray CPU workers, each streaming its own
    #: shards. Purely a throughput knob -- the output is identical for any value.
    weight_transform_parallel_size: int = 1
