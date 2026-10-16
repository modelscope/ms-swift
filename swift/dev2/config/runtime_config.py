# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    """Side-effectful run settings that do not belong to a business recipe."""

    seed: int = 42
    # Data-parallel workers a streaming weight transform (merge / load-time quantize) splits its
    # keyspace across; purely a throughput knob, the output is identical for any value.
    weight_transform_parallel_size: int = 1
