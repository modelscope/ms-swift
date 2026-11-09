# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    """Side-effectful run settings that do not belong to a business recipe."""

    seed: int = 42
    # Data-parallel workers the MERGE weight transform splits its key-space across; purely a
    # throughput knob, the output is identical for any value. The load-time quantize transform is
    # single-process and does not read this knob.
    weight_transform_parallel_size: int = 1
