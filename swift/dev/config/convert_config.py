"""HF <-> Megatron(mcore) weight-conversion configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ConvertConfig:
    """Conversion direction, mcore checkpoint sources, and precision-test options.

    Separate from DistributedConfig on purpose: those fields describe how a model is PARALLELIZED for
    training, whereas these describe a one-shot offline format migration. The parallelism sizes
    (tp/pp/ep/...) still come from DistributedConfig, because the mcore checkpoint layout depends on
    them.
    """

    # === direction ===
    # Exactly one is expected. hf->mcore is the `to_mcore` case with no mcore_model set; mcore->hf is
    # `to_hf`; mcore->mcore (to_mcore WITH mcore_model) is the resharding / LoRA-merge case.
    to_mcore: bool = False
    to_hf: bool = False

    # === sources ===
    mcore_model: Optional[str] = None
    # An mcore-format LoRA checkpoint; when set it is merged into the base weights before saving.
    mcore_adapter: Optional[str] = None

    # === torch-dist sharding ===
    # None => derived from the checkpoint size (one shard-writer thread per ~10GB, min 2), matching
    # legacy. Only affects write throughput, not the produced weights.
    thread_count: Optional[int] = None

    # === verification ===
    # Runs both models on the same input and compares outputs. Costs a second model in memory, so it
    # is opt-in.
    test_convert_precision: bool = False
    test_convert_dtype: Literal['float16', 'bfloat16', 'float32'] = 'float32'
