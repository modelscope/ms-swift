from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LmdeployArguments:
    # lmdeploy
    tp: int = 1
    cache_max_entry_count: float = 0.8
    quant_policy: int = 0  # e.g. 4, 8
    vision_batch_size: int = 1  # max_batch_size in VisionConfig
