from dataclasses import dataclass, field
from typing import Literal

import torch.distributed as dist


@dataclass
class MegatronArguments:
    micro_batch_size: int = 1
    global_batch_size: int = 16
    recompute_activations: bool = False
    recompute_granularity: Literal['selective', 'full'] = 'selective'
    use_cpu_initialization: bool = False
    train_iters


dist.barrier()
