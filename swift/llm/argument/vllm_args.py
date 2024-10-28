from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VllmArguments:
    # vllm
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    max_num_seqs: int = 256
    max_model_len: Optional[int] = None
    disable_custom_all_reduce: bool = True  # Default values different from vllm
    enforce_eager: bool = False
    limit_mm_per_prompt: Optional[str] = None  # '{"image": 10, "video": 5}'
    vllm_max_lora_rank: int = 16
    lora_modules: List[str] = field(default_factory=list)
    max_logprobs: int = 20

    def __post_init__(self):
        self.vllm_enable_lora = len(self.lora_modules) > 0
        self.vllm_max_loras = max(len(self.lora_modules), 1)
