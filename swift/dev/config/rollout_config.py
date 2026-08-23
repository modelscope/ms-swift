"""vLLM inference engine configuration, rollout mode, and weight sync."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


# TODO: integrate it
@dataclass
class RolloutConfig:
    """Rollout/inference engine configuration for vLLM, async scheduling, and generation."""

    # === vLLM Engine Parameters ===
    vllm_gpu_memory_utilization: float = 0.9
    vllm_tensor_parallel_size: int = 1
    vllm_pipeline_parallel_size: int = 1
    vllm_enable_expert_parallel: bool = False
    vllm_max_num_seqs: Optional[int] = None
    vllm_max_model_len: Optional[int] = None
    vllm_disable_custom_all_reduce: bool = True
    vllm_enforce_eager: bool = False
    vllm_limit_mm_per_prompt: Optional[str] = None
    vllm_max_lora_rank: int = 16
    vllm_enable_prefix_caching: bool = True
    vllm_use_async_engine: Optional[bool] = None
    vllm_quantization: Optional[str] = None
    vllm_reasoning_parser: Optional[str] = None
    vllm_disable_cascade_attn: bool = False
    vllm_mm_processor_cache_gb: Optional[float] = None
    vllm_speculative_config: Optional[str] = None
    vllm_engine_kwargs: Optional[str] = None
    vllm_data_parallel_size: int = 1

    # === Rollout Mode ===
    use_vllm: bool = False
    vllm_mode: Optional[Literal['server', 'colocate']] = None
    vllm_enable_lora: bool = False

    # === External Server ===
    vllm_server_base_url: Optional[List[str]] = None
    vllm_server_host: Optional[List[str]] = None
    vllm_server_port: List[int] = field(default_factory=lambda: [8000])
    vllm_server_timeout: float = 240.0
    vllm_server_group_port: Optional[List[int]] = None
    vllm_server_pass_dataset: bool = False

    # === Async & Scheduling ===
    async_generate: bool = False
    sleep_level: int = 0
    move_model_batches: Optional[int] = None
    offload_optimizer: bool = False
    offload_model: bool = False
    enable_flattened_weight_sync: bool = True

    # === Generation Parameters ===
    top_k: int = -1
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stop_words: List[str] = field(default_factory=list)
    structured_outputs_regex: Optional[str] = None

    # === Batch Control ===
    generation_batch_size: Optional[int] = None
    steps_per_generation: Optional[int] = None

    # === Other ===
    teacher_tag_key: str = 'dataset'
