"""vLLM inference engine configuration, rollout mode, and weight sync."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# TODO: integrate it
@dataclass
class RolloutConfig:
    """Rollout/inference engine configuration, scheduling, and weight synchronization."""

    # === vLLM Engine Parameters ===
    vllm_gpu_memory_utilization: float = 0.9
    vllm_tensor_parallel_size: int = 1
    vllm_pipeline_parallel_size: int = 1
    vllm_enable_expert_parallel: bool = False
    vllm_max_num_seqs: Optional[int] = None
    vllm_max_model_len: Optional[int] = None
    vllm_disable_custom_all_reduce: bool = True
    vllm_enforce_eager: bool = False
    vllm_limit_mm_per_prompt: Optional[Dict[str, int]] = None
    vllm_max_lora_rank: int = 16
    vllm_enable_prefix_caching: bool = True
    vllm_use_async_engine: Optional[bool] = None
    vllm_quantization: Optional[str] = None
    vllm_reasoning_parser: Optional[str] = None
    vllm_disable_cascade_attn: bool = False
    vllm_mm_processor_cache_gb: Optional[float] = None
    vllm_speculative_config: Optional[Dict[str, Any]] = None
    vllm_engine_kwargs: Optional[Dict[str, Any]] = None
    vllm_data_parallel_size: int = 1

    # === SGLang Engine Parameters ===
    # Parallel to the vLLM block above rather than shared with it: the two engines name and scope these
    # knobs differently -- SGLang splits data parallelism into a replica count plus attention-level DP,
    # and reserves a fraction of memory statically where vLLM takes a utilisation target -- so one set
    # of fields could not be handed to both without a translation layer that hides those differences.
    sglang_tp_size: int = 1
    sglang_pp_size: int = 1
    sglang_dp_size: int = 1
    sglang_ep_size: int = 1
    sglang_enable_ep_moe: bool = False
    #: Shard attention over the DP ranks as well, instead of replicating it per replica.
    sglang_enable_dp_attention: bool = False
    #: Fraction of device memory reserved up front for weights and KV cache. None lets SGLang choose.
    sglang_mem_fraction_static: Optional[float] = None
    #: Max sequence length the engine is built for. None takes the model's own.
    sglang_context_length: Optional[int] = None
    sglang_disable_cuda_graph: bool = False
    sglang_quantization: Optional[str] = None
    sglang_kv_cache_dtype: str = 'auto'
    #: Defaults to True, unlike the vLLM side: SGLang's custom all-reduce has been the less reliable of
    #: the two, and this preserves the behaviour rollout ran with before the migration.
    sglang_disable_custom_all_reduce: bool = True
    #: Speculative decoding. The three sizes below are only read once an algorithm is named.
    sglang_speculative_algorithm: Optional[str] = None
    sglang_speculative_num_steps: Optional[int] = None
    sglang_speculative_eagle_topk: Optional[int] = None
    sglang_speculative_num_draft_tokens: Optional[int] = None

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

    # === Batch Control ===
    generation_batch_size: Optional[int] = None
    steps_per_generation: Optional[int] = None

    # === Other ===
    teacher_tag_key: str = 'dataset'

    # === Tools & sandbox (multi-turn rollout) ===
    # A multi-turn rollout lets the model call tools inside a sandbox env. Tools are opt-in and come
    # from tool plugins: ``tools`` names registered 'tool' plugins (resolved through PluginRegistry),
    # and an empty list -- the default -- rolls out with no tools at all, exactly as before. The sandbox
    # is a local ``LocalEnv`` by default (light enough to run in-process); naming a ``sandbox_template``
    # switches it to an ``AgentEnv`` microVM instead.
    #: Registered 'tool' plugin names to expose to the model. Empty = no tools.
    tools: List[str] = field(default_factory=list)
    #: AgentENV template name/ID. None builds a local ``LocalEnv``; a value builds an ``AgentEnv`` microVM.
    sandbox_template: Optional[str] = None
    #: Root directory the per-slot ``LocalEnv`` workspaces are created under. None uses a temp dir.
    sandbox_workspace_root: Optional[str] = None
    #: How many env instances to pool. One is leased per concurrent episode, so this is also the max
    #: number of trajectories rolled out at once. Must be >= 1.
    sandbox_num_envs: int = 1
    #: AgentENV control-plane base URL (``sandbox_template`` set). None lets the SDK read ``E2B_API_URL``.
    sandbox_api_url: Optional[str] = None
    #: AgentENV sandbox idle timeout in seconds. None takes ``AgentEnv``'s own default.
    sandbox_timeout: Optional[int] = None
    #: Default per-command timeout in seconds, for both ``LocalEnv`` and ``AgentEnv``.
    sandbox_command_timeout: int = 60
    #: Address-space cap per call for a ``LocalEnv``; None does not cap.
    sandbox_memory_limit_gb: Optional[float] = 2.0
