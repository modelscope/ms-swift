# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

# Weight-publication strategy name the streaming RL driver selects; a shared alias so the config
# field and the recipe constructor params cannot drift (twinkle's WeightSyncStrategy is the object).
WeightSyncStrategyName = Literal['in_place', 'adapter_snapshot']


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
    # Parallel to the vLLM block, not shared: the engines name and scope these knobs differently.
    sglang_tp_size: int = 1
    sglang_pp_size: int = 1
    sglang_dp_size: int = 1
    sglang_ep_size: int = 1
    sglang_enable_ep_moe: bool = False
    sglang_enable_dp_attention: bool = False  # shard attention over the DP ranks too
    sglang_mem_fraction_static: Optional[float] = None  # fraction reserved up front; None = engine
    sglang_context_length: Optional[int] = None  # max seq len; None takes the model's own
    sglang_disable_cuda_graph: bool = False
    sglang_quantization: Optional[str] = None
    sglang_kv_cache_dtype: str = 'auto'
    # True unlike the vLLM side: SGLang's custom all-reduce has been the less reliable of the two.
    sglang_disable_custom_all_reduce: bool = True
    # Speculative decoding; the sizes below are read only once an algorithm is named.
    sglang_speculative_algorithm: Optional[str] = None
    sglang_speculative_num_steps: Optional[int] = None
    sglang_speculative_eagle_topk: Optional[int] = None
    sglang_speculative_num_draft_tokens: Optional[int] = None

    # === Rollout Mode ===
    # Trainer/sampler GPU placement (renamed from v4 vllm_mode): 'colocate' shares one DeviceGroup
    # (CUDA-IPC); 'disaggregated'/'server' use disjoint GPUs (NCCL). None disaggregates.
    sampler_mode: Optional[Literal['disaggregated', 'server', 'colocate']] = None
    # Which engine backs the independent-process samplers; 'vllm' and 'sglang' are both wired
    # (weight-sync via CheckpointEngineManager). 'transformers' cannot be weight-synced.
    rollout_sampler: Literal['vllm', 'sglang'] = 'vllm'

    # === Async & Scheduling ===
    # How far the rollout may run ahead (off-policy degree); async_generate is the legacy alias.
    async_mode: Literal['none', 'one_step_off', 'fully_async'] = 'none'
    # LEGACY alias of async_mode='one_step_off'; process folds it in when async_mode is unset and
    # validate rejects setting both inconsistently. Prefer async_mode.
    async_generate: bool = False
    # How many policy versions an in-flight trajectory may lag the training one (fully_async only;
    # one_step_off pins it to 1; 'none' pins the driver to 0 and refuses an explicit non-default).
    max_staleness: int = 1
    # Interrupt an in-flight generation at a publish and resume it on the fresh weights
    # (PartialRolloutMixin); REQUIRED under in_place at an overlapping staleness, inert otherwise.
    allow_partial_rollout: bool = False
    # How a trained policy is published: 'adapter_snapshot' saves each version as a LoRA adapter
    # path; 'in_place' overwrites the live copy (full-param ok; partial rollout at overlap).
    weight_sync_strategy: WeightSyncStrategyName = 'adapter_snapshot'
    # Transport backend for every weight sync into the rollout sampler: 'full' ships the whole
    # weight set; 'delta_sharded' ships only the changed elements as bit-exact sparse patches
    # (first sync still seeds densely). Needs disjoint-GPU placement (sampler_mode
    # disaggregated/server) and a vllm rollout sampler; the sync stays `sync_weights`, so the
    # training loop itself is unchanged.
    weight_sync_backend: Literal['full', 'delta_sharded'] = 'full'
    # One weight publication every K recorded steps; 1 = every step (historical cadence), K > 1
    # publishes sparsely (verl's trigger_parameter_sync_step). Inert on the sync fallback.
    parameter_sync_step: int = 1
    # Colocate vLLM sleep, same meaning as swift v4. 0 leaves the engine resident, 1 offloads
    # its weights and discards the KV cache, 2 discards both. Ignored when the sampler is not
    # colocated. resource_mode=adaptive accepts 0 and 1 only; 2 would drop weights the shared
    # sampler still needs.
    sleep_level: int = 0
    # Colocate only, and independent of each other. Move the trainer weights, the optimizer
    # state, both, or neither between rollout and the update. Default False leaves them on the
    # GPU. Trainer offload is separate from vLLM sleep. Ignored when the sampler is not
    # colocated. resource_mode=adaptive still offloads while it borrows trainer GPUs and
    # refuses to start when both of these are false.
    offload_optimizer: bool = False
    offload_model: bool = False

    # === Resource Planning ===
    # 'manual' keeps every knob; 'estimate' applies one static plan; 'adaptive' moves replicas.
    resource_mode: Literal['manual', 'estimate', 'adaptive'] = 'manual'
    # Maximum physical GPUs available to the run; None derives from the cluster (manual ignores).
    resource_total_gpus: Optional[int] = None
    # Per-card memory override in GiB; None uses the smallest visible cluster GPU Ray reports.
    resource_gpu_memory_gib: Optional[float] = None
    # Disjoint GPUs reserved at initialization for runtime role changes.
    resource_flex_gpus: int = 0
    # Optimizer-step windows the adaptive controller compares cumulative statistics over.
    resource_window_steps: int = 8
    # Minimum optimizer steps between two completed resource changes.
    resource_min_hold_steps: int = 16
    # Explicit estimator overrides; keys validated against RLResourcePlan before actors are built.
    resource_overrides: Optional[Dict[str, Any]] = None

    # === Batch Control ===
    generation_batch_size: Optional[int] = None

    # === Tools & sandbox (multi-turn rollout) ===
    # Tools opt-in from 'tool' plugins; LocalEnv by default, AgentEnv microVM when template set.
    tools: List[str] = field(default_factory=list)
    # Multi-turn is on iff max_turns is set; per-turn length is sampling max_tokens, and
    # max_trajectory_tokens caps the whole trajectory (independent knobs, either may be None).
    max_turns: Optional[int] = None
    max_trajectory_tokens: Optional[int] = None
    # AgentENV template name/ID; None builds a local LocalEnv.
    sandbox_template: Optional[str] = None
    # Root directory for the per-slot LocalEnv workspaces; None uses a temp dir.
    sandbox_workspace_root: Optional[str] = None
    # How many env instances to pool (one leased per concurrent episode); must be >= 1.
    sandbox_num_envs: int = 1
    # AgentENV control-plane base URL (sandbox_template set); None lets the SDK read E2B_API_URL.
    sandbox_api_url: Optional[str] = None
    # AgentENV sandbox idle timeout in seconds; None takes AgentEnv's own default.
    sandbox_timeout: Optional[int] = None
    # Default per-command timeout in seconds, for both LocalEnv and AgentEnv.
    sandbox_command_timeout: int = 60
    # Address-space cap per call for a LocalEnv; None does not cap.
    sandbox_memory_limit_gb: Optional[float] = 2.0
