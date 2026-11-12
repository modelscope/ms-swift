# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


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
    # (CUDA-IPC); 'disaggregated' uses disjoint GPUs (NCCL). None disaggregates. (The v4 'server'
    # spelling is retired: it had no distinct placement -- same disjoint-GPU plan.)
    sampler_mode: Optional[Literal['disaggregated', 'colocate']] = None
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
    # A sync pauses the sampler's running requests and lets them continue on the new weights.
    max_staleness: int = 1
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
    # How many weight syncs the adaptive ClusterAdapter averages each role's pressure over.
    resource_window_steps: int = 3
    # Weight syncs the adaptive ClusterAdapter waits after moving a replica before the next move.
    resource_min_hold_steps: int = 5
    # Explicit estimator overrides; keys validated against RLResourcePlan before actors are built.
    resource_overrides: Optional[Dict[str, Any]] = None

    # === Rollout workers ===
    # Ray CPU replicas the rollout runs on; episodes are routed to them by episode id.
    rollout_num_workers: int = 1

    # === Batch Control ===
    # Prompts per rollout round. None derives from the training batch so each batch fills whole
    # optimizer updates (recipe/utils.resolve_generation_batch_size) -- never the whole dataset:
    # that fallback made the first batch generate for hours before the first training log line.
    generation_batch_size: Optional[int] = None

    # === Tools & sandbox (multi-turn rollout) ===
    # Tools opt-in from 'tool' plugins; the env pool family is picked by sandbox_kind.
    tools: List[str] = field(default_factory=list)
    # Multi-turn is on iff max_turns is set; per-turn length is sampling max_tokens, and
    # max_trajectory_tokens caps the whole trajectory (independent knobs, either may be None).
    max_turns: Optional[int] = None
    max_trajectory_tokens: Optional[int] = None
    # Which sandbox family the pool builds. 'local' (in-process workspaces, default) and
    # 'agentenv' (microVM, needs sandbox_template) are the legacy pair -- leaving the kind at
    # 'local' while naming a sandbox_template still infers agentenv (back-compat). 'openenv'
    # embeds an OpenEnv environment package in-process (needs sandbox_env_name); 'openenv_client'
    # connects to a running OpenEnv server (needs sandbox_env_name and sandbox_api_url or
    # OPENENV_BASE_URL). Same four names the webui resource panel exposes.
    sandbox_kind: Literal['local', 'agentenv', 'openenv', 'openenv_client'] = 'local'
    # OpenEnv environment package name for sandbox_kind='openenv'/'openenv_client'
    # (e.g. 'coding_env' for OpenEnv's code environments: pip install openenv-<name>).
    sandbox_env_name: Optional[str] = None
    # AgentENV template name/ID; None builds a local LocalEnv.
    sandbox_template: Optional[str] = None
    # Directory holding the LocalEnv sessions' workspaces; None uses a temp dir.
    sandbox_workspace_root: Optional[str] = None
    # How many episodes may hold a sandbox session at once, over all env replicas; must be >= 1.
    sandbox_num_envs: int = 1
    # Ray CPU replicas the sandbox env runs on; each holds ceil(sandbox_num_envs / sandbox_num_workers)
    # sessions.
    sandbox_num_workers: int = 1
    # AgentENV control-plane base URL (sandbox_template set); None lets the SDK read E2B_API_URL.
    sandbox_api_url: Optional[str] = None
    # AgentENV sandbox idle timeout in seconds; None takes AgentEnv's own default.
    sandbox_timeout: Optional[int] = None
    # Default per-command timeout in seconds, for both LocalEnv and AgentEnv.
    sandbox_command_timeout: int = 60
    # Address-space cap per call for a LocalEnv; None does not cap.
    sandbox_memory_limit_gb: Optional[float] = 2.0
