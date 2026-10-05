"""vLLM inference engine configuration, rollout mode, and weight sync."""
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
    #: Trainer/sampler GPU placement for the online-RL rollout (any backend, vLLM or SGLang -- the
    #: ``vllm_`` prefix is historical). ``'colocate'`` shares one DeviceGroup (CUDA-IPC weight sync);
    #: ``'disaggregated'`` puts the sampler on disjoint GPUs (NCCL weight sync). ``'server'`` is a legacy
    #: alias of ``'disaggregated'``: it is a historical misnomer, NOT an HTTP server -- the RL training path
    #: has no HTTP server, the sampler is a Ray actor reached over a DeviceGroup (see plan §2.H).
    #: ``None`` disaggregates (plan_rl_device_groups treats anything but ``'colocate'`` as disaggregated).
    vllm_mode: Optional[Literal['disaggregated', 'server', 'colocate']] = None
    vllm_enable_lora: bool = False
    #: Which rollout engine backs the independent-process samplers (GRPO/PPO/RFT and the distill family).
    #: Both ``'vllm'`` and ``'sglang'`` are wired: the loop weight-syncs the trained policy into the
    #: sampler every step through a CheckpointEngineManager, which both engines support. ``'transformers'``
    #: is deliberately not an option -- TransformersSampler has no CheckpointEngineMixin, so it cannot be
    #: weight-synced; ``validate_rollout_config`` and ``run_grpo._sampler_backend`` reject it.
    rollout_sampler: Literal['vllm', 'sglang'] = 'vllm'

    # === Async & Scheduling ===
    #: How much the rollout generation may overlap training, i.e. how off-policy the trained batch is.
    #: This is the single source of truth for the async regime; ``async_generate`` below is a legacy
    #: bool alias folded into it by ``process._derive_async_mode``.
    #:
    #: - ``'none'``: synchronous, strictly on-policy (the behaviour policy equals the trained policy every
    #:   step). An online loop (GRPO/PPO/GKD/OPSD/MOPD) with no sync-only feature still rides the SAME
    #:   per-sample streaming driver, at ``max_staleness=0`` -- admit one window, DRAIN it, train it, publish,
    #:   repeat, with NO overlap -- and a colocated sampler is allowed there (serialized by the drain barrier
    #:   + device hand-over). Only a sync-only feature, which needs the WHOLE fixed batch up front (GRPO
    #:   ``dynamic_sample`` / ``advantage_estimator='remax'``, or a distillation off-policy round
    #:   ``lmbda != 1.0``), falls back to the base loop's synchronous ``_run_sync``.
    #: - ``'one_step_off'``: overlap generation with training by ONE step of look-ahead. Every online loop
    #:   (GRPO/PPO and the distillation family GKD/OPSD/MOPD) runs the SAME per-sample streaming driver
    #:   (:class:`swift.dev.recipe._streaming_loop.StreamingLoopMixin`) with ``max_staleness`` pinned to 1
    #:   (a single admission window of look-ahead: train version ``v`` while admitting ``v+1``), so the
    #:   trained batch is one step stale (``staleness <= 1``). Distillation only overlaps the purely
    #:   on-policy regime (``lmbda==1.0``); an off-policy dataset round generates nothing to admit ahead and
    #:   stays synchronous.
    #: - ``'fully_async'``: GRPO/PPO only. The disaggregated sampler keeps generating on its OWN GPUs while
    #:   the trainer advances several steps through the SAME per-sample streaming driver, new policy versions
    #:   are published SPARSELY (each as a pinned adapter snapshot, or by an in-place overwrite that aborts
    #:   and resumes the in-flight generations -- see ``weight_sync_strategy``), and a ready buffer of
    #:   in-flight trajectories absorbs the version skew -- so ``staleness`` may exceed 1, bounded by
    #:   ``max_staleness`` (verl's fully-async regime). Staleness > 1 is a property of the DEPLOYMENT
    #:   (resource isolation + sparse publish + a queue buffer), not of keeping several weight copies: the
    #:   sampler is never asked to sit idle, so queued samples age several versions. The GRPO and PPO loops
    #:   implement both overlapping regimes
    #:   (:class:`swift.dev.recipe.grpo_async.StreamingGRPOLoop` /
    #:   :class:`swift.dev.recipe.ppo_async.StreamingPPOLoop`, both composed from
    #:   :class:`swift.dev.recipe._streaming_loop.StreamingLoopMixin`), reusing twinkle's
    #:   ``RLContextManager`` for the staleness gate and version tracking.
    #:
    #: Any staleness > 0 is off-policy: GRPO trains on raw sampled tokens, so it MUST be paired with
    #: ``rollout_importance_sampling_mode``; PPO's clipped surrogate (like the distillation family's teacher
    #: target) already bounds the update, so it needs no extra correction. The two OVERLAPPING modes
    #: (``'one_step_off'``/``'fully_async'``) both require ``vllm_mode='disaggregated'`` (colocate time-shares
    #: one DeviceGroup and cannot overlap) and are incompatible with ``dynamic_sample`` / ``remax`` (a
    #: sync-only feature needs the whole batch up front; under ``'none'`` it falls back to ``_run_sync``
    #: instead). Multi-turn is NOT refused: GRPO wires it before the dispatch and its streaming loop admits
    #: each episode per-sample. ``validate._check_async_mode`` enforces all of this.
    async_mode: Literal['none', 'one_step_off', 'fully_async'] = 'none'
    #: LEGACY alias of ``async_mode='one_step_off'``, kept so existing ``--async_generate`` scripts run
    #: unchanged. ``process._derive_async_mode`` folds it into ``async_mode`` (True and ``async_mode``
    #: unset -> ``'one_step_off'``); ``validate._check_async_mode`` rejects setting both inconsistently.
    #: Prefer ``async_mode``.
    async_generate: bool = False
    #: How many policy versions an in-flight trajectory may lag the policy that trains it before
    #: ``RLContextManager`` refuses to admit more work (``'fully_async'`` only; ``'one_step_off'`` pins it to
    #: 1). ``1`` admits one window ahead of the oldest untrained one; larger values deepen the buffer and let
    #: the disaggregated sampler run further ahead. Must be >= 1 under ``'fully_async'``. Under ``'none'`` the
    #: dispatch pins the driver to staleness 0 (strict lockstep: admit one window, drain, train, publish), so
    #: this knob is IGNORED there and an explicit non-default value is refused as inert.
    max_staleness: int = 1
    #: Whether a streaming regime may interrupt an in-flight generation at a publish and resume it on the
    #: freshly-synced weights (partial rollout -- twinkle's ``PartialRolloutMixin``, which the dev vLLM/SGLang
    #: samplers expose), instead of letting it finish on the pre-publish policy. REQUIRED under
    #: ``weight_sync_strategy='in_place'`` at an OVERLAPPING staleness (``'one_step_off'``/``'fully_async'``:
    #: a publish overwrites the sampler's single live weight copy under the in-flight generations, so each
    #: must abort and resume on the fresh weights to stay correctable); inert under ``'adapter_snapshot'``
    #: (each version is pinned by its own path, nothing is overwritten) and at staleness 0 (``'none'``: the
    #: drain empties the in-flight set before every publish, so a publish never meets a generation to
    #: resume). It IS live under EVERY overlapping streaming loop -- GRPO/PPO ``'one_step_off'``/
    #: ``'fully_async'`` AND the distillation family's ``'one_step_off'`` (GKD/OPSD/MOPD route to
    #: ``StreamingGKDLoop`` et al.) -- because the driver admits a lookahead window, so a publish meets an
    #: in-flight generation.
    #: ``validate._check_streaming_publication`` enforces the in_place pairing and
    #: ``_reject_inert_streaming_knobs`` refuses an explicit True where it is inert (a staleness-0 or
    #: ``_run_sync`` path).
    allow_partial_rollout: bool = False
    #: How a just-trained policy is published to the sampler.
    #:
    #: - ``'adapter_snapshot'`` (a streaming-driver path): save each trained version as a LoRA adapter
    #:   on disk and pin it by path, so several versions stay resident and the streaming sampler keeps
    #:   generating each in-flight batch from the version it was admitted under -- publishing a new version
    #:   never disturbs a generation already running. Needs a LoRA run (a full-parameter policy has no
    #:   adapter to snapshot) and must NOT set ``allow_partial_rollout`` (nothing is overwritten in place, so
    #:   there is nothing to interrupt and resume -- the flag would be inert). Cannot serve a COLOCATED
    #:   sampler: publishing writes a new path with no weight sync, so nothing hands the shared device back
    #:   to generation each cycle (use ``'in_place'`` there).
    #: - ``'in_place'``: overwrite the sampler's single live weight copy through twinkle's
    #:   ``CheckpointEngineManager`` -- CUDA IPC when colocated, NCCL when disaggregated (the transport
    #:   follows ``vllm_mode``, it is NOT a separate strategy). At staleness 0 (``'none'``) the drain runs
    #:   before every publish, so nothing is in flight and ``allow_partial_rollout`` is not required. Under an
    #:   OVERLAPPING driver -- GRPO/PPO ``'one_step_off'``/``'fully_async'`` and the distillation family's
    #:   ``'one_step_off'`` -- a buffer always has a generation in flight when a version is published, so
    #:   overwriting the one live copy underneath it is sound ONLY with ``allow_partial_rollout``: each publish
    #:   then aborts every in-flight generation and resumes it from its own tokens on the fresh weights
    #:   (twinkle ``PartialRolloutMixin`` + ``InPlaceWeightSync`` abort-on-publish), so no generation decodes
    #:   across the update and its logprobs stay correctable. It supports a full-parameter policy (merged base
    #:   weights) as well as LoRA, and is the only strategy a colocated or full-parameter run can use (dev
    #:   derives it there). ``validate._check_streaming_publication`` enforces the pairing.
    weight_sync_strategy: Literal['in_place', 'adapter_snapshot'] = 'adapter_snapshot'
    #: How often a just-trained policy is published to the sampler, in consume-reported steps: one weight
    #: publication every ``parameter_sync_step`` optimizer steps (GRPO) / recorded steps (PPO). ``1`` (the
    #: default) publishes every step -- the densest sound cadence, matching the historical per-step sync;
    #: ``K > 1`` publishes sparsely (verl's ``trigger_parameter_sync_step``), letting the sampler run K
    #: steps' worth of generations against one published version before the next overwrite, which trades a
    #: little extra staleness for fewer weight transfers. Read by EVERY regime that rides the streaming
    #: driver -- GRPO/PPO/GKD/OPSD/MOPD under ``'none'`` (staleness 0), ``'one_step_off'`` and
    #: ``'fully_async'`` -- and inert only on the synchronous ``_run_sync`` fallback (a sync-only feature),
    #: where ``validate._reject_inert_streaming_knobs`` refuses an explicit non-default value. The staleness
    #: bound (``max_staleness``) is measured in these publication cycles: the tracked policy version bumps
    #: once per publish, never per optimizer step.
    parameter_sync_step: int = 1
    sleep_level: int = 0
    offload_optimizer: bool = False
    offload_model: bool = False

    # === Batch Control ===
    generation_batch_size: Optional[int] = None

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
