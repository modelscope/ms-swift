# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class TrainConfig:
    """Cross-backend training hyperparameters for lr, batch, optimizer, scheduler, gradient, eval."""

    # === Learning Rate & Scheduling ===
    learning_rate: float = 1e-5
    # Schedule name, resolved by naming.resolve_scheduler (also accepts an external class/source);
    # the v4 spelling --lr_scheduler_type still resolves through LEGACY_ALIASES.
    lr_scheduler: str = 'cosine'
    lr_scheduler_kwargs: Optional[Dict[str, Any]] = None
    warmup_ratio: float = 0.0
    warmup_steps: int = 0

    # === Batch & Steps ===
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = -1
    num_train_epochs: float = 3.0

    # === Gradient ===
    # One name for BOTH backends (legacy Megatron never read max_grad_norm); clip_grad is deprecated.
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None
    vit_gradient_checkpointing: Optional[bool] = None
    # Vision-tower options are separate: the two towers are wrapped independently.
    vit_gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None

    # === Optimizer ===
    optim: str = 'adamw_torch_fused'
    optim_args: Optional[str] = None
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8

    # === Optimizer: GaLore ===
    # Full-param low-rank gradient projection (transformers only); galore_quantization = QGaLore.
    use_galore: bool = False
    galore_target_modules: Optional[List[str]] = None
    galore_rank: int = 128
    galore_update_proj_gap: int = 50
    galore_scale: float = 1.0
    galore_proj_type: str = 'std'
    # No twinkle counterpart; rejected by validate._check_galore.
    galore_optim_per_parameter: bool = False
    galore_with_embedding: bool = False
    galore_quantization: bool = False
    galore_proj_quant: bool = False
    galore_proj_bits: int = 4
    galore_proj_group_size: int = 256
    galore_cos_threshold: float = 0.4
    galore_gamma_proj: int = 2
    galore_queue_size: int = 5

    # === Optimizer: Megatron-only ===
    # Deprecated alias of max_grad_norm (resolve_max_grad_norm folds the two).
    clip_grad: Optional[float] = None
    weight_decay_incr_style: Literal['constant', 'linear', 'cosine'] = 'constant'
    start_weight_decay: Optional[float] = None
    end_weight_decay: Optional[float] = None
    # Megatron's scheduler takes the lr floor as a first-class arg: 'cosine' + min_lr here is
    # cosine_with_min_lr.
    min_lr: float = 0.0

    # === Optimizer: Muon ===
    # Megatron's optimizer (distinct from optim); 'dist_muon' is the sharded, overlap-compatible one.
    optimizer: Literal['adam', 'sgd', 'muon', 'dist_muon'] = 'adam'
    # Dual-backend (Megatron optimizer=muon*; transformers optim='muon' into MuonClip). Same for
    # muon_use_nesterov / muon_num_ns_steps; the other muon_* fields are Megatron-only.
    muon_momentum: float = 0.9
    # Orthogonalise q/k/v separately; the fused weight mixes the three heads' scales. Megatron-only.
    muon_split_qkv: bool = True
    muon_use_nesterov: bool = False
    muon_scale_mode: Literal['spectral', 'unit_rms_norm', 'shape_scaling'] = 'spectral'
    # Precision of the Newton-Schulz matmuls, not of the weights. Megatron-only.
    muon_fp32_matmul_prec: Literal['low', 'medium', 'high'] = 'medium'
    muon_coefficient_type: str = 'quintic'
    muon_num_ns_steps: int = 5  # Newton-Schulz iterations; closer orthogonalisation at linear cost
    # 'blockwise' depends on the TP width, the other two do not. Megatron-only.
    muon_tp_mode: Literal['blockwise', 'duplicated', 'distributed'] = 'blockwise'
    muon_extra_scale_factor: float = 1.
    # For parameters muon does not handle (scalars, biases, norms: no matrix structure).
    muon_scalar_optimizer: str = 'adam'
    # QK-Clip rescales the Q/K WEIGHTS when the peak logit exceeds tau (Kimi K2). DEFAULT OFF: a
    # small tau geometrically collapses a pretrained model's Q/K; opt in only from scratch.
    qk_clip_enabled: bool = False
    # The attention-logit ceiling; 10000 matches legacy, NOT twinkle MuonConfig's aggressive 100.
    qk_clip_tau: float = 10000.0
    sgd_momentum: float = 0.9  # Megatron SGD; read only when optimizer='sgd'
    # Megatron spells adam's epsilon differently from HF's adam_epsilon; each backend reads its own.
    adam_eps: Optional[float] = None

    # === Optimizer: precision & offload (Megatron-only) ===
    # Keep optimizer state below fp32: saves memory, most likely to change convergence.
    use_precision_aware_optimizer: bool = False
    main_params_dtype: Literal['fp32', 'fp16'] = 'fp32'
    main_grads_dtype: Literal['fp32', 'bf16'] = 'fp32'
    exp_avg_dtype: Literal['fp32', 'fp16', 'bf16', 'fp8'] = 'fp32'
    exp_avg_sq_dtype: Literal['fp32', 'fp16', 'bf16', 'fp8'] = 'fp32'
    # fp32 gradient all-reduce under bf16 training: costs bandwidth, removes a drift source.
    accumulate_allreduce_grads_in_fp32: bool = False
    # Hold optimizer state on the host, move per step: device memory at PCIe-traffic cost.
    optimizer_cpu_offload: bool = False
    optimizer_offload_fraction: float = 1.  # fraction offloaded; read only when offload is on
    # Capture the optimizer step in a CUDA graph; needs shapes that never change.
    optimizer_cuda_graph: bool = False

    # === Training cadence (Megatron-only) ===
    # Optional spellings (unset vs explicit); process folds each into its HF counterpart.
    train_iters: Optional[int] = None  # Megatron spelling of max_steps
    micro_batch_size: Optional[int] = None  # Megatron spelling of per_device_train_batch_size
    # Tokens-per-step expressed directly; Megatron derives the accumulation count from it.
    global_batch_size: Optional[int] = None
    eval_iters: int = -1  # evaluation iterations per eval; -1 = the whole eval set
    microbatch_group_size_per_vp_stage: Optional[int] = None  # None lets Megatron choose

    # === LR schedule (Megatron-only) ===
    lr: Optional[float] = None  # Megatron spelling of learning_rate
    # 'WSD' has no HF counterpart, hence the two fields cannot merge.
    lr_decay_style: Literal['constant', 'linear', 'cosine', 'inverse-square-root', 'WSD'] = 'cosine'
    lr_warmup_fraction: Optional[float] = None  # Megatron spelling of warmup_ratio
    lr_warmup_iters: Optional[int] = None  # Megatron spelling of warmup_steps
    # Decay iterations, taken separately from the total length; None = across train_iters.
    lr_decay_iters: Optional[int] = None
    lr_warmup_init: float = 0.  # warmup starts from this rather than 0
    lr_wsd_decay_iters: Optional[int] = None  # the decay leg of WSD; read only when WSD
    lr_wsd_decay_style: Literal['exponential', 'linear', 'cosine', 'minus_sqrt'] = 'exponential'

    # === General ===
    seed: int = 42
    full_determinism: bool = False
    use_liger_kernel: bool = False
    neftune_noise_alpha: Optional[float] = None

    # === Training Strategy ===
    router_aux_loss_coef: float = 0.0
    enable_dft_loss: bool = False
    enable_channel_loss: bool = False
    # Supervised-loss name (v4 --loss_type still resolves via aliases); distinct from
    # RLConfig.loss_type (the GRPO objective), which keeps its name.
    loss: Optional[str] = None
    mrl_dims: Optional[Dict[int, float]] = None
    acc_strategy: Literal['token', 'seq'] = 'token'

    # === Per-group learning rates ===
    # Per-tower lr for multimodal runs (None falls back to learning_rate; inert on a plain LLM).
    vit_lr: Optional[float] = None
    aligner_lr: Optional[float] = None
    # Extra param_groups plugins, appended after the built-in per-tower / LoRA+ grouping.
    param_groups_plugins: List[str] = field(default_factory=list)

    # === Full-parameter freeze ===
    # Full-param path only (no TunerConfig); trainable_* wins over freeze_*; ratio refuses PP>1.
    freeze_parameters: List[str] = field(default_factory=list)
    # Freeze parameters whose name matches (re.search) this regex.
    freeze_parameters_regex: Optional[str] = None
    # Freeze the leading fraction (0~1) by cumulative element count in parameters() order.
    freeze_parameters_ratio: float = 0.0
    trainable_parameters: List[str] = field(default_factory=list)
    trainable_parameters_regex: Optional[str] = None

    # === Precision & Performance ===
    tf32: Optional[bool] = None  # allow tf32 on matmuls/convs; None leaves torch's default alone
    # Empty the CUDA cache every N steps: headroom against fragmentation at a sync cost.
    torch_empty_cache_steps: Optional[int] = None
    # The model's KV cache: off during training, only useful when the run generates.
    use_cache: bool = False

    # === Loss ===
    label_smoothing_factor: float = 0.0  # probability mass off the target token; 0 disables
    # Average over real tokens across the whole global batch rather than per micro-batch; matters
    # for packed/padding-free runs. None takes Megatron's default.
    calculate_per_token_loss: Optional[bool] = None
    # Weight decay on q/k layernorm parameters too; off follows the usual norm exemption.
    apply_wd_to_qk_layernorm: bool = False

    # === Nested third-party configs ===
    # Opaque dict on purpose: the keys belong to that library and change with its version.
    liger_kernel_config: Optional[Dict[str, bool]] = None

    # === Evaluation ===
    eval_strategy: Optional[str] = None
    # Integer interval, or a ratio in (0, 1) resolved against total optimizer steps in plan_steps.
    eval_steps: Optional[float] = None
    eval_on_start: bool = False
    # Skip evaluation until this many steps (or epochs, following eval_strategy) have passed.
    eval_delay: float = 0
    # Reload the best checkpoint at train end (metric_for_best_model); needs it to still exist.
    load_best_model_at_end: bool = False
    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None
    predict_with_generate: bool = False

    # === EvalScope Integration ===
    eval_dataset: List[str] = field(default_factory=list)
    eval_dataset_args: Optional[Dict[str, Any]] = None
    eval_limit: Optional[int] = None
    eval_generation_config: Optional[Dict[str, Any]] = None
    extra_eval_args: Optional[Dict[str, Any]] = None
    # Which engine serves predict_with_generate: 'transformers' wraps the live module in-process;
    # 'vllm'/'sglang' place a co-resident engine and weight-sync into it (ray mode only).
    eval_sampler_backend: Literal['transformers', 'vllm', 'sglang'] = 'transformers'

    # === Early stopping ===
    early_stop_interval: Optional[int] = None

    # === Callbacks ===
    # Registered names fired at Trainer-equivalent points; built-ins append from their own knobs.
    callbacks: List[str] = field(default_factory=list)
