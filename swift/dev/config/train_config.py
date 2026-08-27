"""Cross-backend training hyperparameters (lr/batch/optimizer/scheduler/gradient/eval)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass
class TrainConfig:
    """Cross-backend training hyperparameters for lr, batch, optimizer, scheduler, gradient, and eval."""

    # === Learning Rate & Scheduling ===
    learning_rate: float = 1e-5
    lr_scheduler_type: str = 'cosine'
    lr_scheduler_kwargs: Optional[Union[dict, str]] = None
    warmup_ratio: float = 0.0
    warmup_steps: int = 0

    # === Batch & Steps ===
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = -1
    num_train_epochs: float = 3.0
    max_epochs: Optional[int] = None

    # === Gradient ===
    # Gradient-clipping threshold for BOTH backends. legacy split this in two (HF `max_grad_norm`,
    # Megatron `clip_grad`) and its Megatron path simply never read `max_grad_norm` -- setting it
    # there was silently dropped. The two are synonyms, so dev carries one name; see `clip_grad`
    # below for the deprecated alias.
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = None
    vit_gradient_checkpointing: Optional[bool] = None
    #: Checkpointing options for the vision tower, separate from ``gradient_checkpointing_kwargs``
    #: because the two towers are wrapped independently.
    vit_gradient_checkpointing_kwargs: Optional[Union[Dict[str, Any], str]] = None

    # === Optimizer ===
    optim: str = 'adamw_torch_fused'
    optim_args: Optional[str] = None
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8

    # === Optimizer: Megatron-only ===
    # Deprecated alias of max_grad_norm, kept so existing Megatron scripts/argv keep working.
    # `None` means "not set" -- it must stay Optional to tell an explicit `--clip_grad 1.0` apart
    # from the default, otherwise the deprecation warning could not fire on the former.
    # `resolve_max_grad_norm()` (dev/optimizer.py) is the single place that folds the two.
    clip_grad: Optional[float] = None
    weight_decay_incr_style: Literal['constant', 'linear', 'cosine'] = 'constant'
    start_weight_decay: Optional[float] = None
    end_weight_decay: Optional[float] = None
    # Megatron's scheduler takes the lr floor as a first-class arg, so 'cosine' + min_lr is what
    # cosine_with_min_lr means here
    min_lr: float = 0.0

    # === Optimizer: Muon (Megatron-only) ===
    #: Which optimizer Megatron builds. Distinct from ``optim`` above, which names a torch/HF optimizer
    #: for the transformers backend -- the two backends construct optimizers by different means, and
    #: neither accepts the other's names. Only consulted when the Megatron backend is in use.
    #: 'dist_muon' is the sharded variant, and the only one compatible with the overlap options in
    #: DistributedConfig: plain 'muon' needs both overlaps off, because it reads whole parameters.
    optimizer: Literal['adam', 'sgd', 'muon', 'dist_muon'] = 'adam'
    #: The rest are read only when ``optimizer`` selects a muon variant. Requires megatron-core>=0.16.
    muon_momentum: float = 0.9
    #: Orthogonalise q, k and v separately rather than as one fused matrix. On by default because the
    #: fused weight's singular values mix the three heads' scales, which is not what muon assumes.
    muon_split_qkv: bool = True
    muon_use_nesterov: bool = False
    #: How the orthogonalised update is rescaled before it is applied.
    muon_scale_mode: Literal['spectral', 'unit_rms_norm', 'shape_scaling'] = 'spectral'
    #: Precision of the matmuls inside the Newton-Schulz iteration, not of the weights.
    muon_fp32_matmul_prec: Literal['low', 'medium', 'high'] = 'medium'
    muon_coefficient_type: str = 'quintic'
    #: Newton-Schulz iterations per step. More is a closer orthogonalisation at linear cost.
    muon_num_ns_steps: int = 5
    #: How the update is computed across tensor-parallel ranks. 'blockwise' orthogonalises each shard
    #: on its own, so its result depends on the TP width; the other two do not.
    muon_tp_mode: Literal['blockwise', 'duplicated', 'distributed'] = 'blockwise'
    muon_extra_scale_factor: float = 1.
    #: Optimizer used for the parameters muon does not handle -- scalars, biases, norms, which have no
    #: matrix structure to orthogonalise.
    muon_scalar_optimizer: str = 'adam'
    #: Megatron's SGD momentum, read only when ``optimizer='sgd'``.
    sgd_momentum: float = 0.9
    #: Megatron spells adam's epsilon differently from HF's ``adam_epsilon`` above; both are kept
    #: because each backend reads its own.
    adam_eps: float = 1e-8

    # === Optimizer: precision & offload (Megatron-only) ===
    #: Keep optimizer state in lower precision than fp32, using the dtypes below. Off by default: it
    #: saves a large fraction of optimizer memory and is the setting most likely to change convergence.
    use_precision_aware_optimizer: bool = False
    main_params_dtype: Literal['fp32', 'fp16'] = 'fp32'
    main_grads_dtype: Literal['fp32', 'bf16'] = 'fp32'
    exp_avg_dtype: Literal['fp32', 'fp16', 'bf16', 'fp8'] = 'fp32'
    exp_avg_sq_dtype: Literal['fp32', 'fp16', 'bf16', 'fp8'] = 'fp32'
    #: Accumulate the gradient all-reduce in fp32 even when training in bf16. Costs bandwidth and
    #: removes the reduction as a source of drift across ranks.
    accumulate_allreduce_grads_in_fp32: bool = False
    #: Hold optimizer state on the host and move it per step. Buys device memory at the cost of PCIe
    #: traffic every step, so it is for a model that does not otherwise fit.
    optimizer_cpu_offload: bool = False
    #: Fraction of that state to offload; 1.0 offloads all of it. Read only when offload is on.
    optimizer_offload_fraction: float = 1.
    #: Capture the optimizer step in a CUDA graph, removing its launch overhead. Requires a step whose
    #: shapes never change.
    optimizer_cuda_graph: bool = False

    # === Training cadence (Megatron-only) ===
    # Megatron counts iterations where HF counts steps and epochs. These are the Megatron spellings; a
    # single resolver is meant to fold each into its HF counterpart, so they stay Optional to tell "not
    # set" apart from an explicit value -- the same reason clip_grad above is Optional.
    #: Megatron spelling of ``max_steps``.
    train_iters: Optional[int] = None
    #: Megatron spelling of ``per_device_train_batch_size``.
    micro_batch_size: Optional[int] = None
    #: Tokens-per-step expressed directly, rather than as micro-batch x accumulation x data-parallel
    #: width. Megatron derives the accumulation count from it, which is the opposite direction from HF.
    global_batch_size: Optional[int] = None
    #: Evaluation iterations per evaluation. -1 means the whole eval set.
    eval_iters: int = -1
    #: Load the weights but not the optimizer state, scheduler, or step count -- i.e. start a new run
    #: from a checkpoint rather than resuming one. True by default, since dev's usual case is
    #: fine-tuning; CheckpointConfig's ``resume_from_checkpoint`` is what continues a run instead.
    finetune: bool = True
    #: Micro-batches per virtual-pipeline stage group. None lets Megatron choose from the interleaving.
    microbatch_group_size_per_vp_stage: Optional[int] = None

    # === LR schedule (Megatron-only) ===
    # As above: the first four are Megatron spellings of fields already present, the rest are schedules
    # HF has no equivalent for.
    #: Megatron spelling of ``learning_rate``.
    lr: Optional[float] = None
    #: Megatron spelling of ``lr_scheduler_type``. 'WSD' (warmup-stable-decay) is the one value with no
    #: HF counterpart, and the reason the two cannot simply be merged.
    lr_decay_style: Literal['constant', 'linear', 'cosine', 'inverse-square-root', 'WSD'] = 'cosine'
    #: Megatron spelling of ``warmup_ratio``.
    lr_warmup_fraction: Optional[float] = None
    #: Megatron spelling of ``warmup_steps``.
    lr_warmup_iters: Optional[int] = None
    #: Iterations over which the lr decays, which Megatron takes separately from the total run length --
    #: so decay can finish before training does. None decays across ``train_iters``.
    lr_decay_iters: Optional[int] = None
    #: Learning rate warmup starts from, rather than 0.
    lr_warmup_init: float = 0.
    #: The decay leg of the WSD schedule. Read only when ``lr_decay_style='WSD'``.
    lr_wsd_decay_iters: Optional[int] = None
    lr_wsd_decay_style: Literal['exponential', 'linear', 'cosine', 'minus_sqrt'] = 'exponential'

    # === General ===
    seed: int = 42
    full_determinism: bool = False
    use_liger_kernel: bool = False
    neftune_noise_alpha: Optional[float] = None
    average_tokens_across_devices: bool = True

    # === Training Strategy ===
    router_aux_loss_coef: float = 0.0
    enable_dft_loss: bool = False
    enable_channel_loss: bool = False
    loss_type: Optional[str] = None
    mrl_dims: Optional[Union[dict, str]] = None
    acc_strategy: Literal['token', 'seq'] = 'token'
    aligner_lr: Optional[float] = None
    vit_lr: Optional[float] = None
    use_logits_to_keep: Optional[bool] = None
    ds3_gather_for_generation: bool = True

    # === Precision & Performance ===
    # torch_dtype in ModelConfig is what the weights are; these are about the run around them.
    #: Run evaluation in bf16 / fp16 regardless of the training dtype. Saves memory at eval time, and
    #: can move the metric slightly, so a comparison across runs should keep it fixed.
    bf16_full_eval: bool = False
    fp16_full_eval: bool = False
    #: Allow tf32 on matmuls and convolutions. Large speedup on Ampere and later at reduced mantissa;
    #: None leaves torch's own default alone.
    tf32: Optional[bool] = None
    torch_compile: bool = False
    torch_compile_backend: Optional[str] = None
    torch_compile_mode: Optional[str] = None
    #: Empty the CUDA cache every N steps. Buys headroom against fragmentation and costs a
    #: synchronisation each time, so it is for runs that OOM late rather than a default.
    torch_empty_cache_steps: Optional[int] = None
    #: Halve the batch size and retry after an OOM, until it fits. Changes the effective batch size
    #: without saying so, which makes a run's loss curve incomparable to others.
    auto_find_batch_size: bool = False
    #: The model's KV cache. Off during training, where it wastes memory and conflicts with gradient
    #: checkpointing; only useful when the run generates.
    use_cache: bool = False

    # === Loss ===
    #: Which batch keys hold labels. Needed only when a model's signature does not make it obvious.
    label_names: Optional[List[str]] = None
    #: Spread this much probability mass off the target token. 0.0 disables it.
    label_smoothing_factor: float = 0.0
    #: Average the loss over real tokens across the whole global batch rather than per micro-batch.
    #: Matters whenever micro-batches hold unequal token counts, i.e. any packed or padding-free run,
    #: where the per-micro-batch mean quietly weights short batches more. None takes Megatron's default.
    calculate_per_token_loss: Optional[bool] = None
    #: Apply weight decay to the q/k layernorm parameters too. Off by default, following the usual
    #: convention that norm parameters are exempt.
    apply_wd_to_qk_layernorm: bool = False

    # === Nested third-party configs ===
    # Kept as opaque dicts on purpose: the keys belong to those libraries and change with their
    # versions, so enumerating them here would go stale on the next upgrade.
    #: Accelerate integration settings; a dict, or a path to a JSON file.
    accelerator_config: Optional[Union[Dict[str, Any], str]] = None
    #: Which Liger ops to patch in, passed through to ``_apply_liger_kernel_to_instance``. Read only
    #: when ``use_liger_kernel`` is on; None patches Liger's own default set.
    liger_kernel_config: Optional[Dict[str, bool]] = None

    # === Evaluation ===
    eval_strategy: Optional[str] = None
    eval_steps: Optional[int] = None
    eval_on_start: bool = False
    #: Run evaluation at all. Off by default; naming an eval strategy or dataset is the usual way to
    #: turn it on.
    do_eval: bool = False
    #: Batches accumulated on device before their predictions move to CPU. None accumulates the whole
    #: eval set, which is what runs out of memory on a large one.
    eval_accumulation_steps: Optional[int] = None
    #: Skip evaluation until this many steps (or epochs, following ``eval_strategy``) have passed.
    eval_delay: float = 0
    #: Reload the best checkpoint when training ends, as ranked by ``metric_for_best_model``. Requires
    #: that checkpoint to still exist, so it does not combine with an aggressive ``save_total_limit``.
    load_best_model_at_end: bool = False
    #: Only compute the loss during evaluation, returning no logits. Much cheaper, and incompatible
    #: with any metric that needs the predictions themselves.
    prediction_loss_only: bool = False
    eval_metric: Optional[str] = None
    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None
    predict_with_generate: bool = False
    max_new_tokens: int = 64
    temperature: float = 0.0

    # === EvalScope Integration ===
    eval_use_evalscope: bool = False
    eval_dataset: List[str] = field(default_factory=list)
    eval_dataset_args: Optional[str] = None
    eval_limit: Optional[int] = None
    eval_generation_config: Optional[str] = None
    extra_eval_args: Optional[str] = None

    # === Callbacks ===
    callbacks: List[str] = field(default_factory=list)
    early_stop_interval: Optional[int] = None

    # === Other ===
    check_model: bool = True
