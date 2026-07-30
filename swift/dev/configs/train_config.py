"""Cross-backend training hyperparameters (lr/batch/optimizer/scheduler/gradient/eval)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union


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

    # === Evaluation ===
    eval_strategy: Optional[str] = None
    eval_steps: Optional[int] = None
    eval_on_start: bool = False
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
