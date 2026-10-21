# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass
class ModelConfig:
    """Model path, architecture, dtype, and device mapping."""

    model: Optional[str] = None
    model_type: Optional[str] = None
    model_revision: Optional[str] = None
    task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'reranker', 'generative_reranker', None] = None
    torch_dtype: Literal['bfloat16', 'float16', 'float32', None] = None
    attn_impl: Optional[str] = None
    experts_impl: Optional[str] = None
    enable_kernel: bool = False
    new_special_tokens: List[str] = field(default_factory=list)
    num_labels: Optional[int] = None
    problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification', None] = None
    rope_scaling: Optional[Union[Dict[str, Any], str]] = None
    max_model_len: Optional[int] = None
    max_memory: Optional[Dict[str, Any]] = None
    local_repo_path: Optional[str] = None
    # Extra kwargs forwarded to from_pretrained (JSON object or file); the escape hatch for a model
    # whose loading needs something this config does not name.
    model_kwargs: Optional[Dict[str, Any]] = None
    # Ascend-specific model patches, on by default and only consulted on NPU.
    enable_npu_model_patch: bool = True
    # Vision-tower attention, which often supports a different set than the language tower.
    vit_attn_impl: Optional[str] = None
    # Vision preprocessing overrides (min_pixels, max_pixels, fps, ...) keyed by the lowercase
    # constant names of the model's vision_process module; takes priority over the env fallback.
    vision_process_kwargs: Optional[Dict[str, Any]] = None
    # Build only the language tower (train a multimodal checkpoint on text alone).
    language_model_only: bool = False
    # Tie the MTP heads' weights to the main output embedding instead of their own.
    mtp_shared_weights: bool = False
    init_strategy: Literal['zero', 'uniform', 'normal', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
                           'kaiming_normal', 'orthogonal', None] = None

    # === Multi-Token Prediction (Megatron only) ===
    # Depths from the checkpoint; unset silently drops the MTP weights on load.
    mtp_num_layers: Optional[int] = None
    # MTP loss weight (mcore 0.1, slime 0.2 for RL); only read when enable_mtp_training is set.
    mtp_loss_scaling_factor: Optional[float] = None
    # Train the MTP heads jointly; costs an extra transformer layer + full-vocab projection per
    # depth on every pass. Without it the heads load and export but never update.
    enable_mtp_training: bool = False
    # Cut the MTP heads out of the optimizer (carry them to the export without Adam state, and
    # away from weight decay eroding zero-gradient heads).
    mtp_freeze: bool = False
    # Stop the MTP gradient at the main hidden states (verl's detach_encoder): only the heads
    # learn; safer for RL -- the policy is provably unaffected.
    mtp_decoder_input_detach: bool = False

    # === FP4 low-precision training (Megatron only) ===
    # Only 'e2m1'; params quantized too only with fp4_param_gather. Needs Blackwell + TE >= 2.7.
    fp4_format: Optional[Literal['e2m1']] = None
    # Only NVFP4 block scaling is wired, matching legacy.
    fp4_recipe: Literal['nvfp4'] = 'nvfp4'
    # Keep the *parameters* in FP4 and all-gather them as FP4; requires fp4_format and the
    # distributed optimizer (the only one that re-quantizes master shards back into them).
    fp4_param_gather: bool = False

    # === FP8 low-precision training (Megatron only) ===
    # 'e4m3' throughout, or 'hybrid' (e4m3 forward, e5m2 backward); mutually exclusive with FP4.
    fp8_format: Optional[Literal['e4m3', 'hybrid']] = None
    # 'delayed' scales from an amax history; 'blockwise' round-trips DeepSeek FP8 checkpoints;
    # 'tensorwise' per tensor per step; 'mxfp8' is Blackwell-only.
    fp8_recipe: Literal['tensorwise', 'delayed', 'mxfp8', 'blockwise'] = 'delayed'
    # Same requirement as its FP4 counterpart (distributed optimizer only).
    fp8_param_gather: bool = False
    # 1024 is legacy Megatron-SWIFT's default, NOT megatron's (1): the two knobs change numerics,
    # so an identical-looking config must train identically on dev and legacy.
    fp8_amax_history_len: int = 1024
    # 'max' is legacy's default; megatron's is 'most_recent' (same rationale as above).
    fp8_amax_compute_algo: Literal['most_recent', 'max'] = 'max'
