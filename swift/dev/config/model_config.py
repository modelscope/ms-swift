"""Model loading, architecture, and precision configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union


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
    new_special_tokens: List[str] = field(default_factory=list)
    num_labels: Optional[int] = None
    problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification', None] = None
    rope_scaling: Optional[str] = None
    max_model_len: Optional[int] = None
    device_map: Optional[Union[dict, str]] = None
    max_memory: Optional[Union[dict, str]] = None
    local_repo_path: Optional[str] = None
    init_strategy: Literal['zero', 'uniform', 'normal', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
                           'kaiming_normal', 'orthogonal', None] = None

    # -- Multi-Token Prediction (Megatron only) ------------------------------------------------
    # MTP adds heads that predict tokens 2..k ahead. Two separate reasons to care, which is why
    # loading and training are separate knobs: the heads are what a serving engine uses as the draft
    # model for speculative decoding, and they are also an auxiliary training objective.

    #: Number of MTP depths to build, load and export. Required for anything else here to apply.
    #: Set it to whatever the checkpoint carries (``num_nextn_predict_layers`` in its HF config) --
    #: mcore-bridge does NOT infer it, so leaving it unset silently drops the MTP weights on load.
    mtp_num_layers: Optional[int] = None
    #: Weight of the MTP loss relative to the main loss. mcore defaults to 0.1; slime uses 0.2 for
    #: RL. Only read when ``enable_mtp_training`` is set.
    mtp_loss_scaling_factor: Optional[float] = None
    #: Train the MTP heads jointly with the main objective. Off by default because it is not free:
    #: it costs one extra transformer layer plus a full-vocabulary projection per depth on every
    #: training pass, and verl's measurements find it only moves the needle when the loss reaches
    #: all parameters. Without it the heads are loaded and exported but never updated.
    enable_mtp_training: bool = False
    #: Cut the MTP heads out of the optimizer entirely. Use with ``mtp_num_layers`` when the goal is
    #: only to carry the heads through to the exported checkpoint: it drops their Adam state, and it
    #: prevents weight decay from eroding heads whose gradient is always zero.
    mtp_freeze: bool = False
    #: Stop the MTP gradient at the main model's hidden states, so only the MTP heads themselves
    #: learn (verl calls this ``detach_encoder`` and recommends it). Safer for RL -- the policy is
    #: then provably unaffected -- at the cost of the effect verl only observed without it.
    mtp_decoder_input_detach: bool = False
