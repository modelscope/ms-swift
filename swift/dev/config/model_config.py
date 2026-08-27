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
    #: Extra kwargs forwarded to ``from_pretrained``. Accepts a dict, or a JSON string for command lines.
    #: The escape hatch for a model whose loading needs something this config does not name.
    model_kwargs: Optional[Union[dict, str]] = None
    #: Python files imported before anything is built, so that decorated models, templates, datasets and
    #: reward functions register themselves. Import order is the order given.
    external_plugins: List[str] = field(default_factory=list)
    #: Files whose ``register_model`` / ``register_template`` calls add entries the built-in registries
    #: do not have. Distinct from ``external_plugins``, which is for behaviour rather than registration.
    custom_register_path: List[str] = field(default_factory=list)
    #: Apply swift's Ascend-specific model patches. On by default and only consulted on NPU, where a few
    #: models otherwise hit unsupported ops.
    enable_npu_model_patch: bool = True
    #: Attention implementation for the vision tower, which often supports a different set than the
    #: language tower named by ``attn_impl`` -- so they are chosen separately rather than shared.
    vit_attn_impl: Optional[str] = None
    #: Build only the language tower, dropping the vision one. For training a multimodal checkpoint on
    #: text alone, where the unused tower would still occupy memory.
    language_model_only: bool = False
    #: Tie the MTP heads' weights to the main output embedding instead of giving them their own.
    mtp_shared_weights: bool = False
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

    # -- FP4 low-precision training (Megatron only) ---------------------------------------------
    # Names match the legacy Megatron CLI flags so args_to_configs picks them up by same-name copy;
    # builders/model.py renames them to what megatron's TransformerConfig calls them (fp4 /
    # fp4_param). NVFP4 GEMMs need Blackwell (compute capability 10.0+) and Transformer Engine
    # >= 2.7.0.dev0; mcore-bridge's ModelConfig checks both when the model is built.

    #: Enables FP4 compute. Only 'e2m1' exists, but it stays a value rather than a bool to match
    #: megatron, whose ``fp4`` field is the format and whose None means "no FP4". Activations and
    #: GEMM inputs are quantized per transformer layer; the parameters themselves are NOT, unless
    #: ``fp4_param_gather`` is also set. Mutually exclusive with FP8 (which dev does not expose).
    fp4_format: Optional[Literal['e2m1']] = None
    #: Which FP4 scaling recipe to use. Only NVFP4 block scaling is wired, matching legacy; megatron
    #: also accepts 'custom', which needs a quantizer factory that has no place on this surface.
    fp4_recipe: Literal['nvfp4'] = 'nvfp4'
    #: Also keep the *parameters* in FP4, and all-gather them as FP4, which is where the memory and
    #: bandwidth saving comes from. Requires ``fp4_format`` and the distributed optimizer: the FP32
    #: master shards are re-quantized back into the FP4 parameters by DistributedOptimizer, and no
    #: other optimizer has that step -- without it the parameters would never be updated at all.
    fp4_param_gather: bool = False

    # -- FP8 low-precision training (Megatron only) ----------------------------------------------
    # Same shape as the FP4 block above, and mutually exclusive with it: megatron enters one
    # quantization context per transformer layer, so a config asking for both is rejected.
    # FP8 is the older and far more portable of the two -- Hopper and Ada already have the GEMMs,
    # whereas NVFP4 needs Blackwell.

    #: Enables FP8 compute. 'e4m3' uses that format throughout; 'hybrid' keeps e4m3 for the forward
    #: and e5m2 for the backward, which is the usual choice. None means no FP8.
    fp8_format: Optional[Literal['e4m3', 'hybrid']] = None
    #: FP8 scaling recipe. 'delayed' scales from an amax history (the two knobs below apply only to
    #: it); 'tensorwise' rescales per tensor per step; 'blockwise' is what DeepSeek's FP8 checkpoints
    #: store, so it is the one that round-trips their weights; 'mxfp8' is Blackwell-only.
    fp8_recipe: Literal['tensorwise', 'delayed', 'mxfp8', 'blockwise'] = 'delayed'
    #: Also keep the *parameters* in FP8 and all-gather them as FP8. Carries the same requirement as
    #: its FP4 counterpart, and for the same reason -- megatron itself rejects it without the
    #: distributed optimizer ('--fp8-param-gather only supported with distributed optimizer, ...').
    fp8_param_gather: bool = False
    #: How many past amax values the 'delayed' recipe scales from.
    #: NOTE: 1024 is legacy Megatron-SWIFT's default, NOT megatron's (which is 1). Kept aligned with
    #: legacy on purpose: these two knobs change the numerics, so adopting megatron's defaults would
    #: make an identical-looking config train differently on dev than on legacy.
    fp8_amax_history_len: int = 1024
    #: How the scaling factor is picked out of that history. 'max' is legacy's default; megatron's is
    #: 'most_recent' (see the note above).
    fp8_amax_compute_algo: Literal['most_recent', 'max'] = 'max'
