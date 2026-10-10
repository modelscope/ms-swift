# Copyright (c) ModelScope Contributors. All rights reserved.
"""Parameter-efficient fine-tuning configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

@dataclass
class TunerConfig:
    """LoRA (+QLoRA/DoRA/rsLoRA/LoRA-GA), AdaLoRA, TrainableTokens, and LISA settings.

    Scope note: only the tuners that see real-world use on text/multimodal LLMs are covered here.
    The image-generation-oriented methods (LoHa/LoKr/OFT/BOFT ...) are intentionally left out --
    swift does not do text-to-image. Quantization knobs (QLoRA = 4bit + LoRA) are NOT duplicated
    here; they live in QuantizeConfig (quant_method/quant_bits/bnb_4bit_*), since quantization is
    applied at model-load time rather than by the tuner.
    """

    # === Base ===
    tuner_backend: Literal['peft', 'unsloth'] = 'peft'
    #: Which PEFT-backed tuner to build -- 'lora' / 'adalora' / 'trainable_tokens', or 'full' for no
    #: adapter. ``swift.dev.adapter.apply_tuner`` maps it onto the peft config; 'full' is carried as a
    #: None TunerConfig (see ``cli/parser.select_tuner``). DoRA / rsLoRA / QLoRA are LoRA *flags*, not
    #: separate tuners, so there is deliberately no 'dora' / 'rslora' / 'qlora' value here.
    tuner: str = 'lora'
    adapters: List[str] = field(default_factory=list)

    # === Freeze Parameters (multimodal targeting) ===
    # These three select which towers a multimodal LoRA run trains when target_modules is the default
    # 'all-linear': twinkle expands 'all-linear' worker-side (against the real modules) into a target
    # regex over the unfrozen towers, so the flags ride to add_adapter_to_model as plain bools -- they
    # cannot be resolved on the driver, where the model under Ray is a PROXY with no modules. Defaults
    # match legacy: train the language model, leave the vision tower and the aligner frozen. A plain
    # LLM has an empty ModelArch, so there is nothing to expand and the flags are inert. freeze_vit is
    # additionally read by _derive_vit_gradient_checkpointing to default vit_gradient_checkpointing
    # (recompute the vision tower only when it is trained). The full-parameter freeze knobs
    # (freeze_parameters / freeze_parameters_regex / freeze_parameters_ratio / trainable_parameters /
    # trainable_parameters_regex) deliberately do NOT live here: a full run carries no TunerConfig at
    # all (cli/parser.select_tuner and _megatron_compat._select_megatron_tuner both map tuner='full' to
    # None), so those five moved to TrainConfig -- the same reason the full-parameter GaLore knobs live
    # there rather than here.
    freeze_llm: bool = False
    freeze_vit: bool = True
    freeze_aligner: bool = True

    # === Target Modules ===
    target_modules: List[str] = field(default_factory=lambda: ['all-linear'])
    target_regex: Optional[str] = None
    target_parameters: Optional[List[str]] = None
    modules_to_save: List[str] = field(default_factory=list)

    # === LoRA ===
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: Literal['none', 'all'] = 'none'
    use_rslora: bool = False
    use_dora: bool = False
    # Lora: 'true'/'false'/'gaussian'/'pissa'/'pissa_niter_[number of iters]'/'olora'/'lora-ga'
    # ('loftq' also exists upstream but needs a loftq_config, which is not modelled here.)
    init_weights: str = 'true'

    # === LoRA+ ===
    # LoRA+ (https://arxiv.org/abs/2402.12354) gives lora_B a higher learning rate than lora_A and the
    # embedding its own rate. Both are learning-rate knobs, consumed by
    # ``swift.dev.optimizer.build_param_groups_spec``, which emits param-group rules matched worker-side
    # (by name / parent-module type) rather than touching the peft config -- the same reason they ride
    # here next to the LoRA settings they qualify instead of on TrainConfig. ``lorap_lr_ratio=None``
    # leaves LoRA+ off (a plain single-rate LoRA run); ``lorap_emb_lr`` only takes effect once it is on.
    lorap_lr_ratio: Optional[float] = None
    lorap_emb_lr: float = 1e-6

    # === TrainableTokens ===
    # Trains only the given embedding rows instead of the whole embedding matrix -- the standard way
    # to learn newly added special tokens. Can be used standalone or alongside LoRA.
    trainable_token_indices: Optional[List[int]] = None

    # === AdaLoRA ===
    adalora_target_r: int = 8
    adalora_init_r: int = 12
    adalora_tinit: int = 0
    adalora_tfinal: int = 0
    adalora_deltaT: int = 1
    adalora_beta1: float = 0.85
    adalora_beta2: float = 0.85
    adalora_orth_reg_weight: float = 0.5
