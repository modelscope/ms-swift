# Copyright (c) ModelScope Contributors. All rights reserved.
"""Parameter-efficient fine-tuning configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

@dataclass
class TunerConfig:
    """LoRA (+QLoRA/DoRA/rsLoRA), AdaLoRA, and TrainableTokens settings.

    Scope note: only the tuners that see real-world use on text/multimodal LLMs are covered here.
    The image-generation-oriented methods (LoHa/LoKr/OFT/BOFT ...) are intentionally left out --
    swift does not do text-to-image. Quantization knobs (QLoRA = 4bit + LoRA) are NOT duplicated
    here; they live in QuantizeConfig (quant_method/quant_bits/bnb_4bit_*), since quantization is
    applied at model-load time rather than by the tuner.
    """

    # === Base ===
    #: Which PEFT-backed tuner to build -- 'lora' / 'adalora' / 'trainable_tokens', or 'full' for no
    #: adapter. ``swift.dev.builders.adapter.apply_tuner`` maps it onto the peft config; 'full' is carried as a
    #: None TunerConfig (see ``cli/parser.select_tuner``). DoRA / rsLoRA / QLoRA are LoRA *flags*, not
    #: separate tuners, so there is deliberately no 'dora' / 'rslora' / 'qlora' value here.
    tuner: Literal['full', 'lora', 'adalora', 'trainable_tokens'] = 'lora'
    adapters: List[str] = field(default_factory=list)

    # === Freeze Parameters (multimodal targeting) ===
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
    # peft ``init_lora_weights``: 'true'/'false' (a bool) or a strategy name peft recognises
    # ('gaussian'/'pissa'/'pissa_niter_[N]'/'olora'/...). 'loftq' needs a loftq_config that is not
    # modelled here (peft raises); LoRA-GA ('lora-ga') is unsupported and rejected at build time.
    init_weights: str = 'true'

    # === LoRA+ ===
    # LoRA+ (https://arxiv.org/abs/2402.12354) gives lora_B a higher learning rate than lora_A and the
    # embedding its own rate. Both are learning-rate knobs, consumed by
    # ``swift.dev.builders.optimizer.build_param_groups_spec``, which emits param-group rules matched worker-side
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
