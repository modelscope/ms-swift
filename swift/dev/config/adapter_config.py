"""Parameter-efficient fine-tuning configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class TunerConfig:
    """LoRA (+QLoRA/DoRA/rsLoRA/LoRA+/LoRA-GA), AdaLoRA, TrainableTokens, GaLore, and LISA settings.

    Scope note: only the tuners that see real-world use on text/multimodal LLMs are covered here.
    The image-generation-oriented methods (LoHa/LoKr/OFT/BOFT ...) are intentionally left out --
    swift does not do text-to-image. Quantization knobs (QLoRA = 4bit + LoRA) are NOT duplicated
    here; they live in QuantizeConfig (quant_method/quant_bits/bnb_4bit_*), since quantization is
    applied at model-load time rather than by the tuner.
    """

    # === Base ===
    tuner_backend: Literal['peft', 'unsloth'] = 'peft'
    tuner_type: str = 'lora'
    adapters: List[str] = field(default_factory=list)

    # === Freeze Parameters ===
    freeze_parameters: List[str] = field(default_factory=list)
    freeze_parameters_regex: Optional[str] = None
    freeze_parameters_ratio: float = 0.0
    trainable_parameters: List[str] = field(default_factory=list)
    trainable_parameters_regex: Optional[str] = None
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
    lora_dtype: Literal['float16', 'bfloat16', 'float32', None] = None
    use_rslora: bool = False
    use_dora: bool = False
    # Lora: 'true'/'false'/'gaussian'/'pissa'/'pissa_niter_[number of iters]'/'olora'/'lora-ga'
    # ('loftq' also exists upstream but needs a loftq_config, which is not modelled here.)
    init_weights: str = 'true'

    # === LoRA+ ===
    # Scales lora_B's lr to lr * lorap_lr_ratio. None disables LoRA+ (plain LoRA lr for every group).
    # Requires --optimizer lorap to take effect; see swift/optimizers/lorap.py.
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

    # === GaLore ===
    use_galore: bool = False
    galore_target_modules: Optional[List[str]] = None
    galore_rank: int = 128
    galore_update_proj_gap: int = 50
    galore_scale: float = 1.0
    galore_proj_type: str = 'std'
    galore_optim_per_parameter: bool = False
    galore_with_embedding: bool = False
    galore_quantization: bool = False
    galore_proj_quant: bool = False
    galore_proj_bits: int = 4
    galore_proj_group_size: int = 256
    galore_cos_threshold: float = 0.4
    galore_gamma_proj: int = 2
    galore_queue_size: int = 5
