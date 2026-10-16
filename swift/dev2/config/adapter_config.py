# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class TunerConfig:
    """LoRA (+QLoRA/DoRA/rsLoRA), AdaLoRA, and TrainableTokens settings.

    Only tuners with real-world use on text/multimodal LLMs; quantization knobs live in QuantizeConfig.
    """

    # === Base ===
    # 'lora'/'adalora'/'trainable_tokens', or 'full' = no adapter (a None TunerConfig; see
    # select_tuner). DoRA/rsLoRA/QLoRA are LoRA flags, not tuners.
    tuner: Literal['full', 'lora', 'adalora', 'trainable_tokens'] = 'lora'
    adapters: List[str] = field(default_factory=list)

    # === Freeze Parameters (multimodal targeting) ===
    # Which towers 'all-linear' trains (legacy: train LLM, freeze vision+aligner); inert on a plain LLM.
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
    # peft init_lora_weights: a bool or a strategy ('gaussian'/'pissa'/...); 'loftq' and 'lora-ga'
    # are rejected at build time.
    init_weights: str = 'true'

    # === LoRA+ ===
    # lora_B/embedding learning-rate ratio via builders.optimizer param-group rules; None = off.
    lorap_lr_ratio: Optional[float] = None
    lorap_emb_lr: float = 1e-6

    # === TrainableTokens ===
    # Train only the given embedding rows (the standard way to learn new special tokens).
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
