# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class TunerConfig:
    """LoRA (+QLoRA/DoRA/rsLoRA), AdaLoRA, and TrainableTokens settings.

    Only tuners with real-world use on text/multimodal LLMs; quantization knobs live in QuantizeConfig.
    """

    # === Base ===
    # 'lora'/'trainable_tokens', or 'full' = no adapter. Full-parameter training is spelled two
    # ways -- this value, or no TunerConfig at all -- and has_adapter() below is what tells an
    # adapter from either spelling. DoRA/rsLoRA/QLoRA are LoRA flags, not tuners.
    tuner: Literal['full', 'lora', 'trainable_tokens'] = 'lora'
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


def has_adapter(tuner_config: Optional[TunerConfig]) -> bool:
    """Whether ``tuner_config`` installs an adapter (False for ``None`` and for ``tuner='full'``).

    Full-parameter training is spelled two ways: no TunerConfig, or ``TunerConfig(tuner='full')``.
    The CLI maps the second onto the first (``select_tuner``) only after parsing, and a component-API
    caller hands the ``TunerConfig(tuner='full')`` object straight to a recipe -- so every "is there
    an adapter" decision reads this predicate, never ``tuner_config is not None``. A non-None check
    would build a LoRA for a full-parameter run on the component API while the same setting trains
    full-parameter on the CLI.
    """
    return tuner_config is not None and tuner_config.tuner != 'full'
