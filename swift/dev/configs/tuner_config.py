"""Parameter-efficient fine-tuning configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class TunerConfig:
    """LoRA, AdaLoRA, VeRA, BOFT, FourierFT, ReFT, LLaMAPro, and Adapter settings."""

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
    lorap_lr_ratio: Optional[float] = None
    use_rslora: bool = False
    use_dora: bool = False
    init_weights: str = 'true'

    # === LoRA-GA ===
    lora_ga_batch_size: int = 2
    lora_ga_iters: int = 2
    lora_ga_max_length: int = 1024
    lora_ga_direction: str = 'ArB2r'
    lora_ga_scale: str = 'stable'
    lora_ga_stable_gamma: int = 16

    # === FourierFT ===
    fourier_n_frequency: int = 2000
    fourier_scaling: float = 300.0

    # === BOFT ===
    boft_block_size: int = 4
    boft_block_num: int = 0
    boft_dropout: float = 0.0

    # === VeRA ===
    vera_rank: int = 256
    vera_projection_prng_key: int = 0
    vera_dropout: float = 0.0
    vera_d_initial: float = 0.1

    # === AdaLoRA ===
    adalora_target_r: int = 8
    adalora_init_r: int = 12
    adalora_tinit: int = 0
    adalora_tfinal: int = 0
    adalora_deltaT: int = 1
    adalora_beta1: float = 0.85
    adalora_beta2: float = 0.85
    adalora_orth_reg_weight: float = 0.5

    # === LLaMAPro ===
    llamapro_num_new_blocks: int = 4
    llamapro_num_groups: Optional[int] = None

    # === ReFT ===
    reft_layers: Optional[List[int]] = None
    reft_rank: int = 4
    reft_intervention_type: Literal['NoreftIntervention', 'LoreftIntervention', 'ConsreftIntervention',
                                    'LobireftIntervention', 'DireftIntervention',
                                    'NodireftIntervention'] = 'LoreftIntervention'
    reft_args: Optional[str] = None

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

    # === LISA ===
    lisa_activated_layers: int = 0
    lisa_step_interval: int = 20
