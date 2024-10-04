from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class TunerArguments:
    """This dataclass manages the training types"""
    tuner_backend: Literal['swift', 'peft', 'unsloth'] = 'peft'
    sft_type: str = 'lora'

    # tuners
    target_modules: List[str] = field(default_factory=lambda: ['ALL'])
    target_regex: Optional[str] = None
    # e.g. ['wte', 'ln_1', 'ln_2', 'ln_f', 'lm_head']
    modules_to_save: List[str] = field(default_factory=list)

    # lora
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias_trainable: Literal['none', 'all'] = 'none'
    lora_dtype: Literal['fp16', 'bf16', 'fp32', 'AUTO'] = 'AUTO'
    lora_lr_ratio: float = None
    use_rslora: bool = False
    use_dora: bool = False
    # Literal['gaussian', 'pissa', 'pissa_niter_[number of iters]', 'olora', 'loftq', 'true', 'false']
    init_lora_weights: str = 'true'

    # fourierft
    fourier_n_frequency: int = 2000
    fourier_scaling: float = 300.0

    # BOFT
    boft_block_size: int = 4
    boft_block_num: int = 0
    boft_n_butterfly_factor: int = 1
    boft_dropout: float = 0.0

    # Vera
    vera_rank: int = 256
    vera_projection_prng_key: int = 0
    vera_dropout: float = 0.0
    vera_d_initial: float = 0.1

    # adapter
    adapter_act: str = 'gelu'
    adapter_length: int = 128

    # galore
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

    # adalora
    adalora_target_r: int = 8
    adalora_init_r: int = 12
    adalora_tinit: int = 0
    adalora_tfinal: int = 0
    adalora_deltaT: int = 1
    adalora_beta1: float = 0.85
    adalora_beta2: float = 0.85
    adalora_orth_reg_weight: float = 0.5

    # ia3
    ia3_feedforward_modules: List[str] = field(default_factory=list)

    # llamapro
    llamapro_num_new_blocks: int = 4
    llamapro_num_groups: Optional[int] = None

    # neftune
    neftune_noise_alpha: Optional[float] = None  # e.g. 5, 10, 15

    # lisa
    lisa_activated_layers: int = 0
    lisa_step_interval: int = 20

    # reft
    reft_layer_key: Optional[str] = None
    reft_layers: Optional[List[int]] = None
    reft_rank: int = 4
    reft_intervention_type: Literal['NoreftIntervention', 'LoreftIntervention', 'ConsreftIntervention',
                                    'LobireftIntervention', 'DireftIntervention',
                                    'NodireftIntervention'] = 'LoreftIntervention'
    reft_args: Optional[str] = None

    # use_liger
    use_liger: bool = False

    def is_adapter(self) -> bool:
        return self.sft_type in {
            'lora', 'longlora', 'adalora', 'ia3', 'llamapro', 'adapter', 'vera', 'boft', 'fourierft', 'reft'
        }

    def adapters_can_be_merged(self):
        return ['lora', 'longlora', 'llamapro', 'adalora']
