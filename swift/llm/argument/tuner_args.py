# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from transformers.utils import strtobool

from swift.llm import get_model_arch
from swift.utils import get_logger

logger = get_logger()


@dataclass
class TunerArguments:
    """
    TunerArguments is a dataclass that holds configuration for various tuners.

    Args:
        target_modules (List[str]): List of target modules for tuning. Default is ['all-linear'].
        target_regex (Optional[str]): Regular expression to match target modules. Default is None.
        modules_to_save (List[str]): List of modules to save. Default is an empty list.

        lora_rank (int): Rank for LoRA. Default is 8.
        lora_alpha (int): Alpha value for LoRA. Default is 32.
        lora_dropout (float): Dropout rate for LoRA. Default is 0.05.
            Allowed values are 'none', 'all'.
        lora_dtype (Literal): Data type for LoRA. Default is 'AUTO'. Allowed values are 'fp16', 'bf16', 'fp32', 'AUTO'.
        lorap_lr_ratio (float): Learning rate ratio for LoRA. Default is None.
        use_rslora (bool): Flag to indicate if RSLora is used. Default is False.
        use_dora (bool): Flag to indicate if Dora is used. Default is False.
        init_weights (str): Initialization method for weights of supported tuners. Default is 'true'.
        lora_ga_batch_size (int): Batch size used for estimating gradients during initialization in LoRA-GA.
                                    Default value is 2.
        lora_ga_iters (int): Number of iterations for estimating gradients during initialization in LoRA-GA.
                                Default value is 2.
        lora_ga_max_length (int): Maximum input length for estimating gradients during initialization in LoRA-GA.
                                    Default value is 1024.
        lora_ga_direction (str): Initial direction used for gradient estimation during initialization in LoRA-GA.
                                    Default value is `ArB2r`. Allowed: `ArBr`, `A2rBr`, `ArB2r`, and `random`.
        lora_ga_scale (str): The scaling method for initialization in LoRA-GA.
                                Default value is `stable`. Allowed values are: `gd`, `unit`, `stable`, and `weightS`.
        lora_ga_stable_gamma (int): The gamma value when choosing `stable` scaling for initialization.
                                    Default value is 16.

        fourier_n_frequency (int): Number of frequencies for FourierFT. Default is 2000.
        fourier_scaling (float): Scaling factor for FourierFT. Default is 300.0.

        boft_block_size (int): Block size for BOFT. Default is 4.
        boft_block_num (int): Number of blocks for BOFT. Default is 0.
        boft_n_butterfly_factor (int): Butterfly factor for BOFT. Default is 1.
        boft_dropout (float): Dropout rate for BOFT. Default is 0.0.

        vera_rank (int): Rank for Vera. Default is 256.
        vera_projection_prng_key (int): PRNG key for Vera projection. Default is 0.
        vera_dropout (float): Dropout rate for Vera. Default is 0.0.
        vera_d_initial (float): Initial value for Vera D. Default is 0.1.

        adapter_act (str): Activation function for adapter. Default is 'gelu'.
        adapter_length (int): Length of the adapter. Default is 128.

        use_galore (bool): Flag to indicate if Galore is used. Default is False.
        galore_target_modules (Optional[List[str]]): List of target modules for Galore. Default is None.
        galore_rank (int): Rank for Galore. Default is 128.
        galore_update_proj_gap (int): Update projection gap for Galore. Default is 50.
        galore_scale (float): Scaling factor for Galore. Default is 1.0.
        galore_proj_type (str): Projection type for Galore. Default is 'std'.
        galore_optim_per_parameter (bool): Flag to indicate if optimization is per parameter for Galore.
            Default is False.
        galore_with_embedding (bool): Flag to indicate if embedding is used with Galore. Default is False.
        galore_quantization (bool): Flag to indicate if use Q-Galore. Default is False.
        galore_proj_quant (bool): Flag to indicate if projection quantization is used for Galore. Default is False.
        galore_proj_bits (int): Number of bits for projection quantization. Default is 4.
        galore_proj_group_size (int): Group size for projection quantization. Default is 256.
        galore_cos_threshold (float): Cosine threshold for projection quantization. Default is 0.4.
        galore_gamma_proj (int): Gamma for projection quantization. Default is 2.
        galore_queue_size (int): Queue size for projection quantization. Default is 5.

        adalora_target_r (int): Target rank for AdaLoRA. Default is 8.
        adalora_init_r (int): Initial rank for AdaLoRA. Default is 12.
        adalora_tinit (int): Initial T value for AdaLoRA. Default is 100.
        adalora_tfinal (int): Final T value for AdaLoRA. Default is 1000.
        adalora_deltaT (int): Delta T value for AdaLoRA. Default is 10.
        adalora_beta1 (float): Beta1 value for AdaLoRA. Default is 0.85.
        adalora_beta2 (float): Beta2 value for AdaLoRA. Default is 0.85.
        adalora_orth_reg_weight (float): Orthogonal regularization weight for AdaLoRA. Default is 0.5.

        llamapro_num_new_blocks (int): Number of new blocks for LLaMAPro. Default is 4.
        llamapro_num_groups (Optional[int]): Number of groups for LLaMAPro. Default is None.

        lisa_activated_layers (int): Number of activated layers for LISA. Default is 0.
        lisa_step_interval (int): Step interval for LISA activation. Default is 20.

        reft_layer_key (Optional[str]): Key identifier for ReFT layer. Default is None.
        reft_layers (Optional[List[int]]): List of layers involved in ReFT. Default is None.
        reft_rank (int): Rank parameter for ReFT. Default is 4.
        reft_intervention_type (Literal): Type of intervention for ReFT. Default is 'LoreftIntervention'.
        reft_args (Optional[str]): Additional arguments for ReFT. Default is None.

        use_liger (bool): Flag to indicate if Liger-kernel is used. Default is False.
    """
    # full
    freeze_parameters: List[str] = field(default_factory=list)
    freeze_parameters_ratio: float = 0.  # 0 ~ 1
    trainable_parameters: List[str] = field(default_factory=list)
    # lora or full
    freeze_llm: bool = False
    freeze_vit: bool = True
    freeze_aligner: bool = True
    # tuners
    target_modules: List[str] = field(default_factory=lambda: ['all-linear'])
    target_regex: Optional[str] = None
    # e.g. ['wte', 'ln_1', 'ln_2', 'ln_f', 'lm_head']
    modules_to_save: List[str] = field(default_factory=list)

    # lora
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: Literal['none', 'all'] = 'none'
    lora_dtype: Literal['float16', 'bfloat16', 'float32', None] = None
    lorap_lr_ratio: Optional[float] = None
    use_rslora: bool = False
    use_dora: bool = False
    # Lora: Literal['gaussian', 'pissa', 'pissa_niter_[number of iters]', 'olora', 'loftq', 'true', 'false', 'lora-ga']
    lora_ga_batch_size: int = 2
    lora_ga_iters: int = 2
    lora_ga_max_length: int = 1024
    lora_ga_direction: str = 'ArB2r'
    lora_ga_scale: str = 'stable'
    lora_ga_stable_gamma: int = 16

    # Bone: Literal['bat', 'true', 'false']
    init_weights: str = 'true'

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

    # llamapro
    llamapro_num_new_blocks: int = 4
    llamapro_num_groups: Optional[int] = None

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

    def __post_init__(self):
        if isinstance(self.init_weights, str) and self.init_weights.lower() in {'true', 'false'}:
            self.init_weights = bool(strtobool(self.init_weights))
        self._init_multimodal_full()
        if self.target_regex:
            self.target_modules = self.target_regex

    def _init_multimodal_full(self):
        if not self.model_meta.is_multimodal:
            return
        model_arch = get_model_arch(self.model_meta.model_arch)
        if self.freeze_llm:
            self.freeze_parameters += model_arch.language_model
        if self.freeze_vit:
            self.freeze_parameters += model_arch.vision_tower
        if self.freeze_aligner:
            self.freeze_parameters += model_arch.aligner
        else:
            self.trainable_parameters += model_arch.aligner
        self.freeze_parameters += model_arch.generator
        if self.freeze_parameters:
            logger.info(f'freeze_parameters: {self.freeze_parameters}')
        if self.trainable_parameters:
            logger.info(f'trainable_parameters: {self.trainable_parameters}')
