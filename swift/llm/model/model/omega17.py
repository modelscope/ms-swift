# Omega17Exp Model Registration for MS-SWIFT
#
# Omega17Exp - Large Language Model with Mixture of Experts (MoE) architecture
#
# Model specifications:
# - 48 transformer layers
# - 128 experts (8 active per token)
# - Hidden size: 2048
# - Intermediate size: 6144 (MoE intermediate: 768)
# - Context length: 262,144 tokens
# - Vocabulary: 151,936 tokens
#
# Requires: transformers-usf-om-vl-exp-v0 (custom transformers fork with Omega17Exp)

from typing import Any, Dict

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType
from ..model_arch import ModelArch, ModelKeys, register_model_arch
from ..register import Model, ModelGroup, ModelMeta, register_model
from ..utils import ModelInfo

logger = get_logger()


def get_model_tokenizer_omega17_exp(model_dir: str,
                                    model_info: ModelInfo,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    """
    Load Omega17Exp model and tokenizer.
    
    Uses the Omega17Exp implementation from transformers-usf-om-vl-exp-v0 fork
    with optimizations for MoE training.
    
    Model specs:
    - 48 layers, 128 experts (8 active per token)
    - Hidden size: 2048, Intermediate: 6144 (MoE: 768)
    - Context: 262,144 tokens
    """
    # Load tokenizer
    tokenizer = kwargs.get('tokenizer')
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
    
    # Load model
    model = None
    if load_model:
        # Configure quantization if specified
        quantization_config = model_kwargs.get('quantization_config')
        
        # Skip modules for quantization (MoE gates should stay in full precision)
        if isinstance(quantization_config, BitsAndBytesConfig):
            quantization_config.llm_int8_skip_modules = [
                'mlp.gate',
                'mlp.shared_expert_gate',
                'lm_head'
            ]
        
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            **model_kwargs
        )
        
        logger.info(f'Omega17Exp model loaded: {model.__class__.__name__}')
        logger.info(f'Model dtype: {model_info.torch_dtype}')
        if quantization_config:
            logger.info(f'Quantization: {quantization_config.quant_method}')
    
    return model, tokenizer


# Register Omega17Exp model architecture for LoRA targeting
# Defines which modules can be targeted by LoRA adapters
register_model_arch(
    ModelKeys(
        'omega17_exp',
        module_list='model.layers',
        mlp='model.layers.{}.mlp',
        down_proj='model.layers.{}.mlp.down_proj',
        attention='model.layers.{}.self_attn',
        o_proj='model.layers.{}.self_attn.o_proj',
        q_proj='model.layers.{}.self_attn.q_proj',
        k_proj='model.layers.{}.self_attn.k_proj',
        v_proj='model.layers.{}.self_attn.v_proj',
        embedding='model.embed_tokens',
        lm_head='lm_head',
    ),
    exist_ok=True
)

# Add model architecture to ModelArch
if not hasattr(ModelArch, 'omega17_exp'):
    ModelArch.omega17_exp = 'omega17_exp'


# Register Omega17Exp model
register_model(
    ModelMeta(
        LLMModelType.omega17_exp,
        [
            ModelGroup([
                Model('arpitsh018/omega17exp-prod-v1.1', 'arpitsh018/omega17exp-prod-v1.1'),
            ]),
        ],
        # Template: ChatML format
        TemplateType.chatml,
        # Model loader function
        get_model_tokenizer_omega17_exp,
        # Architecture
        architectures=['Omega17ExpForCausalLM'],
        # Model architecture for LoRA targeting
        model_arch=ModelArch.omega17_exp,
        # Required packages
        requires=['transformers-usf-om-vl-exp-v0'],
    )
)
