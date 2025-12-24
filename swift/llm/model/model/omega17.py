# Copyright (c) Alibaba, Inc. and its affiliates.
# Custom model registration for Omega17Exp model
from typing import Any, Dict

from transformers import AutoTokenizer

from swift.llm import TemplateType
from ..constant import LLMModelType
from ..model_arch import ModelArch, ModelKeys, register_model_arch
from ..patcher import patch_output_to_input_device
from ..register import (
    Model, ModelGroup, ModelMeta,
    get_model_tokenizer_with_flash_attn,
    register_model
)
from ..utils import ModelInfo


def get_model_tokenizer_omega17_exp(model_dir: str,
                                    model_info: ModelInfo,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    """
    Custom get_model_tokenizer function for Omega17Exp MoE model.
    
    This model uses:
    - Custom transformers fork: transformers-usf-om-vl-exp-v0
    - Custom tokenizer: Omega17Tokenizer (tokenization_omega17.py)
    - Custom config: Omega17ExpConfig (configuration_omega17_exp.py)
    - Custom model: Omega17ExpForCausalLM (modeling_omega17_exp.py)
    
    Model specs:
    - 48 layers, 128 experts (8 active per token)
    - Hidden size: 2048, Intermediate: 6144 (MoE: 768)
    - Context: 262144 tokens
    """
    # Load custom tokenizer with trust_remote_code
    tokenizer = kwargs.get('tokenizer')
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['tokenizer'] = tokenizer
    
    # Load model with flash attention support
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, **kwargs
    )
    
    if model is not None:
        # Apply MoE-specific patches for proper device placement
        # This ensures expert outputs are on the correct device during forward pass
        try:
            # Get the MLP class from the model layers
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                mlp_cls = model.model.layers[0].mlp.__class__
                for module in model.modules():
                    if isinstance(module, mlp_cls):
                        patch_output_to_input_device(module)
        except (AttributeError, IndexError, TypeError):
            # Model structure may vary, skip patching if not applicable
            pass
    
    return model, tokenizer


# Register Omega17Exp model architecture for LoRA targeting
# This defines which modules can be targeted by LoRA
class Omega17ModelArch:
    omega17_exp = 'omega17_exp'


# Add to ModelArch if not present
if not hasattr(ModelArch, 'omega17_exp'):
    ModelArch.omega17_exp = 'omega17_exp'

# Register model architecture with LoRA target modules
register_model_arch(
    ModelKeys(
        Omega17ModelArch.omega17_exp,
        # Layer structure
        module_list='model.layers',
        # MLP modules
        mlp='model.layers.{}.mlp',
        down_proj='model.layers.{}.mlp.down_proj',
        # Attention modules
        attention='model.layers.{}.self_attn',
        o_proj='model.layers.{}.self_attn.o_proj',
        q_proj='model.layers.{}.self_attn.q_proj',
        k_proj='model.layers.{}.self_attn.k_proj',
        v_proj='model.layers.{}.self_attn.v_proj',
        # Embeddings
        embedding='model.embed_tokens',
        lm_head='lm_head',
    ),
    exist_ok=True
)

# Register Omega17Exp model type
if not hasattr(LLMModelType, 'omega17_exp'):
    LLMModelType.omega17_exp = 'omega17_exp'

register_model(
    ModelMeta(
        LLMModelType.omega17_exp,
        [
            ModelGroup([
                # Add your model paths here (ModelScope ID, HuggingFace ID)
                # Model('your-org/omega17-exp-base', 'your-org/omega17-exp-base'),
            ]),
        ],
        # Template: chatml format using <|im_start|> and <|im_end|> tokens
        TemplateType.chatml,
        # Custom model loader function
        get_model_tokenizer_omega17_exp,
        # Architecture identifier
        architectures=['Omega17ExpForCausalLM'],
        # Model architecture for LoRA targeting
        model_arch=ModelArch.omega17_exp,
        # Additional files to save when saving model (custom Python files)
        additional_saved_files=[
            'tokenization_omega17.py',
            'configuration_omega17_exp.py',
            'modeling_omega17_exp.py',
        ],
        # Required packages
        requires=['transformers-usf-om-vl-exp-v0'],
    )
)
