"""
Omega17Exp Model Registration for MS-SWIFT

This module registers the Omega17Exp model with MS-SWIFT at runtime.
Works with pip-installed MS-SWIFT - no source code modification needed.

Usage:
    # Import this BEFORE using MS-SWIFT
    import register_omega17
    
    # Then use MS-SWIFT normally
    from swift.llm import sft_main, SftArguments
    ...

Or run directly to verify registration:
    python register_omega17.py
"""

import os
# Disable TensorFlow to avoid backend conflict with transformers
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import Any, Dict

from transformers import AutoTokenizer

from swift.llm import TemplateType
from swift.llm.model.constant import LLMModelType
from swift.llm.model.model_arch import ModelArch, ModelKeys, register_model_arch
from swift.llm.model.patcher import patch_output_to_input_device
from swift.llm.model.register import (
    Model, ModelGroup, ModelMeta,
    get_model_tokenizer_with_flash_attn,
    register_model, MODEL_MAPPING
)
from swift.llm.model.utils import ModelInfo


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
    
    For private models, set HF_TOKEN environment variable.
    """
    import os
    
    # Get HuggingFace token for private models
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    # Load custom tokenizer with trust_remote_code
    tokenizer = kwargs.get('tokenizer')
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, 
            trust_remote_code=True,
            token=hf_token
        )
    kwargs['tokenizer'] = tokenizer
    
    # Load model with flash attention support
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, **kwargs
    )
    
    if model is not None:
        # Apply MoE-specific patches for proper device placement
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                mlp_cls = model.model.layers[0].mlp.__class__
                for module in model.modules():
                    if isinstance(module, mlp_cls):
                        patch_output_to_input_device(module)
        except (AttributeError, IndexError, TypeError):
            pass
    
    return model, tokenizer


def register_omega17_model():
    """Register the Omega17Exp model with MS-SWIFT."""
    
    # Skip if already registered
    if 'omega17_exp' in MODEL_MAPPING:
        print("✅ Omega17Exp model already registered")
        return True
    
    # Add custom model type to LLMModelType
    if not hasattr(LLMModelType, 'omega17_exp'):
        LLMModelType.omega17_exp = 'omega17_exp'
    
    # Add custom model architecture to ModelArch
    if not hasattr(ModelArch, 'omega17_exp'):
        ModelArch.omega17_exp = 'omega17_exp'
    
    # Register model architecture with LoRA target modules
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
    
    # Register the model
    register_model(
        ModelMeta(
            LLMModelType.omega17_exp,
            [
                ModelGroup([]),
            ],
            TemplateType.chatml,
            get_model_tokenizer_omega17_exp,
            architectures=['Omega17ExpForCausalLM'],
            model_arch=ModelArch.omega17_exp,
            additional_saved_files=[
                'tokenization_omega17.py',
                'configuration_omega17_exp.py',
                'modeling_omega17_exp.py',
            ],
            requires=['transformers-usf-om-vl-exp-v0'],
        )
    )
    
    print("✅ Omega17Exp model registered successfully!")
    return True


# Auto-register when this module is imported
_registered = register_omega17_model()


if __name__ == "__main__":
    # Verify registration
    print("\n" + "=" * 50)
    print("OMEGA17EXP MODEL REGISTRATION")
    print("=" * 50)
    
    if 'omega17_exp' in MODEL_MAPPING:
        meta = MODEL_MAPPING['omega17_exp']
        print(f"✅ Model Type: omega17_exp")
        print(f"   Template: {meta.template}")
        print(f"   Architectures: {meta.architectures}")
        print(f"   Model Arch: {meta.model_arch}")
        print(f"   Additional Files: {meta.additional_saved_files}")
        print(f"   Requires: {meta.requires}")
    else:
        print("❌ Registration failed!")
    
    print("=" * 50)
