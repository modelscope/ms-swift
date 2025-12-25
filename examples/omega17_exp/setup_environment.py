#!/usr/bin/env python3
"""
Omega17Exp Environment Setup Script

This script permanently fixes all compatibility issues between:
- transformers-usf-om-vl-exp-v0 (custom fork)
- ms-swift
- peft
- huggingface-hub

Run this ONCE after installing dependencies to patch all issues.

Usage:
    python setup_environment.py

What it fixes:
1. BACKENDS_MAPPING missing 'tf' backend in transformers
2. transformers.deepspeed module missing (peft compatibility)
3. ROPE_INIT_FUNCTIONS missing 'default' key
4. swift train_args.py logging_dir AttributeError
5. omega17_exp model registration in swift
6. @check_model_inputs decorator causing TypeError during training
7. bitsandbytes installation for 4-bit quantization (QLoRA)
8. Omega17Exp native registration in transformers (uses optimized Qwen3Moe)
"""

import os
import sys
import subprocess
import site


def get_site_packages():
    """Get the site-packages directory."""
    for p in site.getsitepackages():
        if 'site-packages' in p or 'dist-packages' in p:
            return p
    return site.getsitepackages()[0]


def patch_backends_mapping(transformers_path):
    """Fix BACKENDS_MAPPING missing backends (tf, keras_nlp, etc.)."""
    import_utils_path = os.path.join(transformers_path, 'utils', 'import_utils.py')
    
    if not os.path.exists(import_utils_path):
        print(f"   ⚠️  import_utils.py not found at {import_utils_path}")
        return False
    
    with open(import_utils_path, 'r') as f:
        content = f.read()
    
    modified = False
    
    # All backends that might be missing - add them all at once
    all_backends = [
        ('tf', 'is_tf_available', 'TF_IMPORT_ERROR', 'TensorFlow'),
        ('tensorflow_text', 'is_tensorflow_text_available', 'TENSORFLOW_TEXT_IMPORT_ERROR', 'tensorflow_text'),
        ('tensorflow_probability', 'is_tensorflow_probability_available', 'TENSORFLOW_PROBABILITY_IMPORT_ERROR', 'tensorflow_probability'),
        ('keras_nlp', 'is_keras_nlp_available', 'KERAS_NLP_IMPORT_ERROR', 'keras_nlp'),
        ('keras', 'is_keras_available', 'KERAS_IMPORT_ERROR', 'keras'),
        ('flax', 'is_flax_available', 'FLAX_IMPORT_ERROR', 'flax'),
        ('jax', 'is_jax_available', 'JAX_IMPORT_ERROR', 'jax'),
    ]
    
    # Build dummy functions for all missing backends
    functions_to_add = []
    for backend_name, func_name, error_name, desc in all_backends:
        if f'def {func_name}()' not in content:
            functions_to_add.append(f'''
def {func_name}():
    """{desc} availability check - always returns False."""
    return False

{error_name} = "{desc} is not installed."
''')
    
    if functions_to_add:
        all_functions = '\n# Patched by setup_environment.py for Omega17Exp compatibility' + ''.join(functions_to_add)
        
        if 'BACKENDS_MAPPING = OrderedDict(' in content:
            content = content.replace(
                'BACKENDS_MAPPING = OrderedDict(',
                all_functions + '\nBACKENDS_MAPPING = OrderedDict('
            )
            modified = True
    
    # Add all backends to mapping if not present
    for backend_name, func_name, error_name, _ in all_backends:
        if f'("{backend_name}"' not in content:
            # Find a good place to insert
            if '("av", (is_av_available, AV_IMPORT_ERROR))' in content:
                content = content.replace(
                    '("av", (is_av_available, AV_IMPORT_ERROR))',
                    f'("{backend_name}", ({func_name}, {error_name})),\n        ("av", (is_av_available, AV_IMPORT_ERROR))'
                )
                modified = True
            elif 'BACKENDS_MAPPING = OrderedDict(\n    [' in content:
                content = content.replace(
                    'BACKENDS_MAPPING = OrderedDict(\n    [',
                    f'BACKENDS_MAPPING = OrderedDict(\n    [("{backend_name}", ({func_name}, {error_name})),\n        '
                )
                modified = True
    
    if modified:
        with open(import_utils_path, 'w') as f:
            f.write(content)
        print("   ✅ BACKENDS_MAPPING patched (tf, tensorflow_text, keras_nlp, keras, flax, jax)")
        return True
    else:
        print("   ✅ BACKENDS_MAPPING already patched")
        return True


def patch_deepspeed_module(transformers_path):
    """Fix transformers.deepspeed module missing for peft compatibility."""
    deepspeed_path = os.path.join(transformers_path, 'deepspeed.py')
    
    if os.path.exists(deepspeed_path):
        print("   ✅ deepspeed module already exists")
        return True
    
    # Create a minimal deepspeed compatibility module
    deepspeed_content = '''"""
Minimal deepspeed compatibility module for peft.
Patched by setup_environment.py for Omega17Exp compatibility.
"""

def deepspeed_config():
    """Return None as deepspeed is not configured."""
    return None

def is_deepspeed_zero3_enabled():
    """Check if DeepSpeed ZeRO-3 is enabled."""
    return False

def set_hf_deepspeed_config(hf_ds_config):
    """Set HuggingFace DeepSpeed config - no-op."""
    pass

# For backwards compatibility
HfDeepSpeedConfig = None
'''
    
    with open(deepspeed_path, 'w') as f:
        f.write(deepspeed_content)
    
    print("   ✅ deepspeed module created successfully")
    return True


def patch_rope_init_functions(transformers_path):
    """Add 'default' key to ROPE_INIT_FUNCTIONS in transformers with proper function."""
    rope_utils_path = os.path.join(transformers_path, 'modeling_rope_utils.py')
    
    if not os.path.exists(rope_utils_path):
        print(f"   ⚠️  modeling_rope_utils.py not found: {rope_utils_path}")
        return False
    
    with open(rope_utils_path, 'r') as f:
        content = f.read()
    
    # Check if proper default function already exists
    if 'def _compute_default_rope_parameters' in content and '"default": _compute_default_rope_parameters' in content:
        print("   ✅ ROPE_INIT_FUNCTIONS already has proper 'default' function")
        return True
    
    # Add proper _compute_default_rope_parameters function
    default_func = '''
def _compute_default_rope_parameters(config, device, seq_len=None, **rope_kwargs):
    """Default RoPE parameters - no scaling, just standard rotary embeddings."""
    base = config.rope_theta if hasattr(config, 'rope_theta') else 10000.0
    partial_rotary_factor = getattr(config, 'partial_rotary_factor', 1.0)
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    
    import torch
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, 1.0  # attention_scaling = 1.0

'''
    
    # Add function if not exists
    if 'def _compute_default_rope_parameters' not in content:
        content = content.replace('ROPE_INIT_FUNCTIONS = {', default_func + 'ROPE_INIT_FUNCTIONS = {')
        print("   ✅ Added _compute_default_rope_parameters function")
    
    # Add 'default' key to dict
    old_line = 'ROPE_INIT_FUNCTIONS = {\n    "linear"'
    new_line = 'ROPE_INIT_FUNCTIONS = {\n    "default": _compute_default_rope_parameters,\n    "linear"'
    
    if old_line in content and '"default"' not in content:
        content = content.replace(old_line, new_line)
        print("   ✅ Added 'default' key to ROPE_INIT_FUNCTIONS")
    
    with open(rope_utils_path, 'w') as f:
        f.write(content)
    
    print("   ✅ ROPE_INIT_FUNCTIONS patched successfully")
    return True


def patch_omega17_native_transformers(transformers_path):
    """Register Omega17Exp as native model in transformers using Qwen3Moe classes.
    
    This makes Omega17Exp use the optimized native Qwen3Moe implementation instead of
    trust_remote_code Python code, resulting in ~2x faster training.
    
    Creates:
    - transformers/models/omega17_exp/__init__.py
    - transformers/models/omega17_exp/configuration_omega17_exp.py
    - transformers/models/omega17_exp/modeling_omega17_exp.py
    """
    omega17_dir = os.path.join(transformers_path, 'models', 'omega17_exp')
    os.makedirs(omega17_dir, exist_ok=True)
    
    # Create __init__.py
    init_content = '''"""Omega17Exp model - optimized using Qwen3Moe native implementation."""
from .configuration_omega17_exp import Omega17ExpConfig
from .modeling_omega17_exp import (
    Omega17ExpForCausalLM,
    Omega17ExpModel,
    Omega17ExpPreTrainedModel,
)

__all__ = [
    "Omega17ExpConfig",
    "Omega17ExpForCausalLM",
    "Omega17ExpModel",
    "Omega17ExpPreTrainedModel",
]
'''
    
    # Create configuration_omega17_exp.py
    config_content = '''"""Omega17Exp configuration - inherits from Qwen3MoeConfig."""
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


class Omega17ExpConfig(Qwen3MoeConfig):
    """Configuration for Omega17Exp model.
    
    Omega17Exp is architecturally identical to Qwen3Moe, so we inherit
    all configuration from Qwen3MoeConfig.
    """
    model_type = "omega17_exp"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
'''
    
    # Create modeling_omega17_exp.py
    modeling_content = '''"""Omega17Exp model - uses optimized Qwen3Moe implementation.

Omega17Exp is architecturally identical to Qwen3Moe (same layers, experts, dimensions).
By inheriting from Qwen3Moe classes, we get the native optimized implementation
instead of the slower trust_remote_code Python version.
"""
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoePreTrainedModel,
    Qwen3MoeDecoderLayer,
    Qwen3MoeAttention,
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeMLP,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
)
from .configuration_omega17_exp import Omega17ExpConfig


# Create Omega17Exp classes as aliases of Qwen3Moe classes
class Omega17ExpPreTrainedModel(Qwen3MoePreTrainedModel):
    """Omega17Exp base class."""
    config_class = Omega17ExpConfig
    base_model_prefix = "model"
    _no_split_modules = ["Omega17ExpDecoderLayer"]


class Omega17ExpModel(Qwen3MoeModel):
    """Omega17Exp model (transformer only, no LM head)."""
    config_class = Omega17ExpConfig


class Omega17ExpForCausalLM(Qwen3MoeForCausalLM):
    """Omega17Exp for causal language modeling."""
    config_class = Omega17ExpConfig
    _tied_weights_keys = ["lm_head.weight"]


# Alias the internal classes for compatibility
Omega17ExpDecoderLayer = Qwen3MoeDecoderLayer
Omega17ExpAttention = Qwen3MoeAttention
Omega17ExpSparseMoeBlock = Qwen3MoeSparseMoeBlock
Omega17ExpMLP = Qwen3MoeMLP
Omega17ExpRMSNorm = Qwen3MoeRMSNorm
Omega17ExpRotaryEmbedding = Qwen3MoeRotaryEmbedding
'''
    
    # Write files
    with open(os.path.join(omega17_dir, '__init__.py'), 'w') as f:
        f.write(init_content)
    print(f"   ✅ Created {omega17_dir}/__init__.py")
    
    with open(os.path.join(omega17_dir, 'configuration_omega17_exp.py'), 'w') as f:
        f.write(config_content)
    print(f"   ✅ Created configuration_omega17_exp.py")
    
    with open(os.path.join(omega17_dir, 'modeling_omega17_exp.py'), 'w') as f:
        f.write(modeling_content)
    print(f"   ✅ Created modeling_omega17_exp.py (uses Qwen3Moe native code)")
    
    # Register in transformers AUTO_MAPPING
    auto_init_path = os.path.join(transformers_path, 'models', 'auto', 'modeling_auto.py')
    if os.path.exists(auto_init_path):
        with open(auto_init_path, 'r') as f:
            content = f.read()
        
        # Add to MODEL_FOR_CAUSAL_LM_MAPPING_NAMES if not present
        if 'Omega17ExpForCausalLM' not in content:
            # Find the mapping and add omega17_exp
            if 'MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict' in content:
                old_pattern = 'MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(\n    ['
                new_pattern = 'MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(\n    [("omega17_exp", "Omega17ExpForCausalLM"),\n     '
                if old_pattern in content:
                    content = content.replace(old_pattern, new_pattern)
                    with open(auto_init_path, 'w') as f:
                        f.write(content)
                    print("   ✅ Registered Omega17ExpForCausalLM in AUTO_MAPPING")
    
    # Register config in configuration_auto.py
    config_auto_path = os.path.join(transformers_path, 'models', 'auto', 'configuration_auto.py')
    if os.path.exists(config_auto_path):
        with open(config_auto_path, 'r') as f:
            content = f.read()
        
        if 'omega17_exp' not in content:
            # Add to CONFIG_MAPPING_NAMES
            if 'CONFIG_MAPPING_NAMES = OrderedDict' in content:
                old_pattern = 'CONFIG_MAPPING_NAMES = OrderedDict(\n    ['
                new_pattern = 'CONFIG_MAPPING_NAMES = OrderedDict(\n    [("omega17_exp", "Omega17ExpConfig"),\n     '
                if old_pattern in content:
                    content = content.replace(old_pattern, new_pattern)
                    with open(config_auto_path, 'w') as f:
                        f.write(content)
                    print("   ✅ Registered Omega17ExpConfig in CONFIG_MAPPING")
    
    print("   ✅ Omega17Exp native registration complete")
    return True


def install_bitsandbytes():
    """Install or upgrade bitsandbytes for 4-bit quantization (QLoRA)."""
    try:
        import bitsandbytes
        from packaging import version
        if version.parse(bitsandbytes.__version__) >= version.parse('0.46.1'):
            print(f"   ✅ bitsandbytes {bitsandbytes.__version__} already installed")
            return True
        else:
            print(f"   ⚠️  bitsandbytes {bitsandbytes.__version__} is outdated, upgrading...")
    except ImportError:
        print("   ⚠️  bitsandbytes not installed, installing...")
    except Exception:
        pass
    
    # Install bitsandbytes
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-U', 'bitsandbytes>=0.46.1'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("   ✅ bitsandbytes installed successfully")
        return True
    else:
        print(f"   ❌ Failed to install bitsandbytes: {result.stderr}")
        return False


def patch_check_model_inputs(model_dir=None):
    """Remove @check_model_inputs decorator from model files.
    
    The @check_model_inputs decorator in transformers-usf-om-vl-exp-v0 causes
    TypeError during training: 'got an unexpected keyword argument input_ids'
    
    This patches both local model directory and HuggingFace cache.
    """
    import shutil
    import glob
    
    patched = False
    locations = []
    
    # Add local model directory if provided
    if model_dir and os.path.exists(model_dir):
        locations.append(model_dir)
    
    # Add common local paths
    for local_path in ['./model', '../model', '/workspace/finetune/model']:
        if os.path.exists(local_path) and local_path not in locations:
            locations.append(local_path)
    
    # Add HuggingFace cache locations
    hf_cache = os.path.expanduser('~/.cache/huggingface/modules/transformers_modules')
    if os.path.exists(hf_cache):
        for subdir in glob.glob(os.path.join(hf_cache, '*')):
            if os.path.isdir(subdir):
                locations.append(subdir)
    
    for location in locations:
        model_file = os.path.join(location, 'modeling_omega17_exp.py')
        if not os.path.exists(model_file):
            continue
        
        with open(model_file, 'r') as f:
            content = f.read()
        
        if '@check_model_inputs' not in content:
            continue
        
        # Remove decorator and import
        import re
        new_content = re.sub(r'\s*@check_model_inputs\s*\n', '\n', content)
        new_content = new_content.replace(', check_model_inputs', '')
        new_content = new_content.replace('check_model_inputs, ', '')
        
        with open(model_file, 'w') as f:
            f.write(new_content)
        
        print(f"   ✅ Removed @check_model_inputs from {model_file}")
        patched = True
        
        # Clear __pycache__
        cache_dir = os.path.join(location, '__pycache__')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    
    if not patched:
        print("   ⚠️  No model files found with @check_model_inputs decorator")
    
    return True


def patch_swift_train_args(site_packages):
    """Fix logging_dir AttributeError in swift train_args.py."""
    train_args_path = os.path.join(site_packages, 'swift', 'llm', 'argument', 'train_args.py')
    
    if not os.path.exists(train_args_path):
        print(f"   ⚠️  train_args.py not found: {train_args_path}")
        return False
    
    with open(train_args_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'getattr(self, "logging_dir", None)' in content:
        print("   ✅ train_args.py already patched")
        return True
    
    # Patch the logging_dir check
    if 'if self.logging_dir is None:' in content:
        content = content.replace(
            'if self.logging_dir is None:',
            'if getattr(self, "logging_dir", None) is None:'
        )
        with open(train_args_path, 'w') as f:
            f.write(content)
        print("   ✅ Patched train_args.py (logging_dir fix)")
        return True
    else:
        print("   ⚠️  Could not find logging_dir check to patch")
        return False


def patch_swift_model_registration(site_packages):
    """Add omega17_exp model registration to swift installation."""
    swift_model_path = os.path.join(site_packages, 'swift', 'llm', 'model', 'model')
    
    if not os.path.exists(swift_model_path):
        print(f"   ⚠️  Swift model path not found: {swift_model_path}")
        return False
    
    omega17_path = os.path.join(swift_model_path, 'omega17.py')
    
    # Check if already patched
    if os.path.exists(omega17_path):
        print("   ✅ omega17.py already exists in swift")
        return True
    
    # Create omega17.py module
    omega17_code = '''"""
Omega17Exp Model Registration for MS-SWIFT
Auto-generated by setup_environment.py
"""
import os
from typing import Any, Dict
from transformers import AutoTokenizer
from swift.llm import TemplateType
from ..constant import LLMModelType
from ..model_arch import ModelArch, ModelKeys, register_model_arch
from ..patcher import patch_output_to_input_device
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo


def get_model_tokenizer_omega17_exp(model_dir, model_info, model_kwargs, load_model=True, **kwargs):
    """Custom get_model_tokenizer function for Omega17Exp MoE model."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tokenizer = kwargs.get("tokenizer")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, token=hf_token)
    kwargs["tokenizer"] = tokenizer
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if model is not None:
        try:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                mlp_cls = model.model.layers[0].mlp.__class__
                for module in model.modules():
                    if isinstance(module, mlp_cls):
                        patch_output_to_input_device(module)
        except (AttributeError, IndexError, TypeError):
            pass
    return model, tokenizer


# Register model architecture
if not hasattr(ModelArch, "omega17_exp"):
    ModelArch.omega17_exp = "omega17_exp"

register_model_arch(
    ModelKeys(
        "omega17_exp",
        module_list="model.layers",
        mlp="model.layers.{}.mlp",
        down_proj="model.layers.{}.mlp.down_proj",
        attention="model.layers.{}.self_attn",
        o_proj="model.layers.{}.self_attn.o_proj",
        q_proj="model.layers.{}.self_attn.q_proj",
        k_proj="model.layers.{}.self_attn.k_proj",
        v_proj="model.layers.{}.self_attn.v_proj",
        embedding="model.embed_tokens",
        lm_head="lm_head",
    ),
    exist_ok=True
)

# Register model type
if not hasattr(LLMModelType, "omega17_exp"):
    LLMModelType.omega17_exp = "omega17_exp"

register_model(
    ModelMeta(
        LLMModelType.omega17_exp,
        [ModelGroup([])],
        TemplateType.chatml,
        get_model_tokenizer_omega17_exp,
        architectures=["Omega17ExpForCausalLM"],
        model_arch=ModelArch.omega17_exp,
        additional_saved_files=[
            "tokenization_omega17.py",
            "configuration_omega17_exp.py",
            "modeling_omega17_exp.py",
        ],
        requires=["transformers-usf-om-vl-exp-v0"],
    )
)
'''
    
    with open(omega17_path, 'w') as f:
        f.write(omega17_code)
    print("   ✅ Created omega17.py in swift")
    
    # Update __init__.py to import omega17
    init_path = os.path.join(swift_model_path, '__init__.py')
    if os.path.exists(init_path):
        with open(init_path, 'r') as f:
            content = f.read()
        
        if 'omega17' not in content:
            content = content.rstrip() + '\nfrom . import omega17\n'
            with open(init_path, 'w') as f:
                f.write(content)
            print("   ✅ Updated swift __init__.py to import omega17")
        else:
            print("   ✅ omega17 already in swift __init__.py")
    
    return True


def verify_imports():
    """Verify that all imports work correctly."""
    print("\n🔍 Verifying imports...")
    
    errors = []
    
    # Test transformers
    try:
        from transformers import AutoTokenizer, AutoConfig
        print("   ✅ transformers imports OK")
    except Exception as e:
        errors.append(f"transformers: {e}")
        print(f"   ❌ transformers: {e}")
    
    # Test peft
    try:
        from peft import LoraConfig, get_peft_model
        print("   ✅ peft imports OK")
    except Exception as e:
        errors.append(f"peft: {e}")
        print(f"   ❌ peft: {e}")
    
    # Test swift
    try:
        from swift.llm import TemplateType
        print("   ✅ swift imports OK")
    except Exception as e:
        errors.append(f"swift: {e}")
        print(f"   ❌ swift: {e}")
    
    # Test omega17_exp registration
    try:
        from swift.llm.model.register import MODEL_MAPPING
        if 'omega17_exp' in MODEL_MAPPING:
            print("   ✅ omega17_exp model registered in swift")
        else:
            errors.append("omega17_exp not in MODEL_MAPPING")
            print("   ❌ omega17_exp not registered")
    except Exception as e:
        errors.append(f"omega17_exp check: {e}")
        print(f"   ❌ omega17_exp check: {e}")
    
    return len(errors) == 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Omega17Exp Environment Setup - Patches all compatibility issues'
    )
    parser.add_argument(
        '--model_dir', '-m',
        type=str,
        default=None,
        help='Path to model directory (for patching @check_model_inputs). Can be absolute or relative path.'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("OMEGA17EXP ENVIRONMENT SETUP")
    print("=" * 60)
    
    if args.model_dir:
        print(f"\n📁 Model directory: {args.model_dir}")
    
    site_packages = get_site_packages()
    transformers_path = os.path.join(site_packages, 'transformers')
    
    print(f"📁 Site packages: {site_packages}")
    print(f"📁 Transformers: {transformers_path}")
    
    if not os.path.exists(transformers_path):
        print("\n❌ ERROR: transformers not installed!")
        print("   Run: pip install transformers-usf-om-vl-exp-v0")
        sys.exit(1)
    
    print("\n🔧 Applying patches...")
    
    # Patch 1: BACKENDS_MAPPING
    print("\n1. Patching BACKENDS_MAPPING (tf backend)...")
    patch_backends_mapping(transformers_path)
    
    # Patch 2: deepspeed module
    print("\n2. Patching deepspeed module (peft compatibility)...")
    patch_deepspeed_module(transformers_path)
    
    # Patch 3: ROPE_INIT_FUNCTIONS default key
    print("\n3. Patching ROPE_INIT_FUNCTIONS (add 'default' key)...")
    patch_rope_init_functions(transformers_path)
    
    # Patch 4: Fix swift train_args.py logging_dir bug
    print("\n4. Patching swift train_args.py (logging_dir fix)...")
    patch_swift_train_args(site_packages)
    
    # Patch 5: Register omega17_exp in swift
    print("\n5. Registering omega17_exp model in swift...")
    patch_swift_model_registration(site_packages)
    
    # Patch 6: Remove @check_model_inputs decorator from model files
    print("\n6. Removing @check_model_inputs decorator from model files...")
    patch_check_model_inputs(model_dir=args.model_dir)
    
    # Patch 7: Install bitsandbytes for QLoRA
    print("\n7. Installing bitsandbytes for 4-bit quantization (QLoRA)...")
    install_bitsandbytes()
    
    # Patch 8: Register Omega17Exp natively in transformers
    print("\n8. Registering Omega17Exp natively in transformers (optimized)...")
    patch_omega17_native_transformers(transformers_path)
    
    # Verify
    success = verify_imports()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ SETUP COMPLETE! Environment is ready.")
        print("\nYou can now run:")
        print("   swift sft --model ./model --model_type omega17_exp --dataset alpaca-en --train_type lora --lora_rank 64 --output_dir ./output --gradient_checkpointing true")
        print("\n⚠️  NOTE: Do NOT use --dtype, use --torch_dtype if needed (or omit to auto-detect)")
    else:
        print("⚠️  SETUP COMPLETED WITH WARNINGS")
        print("   Some imports failed. Check errors above.")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
