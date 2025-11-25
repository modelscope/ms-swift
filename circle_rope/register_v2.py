# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Circle-RoPE Model Registration for ms-swift (V2 - Compatible with Latest Transformers)

This module registers the Qwen2.5-VL Circle-RoPE V2 model variant with ms-swift.
V2 is compatible with the latest transformers architecture where:
- Qwen2_5_VLModel contains both visual and language_model
- get_rope_index is in Qwen2_5_VLModel (not in ForConditionalGeneration)

Usage: swift sft --custom_register_path /path/to/circle_rope/register_v2.py
"""
from os.path import exists

from transformers import AutoConfig

from swift.llm import TemplateType
from swift.llm.model import Model, ModelGroup, ModelMeta, register_model
from swift.llm.model.model.qwen import get_model_tokenizer_qwen2_5_vl
from swift.llm.model.model_arch import ModelArch


def get_model_tokenizer_qwen2_5_vl_circle_rope_v2(model_dir: str, *args, **kwargs):
    """
    Load Qwen2.5-VL with Circle-RoPE V2.

    Loads config, then overwrites it with model_config_override from training config.
    V2 version is compatible with latest transformers architecture.
    """
    # Load original config
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    # Apply model_config_override if provided in training config
    model_config_override = kwargs.pop('model_config_override', None)
    if model_config_override:
        # Update config attributes
        for key, value in model_config_override.items():
            if key != 'architectures':  # Handle architectures separately
                setattr(model_config, key, value)

        # Set architectures if provided
        if 'architectures' in model_config_override:
            model_config.architectures = model_config_override['architectures']

    # Pass modified config
    kwargs['model_config'] = model_config

    # Import V2 implementation
    from circle_rope.modular_qwen2_5_vl_circle_rope_v2 import (
        Qwen2_5_VLForConditionalGeneration_CircleRoPE_V2
    )

    kwargs['automodel_class'] = kwargs.get('automodel_class') or Qwen2_5_VLForConditionalGeneration_CircleRoPE_V2

    return get_model_tokenizer_qwen2_5_vl(model_dir, *args, **kwargs)


# Register the Circle-RoPE V2 model type
register_model(
    ModelMeta(
        'qwen2_5_vl',
        [
            ModelGroup([
                # Placeholder - use any Qwen2.5-VL model path
                # Circle-RoPE will be applied via model_config_override in training config
                Model('qwen2_5_vl_circle_rope_v2'),
            ])
        ],
        TemplateType.qwen2_5_vl,
        get_model_tokenizer_qwen2_5_vl_circle_rope_v2,
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen2_5_VLForConditionalGeneration_CircleRoPE_V2'],
        requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6', 'decord'],
        tags=['vision', 'video', 'circle-rope', 'v2']),
    exist_ok=True
)
