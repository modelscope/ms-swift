# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Circle-RoPE Model Registration for ms-swift

This module registers the Qwen2.5-VL Circle-RoPE model variant with ms-swift.

Usage:
    swift sft \\
        --custom_register_path /path/to/circle_rope/register.py \\
        --model_type qwen2_5_vl_circle_rope \\
        --template_type qwen2_5_vl_circle_rope

The template handles AGE mode automatically based on config.circle_rope settings.
"""
from transformers import AutoConfig

from swift.llm import register_template
from swift.llm.model import Model, ModelGroup, ModelMeta, register_model
from swift.llm.model.model.qwen import get_model_tokenizer_qwen2_5_vl
from swift.llm.model.model_arch import ModelArch
from swift.llm.template.template.qwen import QwenTemplateMeta

# Import our custom template
from circle_rope.template_circle_rope import CircleRoPEQwen2_5VLTemplate


def get_model_tokenizer_qwen2_5_vl_circle_rope(model_dir: str, *args, **kwargs):
    """
    Load Qwen2.5-VL with Circle-RoPE.

    This function:
    1. Loads the original Qwen2.5-VL config
    2. Applies model_config_override from training config (includes circle_rope settings)
    3. Uses Qwen2_5_VLForConditionalGeneration_CircleRoPE as the model class
    """
    # Load original config
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    # Apply model_config_override if provided in training config
    model_config_override = kwargs.pop('model_config_override', None)
    if model_config_override:
        model_config.update(model_config_override)
        if 'architectures' in model_config_override:
            model_config.architectures = model_config_override['architectures']

    # Pass modified config
    kwargs['model_config'] = model_config

    # Import and use Circle-RoPE model class
    from circle_rope.modular_qwen2_5_vl_circle_rope import Qwen2_5_VLForConditionalGeneration_CircleRoPE
    kwargs['automodel_class'] = kwargs.get('automodel_class') or Qwen2_5_VLForConditionalGeneration_CircleRoPE

    return get_model_tokenizer_qwen2_5_vl(model_dir, *args, **kwargs)


# Register the Circle-RoPE model type
register_model(
    ModelMeta(
        'qwen2_5_vl',
        [
            ModelGroup([
                # Use any Qwen2.5-VL model path as base
                # Circle-RoPE will be applied via model_config_override in training config
                Model('qwen2_5_vl_circle_rope'),
            ])
        ],
        'qwen2_5_vl_circle_rope',  # model_id_or_path key
        get_model_tokenizer_qwen2_5_vl_circle_rope,
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen2_5_VLForConditionalGeneration_CircleRoPE'],
        requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6', 'decord'],
        tags=['vision', 'video', 'circle-rope', 'age']),
    exist_ok=True,
)

# Register the Circle-RoPE template
# This template automatically handles AGE mode based on config.circle_rope settings
register_template(
    QwenTemplateMeta(
        'qwen2_5_vl_circle_rope',  # template_type
        template_cls=CircleRoPEQwen2_5VLTemplate
    )
)
