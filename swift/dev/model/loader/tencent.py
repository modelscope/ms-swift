# Copyright (c) ModelScope Contributors. All rights reserved.
# Hunyuan multimodal families (hunyuan_ocr); the Hunyuan *text* checkpoints live in llm.py.
from __future__ import annotations

from .base import ModelArch, ModelLoader, register_model


@register_model
class HunyuanOcrLoader(ModelLoader):
    """HunyuanOCR vision-language. Defaults the attention kernel to ``eager`` when the user did not
    pass ``--attn_impl`` (legacy set this in ``get_config``); an explicit choice still wins."""

    model_type = 'hunyuan_ocr'
    model_cls = 'transformers:HunYuanVLForConditionalGeneration'
    architectures = ['HunYuanVLForConditionalGeneration']
    template = 'hunyuan_ocr'
    requires = ['transformers>=4.49.0']
    is_multimodal = True
    models = [('Tencent-Hunyuan/HunyuanOCR', 'tencent/HunyuanOCR')]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(language_model='model', aligner='vit.perceive', vision_tower='vit')

    def build_model(self, model_dir, config, processor, **kwargs):
        kwargs.setdefault('attn_implementation', 'eager')
        return super().build_model(model_dir, config, processor, **kwargs)
