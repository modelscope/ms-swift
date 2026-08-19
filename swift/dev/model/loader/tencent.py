# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the Tencent Hunyuan multimodal families.

Ported from ``swift/model/models/tencent.py`` -- ``hunyuan_ocr`` (HunyuanOCR). The Hunyuan *text*
checkpoints (``HunYuanMoEV1ForCausalLM`` / ``HunYuanDenseV1ForCausalLM``) live in the generic
``llm.py`` grab-bag, not here.
"""
from __future__ import annotations

from .base import ModelArch, ModelLoader, register_model


@register_model
class HunyuanOcrLoader(ModelLoader):
    """HunyuanOCR vision-language. Defaults the attention kernel to ``eager`` when the user did not
    pass ``--attn_impl`` (legacy set this in ``get_config``). A user choice still wins: dev only puts
    ``attn_implementation`` in the load kwargs when it was explicitly requested, so ``setdefault``
    never overrides it."""

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
