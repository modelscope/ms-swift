# Copyright (c) ModelScope Contributors. All rights reserved.
# LFM3 runs its 13 short-conv layers through plain F.conv1d (the hub causal_conv1d kernels
# only cover widths 2-4), so it trains without any extra kernels and needs no custom loader.
from swift.template import TemplateType
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import MLLMModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model

register_model(
    ModelMeta(
        LLMModelType.lfm3,
        [
            ModelGroup([
                Model('LiquidAI/LFM3-1.7B', 'LiquidAI/LFM3-1.7B'),
                Model('LiquidAI/LFM3-1.7B-Base', 'LiquidAI/LFM3-1.7B-Base'),
            ])
        ],
        ModelLoader,
        template=TemplateType.lfm3,
        architectures=['Lfm3ForCausalLM'],
        requires=['transformers>=5.23'],
        tags=['text'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.lfm3_vl,
        [ModelGroup([
            Model('LiquidAI/LFM3-VL-2B', 'LiquidAI/LFM3-VL-2B'),
        ])],
        ModelLoader,
        template=TemplateType.lfm3_vl,
        model_arch=MLLMModelArch.lfm3_vl,
        architectures=['Lfm3VlForConditionalGeneration'],
        requires=['transformers>=5.23'],
        tags=['vision'],
    ))
