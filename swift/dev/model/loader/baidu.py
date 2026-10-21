# Copyright (c) ModelScope Contributors. All rights reserved.
# ERNIE-4.5 families + paddleocr_vl (paddle_ocr dropped: transformers<5.0, MODEL_MIGRATION.md).
from __future__ import annotations

from .base import ModelArch, ModelLoader, register_model


@register_model
class Ernie4_5Loader(ModelLoader):

    model_type = 'ernie4_5'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['Ernie4_5_ForCausalLM']
    template = 'ernie'
    models = [
        ('PaddlePaddle/ERNIE-4.5-0.3B-Base-PT', 'baidu/ERNIE-4.5-0.3B-PT'),
        ('PaddlePaddle/ERNIE-4.5-0.3B-PT', 'baidu/ERNIE-4.5-0.3B-PT'),
    ]


@register_model
class Ernie4_5MoeLoader(ModelLoader):

    model_type = 'ernie4_5_moe'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['Ernie4_5_MoeForCausalLM']
    template = 'ernie'
    models = [
        ('PaddlePaddle/ERNIE-4.5-21B-A3B-Base-PT', 'baidu/ERNIE-4.5-21B-A3B-Base-PT'),
        ('PaddlePaddle/ERNIE-4.5-21B-A3B-PT', 'baidu/ERNIE-4.5-21B-A3B-PT'),
        ('PaddlePaddle/ERNIE-4.5-300B-A47B-Base-PT', 'baidu/ERNIE-4.5-300B-A47B-Base-PT'),
        ('PaddlePaddle/ERNIE-4.5-300B-A47B-PT', 'baidu/ERNIE-4.5-300B-A47B-PT'),
    ]


@register_model
class Ernie4_5MoeThinkingLoader(Ernie4_5MoeLoader):
    """The ERNIE-4.5 MoE Thinking checkpoint, which uses the ``ernie_thinking`` chat format."""

    model_type = 'ernie4_5_moe_thinking'
    architectures = []
    template = 'ernie_thinking'
    models = [('PaddlePaddle/ERNIE-4.5-21B-A3B-Thinking', 'baidu/ERNIE-4.5-21B-A3B-Thinking')]


@register_model
class PaddleOCR1_5Loader(ModelLoader):
    """PaddleOCR-VL-1.5/1.6 (transformers-native, ``transformers>=5.0``). Loads via the generic
    ``AutoModelForImageTextToText``; no ``architectures`` reverse-lookup entry (legacy declared none,
    resolution is by checkpoint id)."""

    model_type = 'paddleocr_vl'
    model_cls = 'transformers:AutoModelForImageTextToText'
    template = 'paddle_ocr_1_5'
    requires = ['transformers>=5.0']
    is_multimodal = True
    models = [
        ('PaddlePaddle/PaddleOCR-VL-1.5', 'PaddlePaddle/PaddleOCR-VL-1.5'),
        ('PaddlePaddle/PaddleOCR-VL-1.6', 'PaddlePaddle/PaddleOCR-VL-1.6'),
    ]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner='model.projector',
            vision_tower='model.visual',
        )


@register_model
class ErnieVLLoader(ModelLoader):
    """ERNIE-4.5-VL: a transformers-native MoE VLM (``Ernie4_5_VLMoeForConditionalGeneration``).

    Legacy's ``leaf_modules = MOEAllGatherLayerV2`` is stated declaratively as ``ModelArch.moe_block``
    (matched by class *name*, no dynamic import -- see ``apply_z3_leaf_modules``), and its
    ``model.add_image_preprocess(processor)`` rides ``build_model`` (which receives ``processor``).
    """

    model_type = 'ernie_vl'
    model_cls = 'transformers:AutoModelForImageTextToText'
    architectures = ['Ernie4_5_VLMoeForConditionalGeneration']
    template = 'ernie_vl'
    requires = ['transformers>=4.52', 'moviepy']
    is_multimodal = True
    is_moe = True
    models = [
        ('PaddlePaddle/ERNIE-4.5-VL-28B-A3B-PT', 'baidu/ERNIE-4.5-VL-28B-A3B-PT'),
        ('PaddlePaddle/ERNIE-4.5-VL-424B-A47B-PT', 'baidu/ERNIE-4.5-VL-424B-A47B-PT'),
        ('PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Base-PT', 'baidu/ERNIE-4.5-VL-28B-A3B-Base-PT'),
        ('PaddlePaddle/ERNIE-4.5-VL-424B-A47B-Base-PT', 'baidu/ERNIE-4.5-VL-424B-A47B-Base-PT'),
    ]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['model', 'lm_head'],
            aligner='model.resampler_model',
            vision_tower='vision_model',
            moe_block=['MOEAllGatherLayerV2'],
        )

    def build_model(self, model_dir, config, processor, **kwargs):
        model = super().build_model(model_dir, config, processor, **kwargs)
        model.add_image_preprocess(processor)
        return model


@register_model
class ErnieVLThinkingLoader(ErnieVLLoader):
    """Thinking checkpoint: same architecture, ``ernie_vl_thinking`` chat format."""

    model_type = 'ernie_vl_thinking'
    architectures = []  # template variant: reverse-lookup must land on `ernie_vl`
    template = 'ernie_vl_thinking'
    models = [('PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Thinking', 'baidu/ERNIE-4.5-VL-28B-A3B-Thinking')]
