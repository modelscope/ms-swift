# Copyright (c) ModelScope Contributors. All rights reserved.
# got_ocr2_hf / step3_vl migrated (got_ocr2 / step_audio* dropped, MODEL_MIGRATION.md).
from __future__ import annotations

from .base import ModelArch, ModelLoader, register_model


@register_model
class GotOCR2HfLoader(ModelLoader):
    """GOT-OCR-2.0 (transformers-native).

    Legacy also set ``_no_split_modules = ['GotOcr2VisionLayer']`` on the model class -- that
    survives here (instance-level via ``process_model``) because it has a consumer beyond HF
    ``device_map``: twinkle's FSDP2 reads ``_no_split_modules`` to pick its wrap boundary, and the
    in-tree class declares none, so without it the vision layers lose their no-split guarantee.
    """

    model_type = 'got_ocr2_hf'
    model_cls = 'transformers:GotOcr2ForConditionalGeneration'
    architectures = ['GotOcr2ForConditionalGeneration']
    template = 'got_ocr2_hf'
    tags = ['vision']
    is_multimodal = True
    models = [('stepfun-ai/GOT-OCR-2.0-hf', 'stepfun-ai/GOT-OCR-2.0-hf')]

    @property
    def model_arch(self) -> ModelArch:
        # legacy `ModelArch.llava_hf`, transformers>=4.52 (model.* prefix) branch.
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner='model.multi_modal_projector',
            vision_tower='model.vision_tower',
        )

    def process_model(self, model):
        # Class-level in legacy; instance-level here -- twinkle FSDP2 reads it off the model.
        model._no_split_modules = ['GotOcr2VisionLayer']
        return model


@register_model
class Step3VLLoader(ModelLoader):
    """Step3-VL: remote-code ``StepVLForConditionalGeneration`` via ``AutoModelForImageTextToText``.

    The top-level ``config.vocab_size`` is stale on these checkpoints, so it is copied from
    ``config.text_config`` (a config fix, hence ``process_config``); and the checkpoint's flat weight
    names are remapped to the class's nested layout through transformers' ``key_mapping`` (a
    ``build_model`` kwarg default -- an explicit caller kwarg still wins).
    """

    model_type = 'step3_vl'
    model_cls = 'transformers:AutoModelForImageTextToText'
    trust_remote_code = True
    architectures = ['StepVLForConditionalGeneration']
    template = 'step3_vl'
    requires = ['transformers>=4.57.0']
    tags = ['vision']
    is_multimodal = True
    models = [
        ('stepfun-ai/Step3-VL-10B-Base', 'stepfun-ai/Step3-VL-10B-Base'),
        ('stepfun-ai/Step3-VL-10B', 'stepfun-ai/Step3-VL-10B'),
    ]
    key_mapping = {
        '^vision_model': 'model.vision_model',
        r'^model(?!\.(language_model|vision_model))': 'model.language_model',
        'vit_large_projector': 'model.vit_large_projector',
    }

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner='model.vit_large_projector',
            vision_tower='model.vision_model',
        )

    def process_config(self, config):
        config = super().process_config(config)
        config.vocab_size = config.text_config.vocab_size
        return config

    def build_model(self, model_dir, config, processor, **kwargs):
        kwargs.setdefault('key_mapping', self.key_mapping)
        return super().build_model(model_dir, config, processor, **kwargs)
