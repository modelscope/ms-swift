# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the StepFun families.

Ported from ``swift/model/models/stepfun.py`` -- the transformers-native ``got_ocr2_hf`` only.

Not migrated here (see MODEL_MIGRATION.md):
  * ``got_ocr2`` -- the original remote-code ``GOTQwenForCausalLM`` loaded via bare ``AutoModel``
    (bucket B: remote-code / AutoModel load seam).
  * ``step3_vl`` -- needs weight ``key_mapping`` surgery at load (bucket B).
  * ``step_audio`` / ``step_audio2_mini`` -- ``git_clone`` an external repo / forward-time patch
    (bucket C).
"""
from __future__ import annotations

from .base import ModelArch, ModelLoader, register_model


@register_model
class GotOCR2HfLoader(ModelLoader):
    """GOT-OCR-2.0 (transformers-native). Legacy patched ``_no_split_modules`` onto the model class
    to steer accelerate's ``device_map`` sharding, but dev places the model through twinkle
    strategies and never uses an HF ``device_map`` (see PATCH_INVENTORY.md: ``patch_device_map`` is
    obsolete), so no override is needed -- a plain ``model_cls`` + ``model_arch`` suffices."""

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


@register_model
class Step3VLLoader(ModelLoader):
    """Step3-VL: remote-code ``StepVLForConditionalGeneration`` (no in-tree class on transformers 5.5),
    so it loads via ``AutoModelForImageTextToText`` + the ``trust_remote_code`` flag.

    Two faithful seams:
      * the top-level ``config.vocab_size`` is unset/stale on these checkpoints, so it is copied from
        ``config.text_config`` -- a config fix, hence ``process_config``.
      * the checkpoint's weight names are flat (``vision_model.*`` / ``model.*``) while the class
        expects them nested under ``model.``; legacy passes transformers' ``key_mapping`` to remap on
        load, forwarded here as a ``build_model`` kwarg default (an explicit caller kwarg still wins).
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
