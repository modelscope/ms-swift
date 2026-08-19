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
