# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the MiniMax text families.

Ported from ``swift/model/models/minimax.py`` -- only ``minimax_m2`` (a plain
``MiniMaxM2ForCausalLM``) and its chat-template variants are migrated.

Not migrated here (see MODEL_MIGRATION.md):
  * ``minimax`` / ``minimax_m1`` -- ``MinimaxTextLoader`` builds a manual multi-GPU ``device_map``,
    rewrites Quanto ``modules_to_not_convert``, and warns the family "does not support training".
  * ``minimax_vl`` -- multimodal, builds a manual ``device_map`` (bucket C).
"""
from __future__ import annotations

from transformers import AutoProcessor

from .base import ModelArch, ModelLoader, register_model


@register_model
class MinimaxM2Loader(ModelLoader):

    model_type = 'minimax_m2'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['MiniMaxM2ForCausalLM']
    template = 'minimax_m2'
    requires = ['transformers==4.57.1']
    models = [('MiniMax/MiniMax-M2', 'MiniMaxAI/MiniMax-M2')]


@register_model
class MinimaxM2_1Loader(MinimaxM2Loader):

    model_type = 'minimax_m2_1'
    architectures = []
    template = 'minimax_m2_1'
    models = [('MiniMax/MiniMax-M2.1', 'MiniMaxAI/MiniMax-M2.1')]


@register_model
class MinimaxM2_5Loader(MinimaxM2Loader):

    model_type = 'minimax_m2_5'
    architectures = []
    template = 'minimax_m2_5'
    models = [('MiniMax/MiniMax-M2.5', 'MiniMaxAI/MiniMax-M2.5')]


@register_model
class MinimaxM2_7Loader(MinimaxM2Loader):

    model_type = 'minimax_m2_7'
    architectures = []
    template = 'minimax_m2_7'
    models = [('MiniMax/MiniMax-M2.7', 'MiniMaxAI/MiniMax-M2.7')]


@register_model
class MinimaxM3VLLoader(ModelLoader):
    """MiniMax-M3 vision-language. Loads via the generic ``AutoModelForImageTextToText`` (the model
    code ships in-tree, so no ``trust_remote_code`` for the model), but its *processor* is
    remote-code and must be built with ``trust_remote_code=True`` -- the one legacy split."""

    model_type = 'minimax_m3_vl'
    model_cls = 'transformers:AutoModelForImageTextToText'
    architectures = ['MiniMaxM3SparseForConditionalGeneration']
    template = 'minimax_m3_vl'
    tags = ['vision', 'video']
    is_multimodal = True
    models = [('MiniMax/MiniMax-M3', 'MiniMaxAI/MiniMax-M3')]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner='model.multi_modal_projector',
            vision_tower='model.vision_tower',
        )

    def build_processor(self, model_dir, config, **kwargs):
        kwargs['trust_remote_code'] = True
        return AutoProcessor.from_pretrained(model_dir, **kwargs)
