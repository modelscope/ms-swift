# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the TeleChat families.

Ported from ``swift/model/models/telechat.py``. ``telechat`` copies a handful of special-token ids
off the checkpoint's ``generation_config`` onto the processor (legacy did this in ``get_model``;
here it rides ``build_processor``, which already has ``model_dir``). ``telechat2`` is a plain LLM.
"""
from __future__ import annotations

from transformers import GenerationConfig

from .base import ModelLoader, register_model


@register_model
class TelechatLoader(ModelLoader):

    model_type = 'telechat'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['TelechatForCausalLM', 'TeleChatForCausalLM']
    template = 'telechat'
    models = [
        ('TeleAI/TeleChat-7B', 'Tele-AI/telechat-7B'),
        ('TeleAI/TeleChat-12B', 'Tele-AI/TeleChat-12B'),
        ('TeleAI/TeleChat-12B-v2', 'Tele-AI/TeleChat-12B-v2'),
        ('TeleAI/TeleChat-52B', 'TeleAI/TeleChat-52B'),
        'swift/TeleChat-12B-V2-GPTQ-Int4',
        ('TeleAI/TeleChat2-35B', 'Tele-AI/TeleChat2-35B'),
        ('TeleAI/TeleChat2-115B', 'Tele-AI/TeleChat2-115B'),
    ]

    def build_processor(self, model_dir, config, **kwargs):
        processor = super().build_processor(model_dir, config, **kwargs)
        generation_config = GenerationConfig.from_pretrained(model_dir)
        for k in ['bos_token_id', 'eos_token_id', 'pad_token_id', 'user_token_id', 'bot_token_id']:
            setattr(processor, k, getattr(generation_config, k))
        return processor


@register_model
class Telechat2Loader(ModelLoader):

    model_type = 'telechat2'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['TeleChat2ForCausalLM']
    template = 'telechat2'
    models = [
        ('TeleAI/TeleChat2-3B', 'Tele-AI/TeleChat2-3B'),
        ('TeleAI/TeleChat2-7B-32K', 'Tele-AI/TeleChat2-7B-32K'),
        ('TeleAI/TeleChat2-35B-32K', 'Tele-AI/TeleChat2-35B-32K'),
        ('TeleAI/TeleChat2-35B-Nov', 'Tele-AI/TeleChat2-35B-Nov'),
    ]
