# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the Microsoft Phi text families.

Ported from ``swift/model/models/microsoft.py`` -- the plain causal-LM Phi checkpoints only.

Not migrated here (custom loaders / MLLM, see MODEL_MIGRATION.md): ``phi3_vision`` / ``phi4_multimodal``
/ ``florence`` (multimodal), and ``phi3_small`` (a per-layer ``rotary_emb.forward`` dtype patch over a
hardcoded 32 layers).
"""
from __future__ import annotations

from .base import ModelLoader, register_model


@register_model
class Phi2Loader(ModelLoader):

    model_type = 'phi2'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['PhiForCausalLM']
    template = 'default'
    models = [('AI-ModelScope/phi-2', 'microsoft/phi-2')]


@register_model
class Phi3Loader(ModelLoader):

    model_type = 'phi3'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['Phi3ForCausalLM']
    template = 'phi3'
    requires = ['transformers>=4.36']
    models = [
        ('LLM-Research/Phi-3-mini-4k-instruct', 'microsoft/Phi-3-mini-4k-instruct'),
        ('LLM-Research/Phi-3-mini-128k-instruct', 'microsoft/Phi-3-mini-128k-instruct'),
        ('LLM-Research/Phi-3-medium-4k-instruct', 'microsoft/Phi-3-medium-4k-instruct'),
        ('LLM-Research/Phi-3-medium-128k-instruct', 'microsoft/Phi-3-medium-128k-instruct'),
        ('LLM-Research/Phi-3.5-mini-instruct', 'microsoft/Phi-3.5-mini-instruct'),
        ('LLM-Research/Phi-4-mini-instruct', 'microsoft/Phi-4-mini-instruct'),
    ]


@register_model
class Phi4Loader(Phi3Loader):
    """phi-4 shares ``Phi3ForCausalLM`` with phi3; only the chat template differs."""

    model_type = 'phi4'
    architectures = []
    template = 'phi4'
    models = [('LLM-Research/phi-4', 'microsoft/phi-4')]


@register_model
class Phi3MoeLoader(ModelLoader):

    model_type = 'phi3_moe'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['PhiMoEForCausalLM']
    template = 'phi3'
    requires = ['transformers>=4.36']
    models = [('LLM-Research/Phi-3.5-MoE-instruct', 'microsoft/Phi-3.5-MoE-instruct')]
