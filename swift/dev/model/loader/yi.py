# Copyright (c) ModelScope Contributors. All rights reserved.
# Yi (Llama-arch) + template-variant subclasses; yi_vl dropped (git-clone construction, MODEL_MIGRATION.md).
from __future__ import annotations

from .base import ModelLoader, register_model


@register_model
class YiLoader(ModelLoader):

    model_type = 'yi'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['LlamaForCausalLM']
    template = 'chatml'
    mcore_model_type = 'gpt'
    models = [
        # yi
        ('01ai/Yi-6B', '01-ai/Yi-6B'),
        ('01ai/Yi-6B-200K', '01-ai/Yi-6B-200K'),
        ('01ai/Yi-6B-Chat', '01-ai/Yi-6B-Chat'),
        ('01ai/Yi-6B-Chat-4bits', '01-ai/Yi-6B-Chat-4bits'),
        ('01ai/Yi-6B-Chat-8bits', '01-ai/Yi-6B-Chat-8bits'),
        ('01ai/Yi-9B', '01-ai/Yi-9B'),
        ('01ai/Yi-9B-200K', '01-ai/Yi-9B-200K'),
        ('01ai/Yi-34B', '01-ai/Yi-34B'),
        ('01ai/Yi-34B-200K', '01-ai/Yi-34B-200K'),
        ('01ai/Yi-34B-Chat', '01-ai/Yi-34B-Chat'),
        ('01ai/Yi-34B-Chat-4bits', '01-ai/Yi-34B-Chat-4bits'),
        ('01ai/Yi-34B-Chat-8bits', '01-ai/Yi-34B-Chat-8bits'),
        # yi1.5
        ('01ai/Yi-1.5-6B', '01-ai/Yi-1.5-6B'),
        ('01ai/Yi-1.5-6B-Chat', '01-ai/Yi-1.5-6B-Chat'),
        ('01ai/Yi-1.5-9B', '01-ai/Yi-1.5-9B'),
        ('01ai/Yi-1.5-9B-Chat', '01-ai/Yi-1.5-9B-Chat'),
        ('01ai/Yi-1.5-9B-Chat-16K', '01-ai/Yi-1.5-9B-Chat-16K'),
        ('01ai/Yi-1.5-34B', '01-ai/Yi-1.5-34B'),
        ('01ai/Yi-1.5-34B-Chat', '01-ai/Yi-1.5-34B-Chat'),
        ('01ai/Yi-1.5-34B-Chat-16K', '01-ai/Yi-1.5-34B-Chat-16K'),
        # yi1.5 quant
        ('AI-ModelScope/Yi-1.5-6B-Chat-GPTQ', 'modelscope/Yi-1.5-6B-Chat-GPTQ'),
        ('AI-ModelScope/Yi-1.5-6B-Chat-AWQ', 'modelscope/Yi-1.5-6B-Chat-AWQ'),
        ('AI-ModelScope/Yi-1.5-9B-Chat-GPTQ', 'modelscope/Yi-1.5-9B-Chat-GPTQ'),
        ('AI-ModelScope/Yi-1.5-9B-Chat-AWQ', 'modelscope/Yi-1.5-9B-Chat-AWQ'),
        ('AI-ModelScope/Yi-1.5-34B-Chat-GPTQ', 'modelscope/Yi-1.5-34B-Chat-GPTQ'),
        ('AI-ModelScope/Yi-1.5-34B-Chat-AWQ', 'modelscope/Yi-1.5-34B-Chat-AWQ'),
    ]


@register_model
class YiCoderLoader(YiLoader):
    """Yi-Coder: Yi loading, the coding chat template (template variant)."""

    model_type = 'yi_coder'
    template = 'yi_coder'
    architectures = []
    tags = ['coding']
    models = [
        ('01ai/Yi-Coder-1.5B', '01-ai/Yi-Coder-1.5B'),
        ('01ai/Yi-Coder-9B', '01-ai/Yi-Coder-9B'),
        ('01ai/Yi-Coder-1.5B-Chat', '01-ai/Yi-Coder-1.5B-Chat'),
        ('01ai/Yi-Coder-9B-Chat', '01-ai/Yi-Coder-9B-Chat'),
    ]


@register_model
class SusChatLoader(YiLoader):
    """SUS-Chat: Yi loading, the SUS chat template (template variant)."""

    model_type = 'sus_chat'
    template = 'sus'
    architectures = []
    models = [('SUSTC/SUS-Chat-34B', 'SUSTech/SUS-Chat-34B')]
