# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the CodeFuse families.

Ported from ``swift/model/models/codefuse.py``. Only ``codefuse_codellama`` is migrated. Dropped
(see MODEL_MIGRATION.md): ``codefuse_qwen`` reused the un-migrated Qwen1 ``QWenLMHeadModel`` loader,
and ``codefuse_codegeex2`` reused ``ChatGLMLoader`` pinned to ``transformers<4.34`` -- both old,
trust-remote-code constructions.
"""
from __future__ import annotations

from transformers import AutoTokenizer

from .base import ModelLoader, register_model


@register_model
class CodeFuseCodeLlamaLoader(ModelLoader):

    model_type = 'codefuse_codellama'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['LlamaForCausalLM']
    template = 'codefuse_codellama'
    mcore_model_type = 'gpt'
    tags = ['coding']
    models = [('codefuse-ai/CodeFuse-CodeLlama-34B', 'codefuse-ai/CodeFuse-CodeLlama-34B')]

    def build_processor(self, model_dir, config, **kwargs):
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False, legacy=False)
