# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loader for Mamba.

Ported from ``swift/model/models/mamba.py``. Install ``causal-conv1d>=1.2.0`` and ``mamba-ssm`` or
training/inference will be very slow (legacy logged this at load time; it is a setup note, not a
runtime behavior, so it lives here rather than as a log line).
"""
from __future__ import annotations

from .base import ModelLoader, register_model


@register_model
class MambaLoader(ModelLoader):

    model_type = 'mamba'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['MambaForCausalLM']
    template = 'default'
    requires = ['transformers>=4.39.0']
    models = [
        ('AI-ModelScope/mamba-130m-hf', 'state-spaces/mamba-130m-hf'),
        ('AI-ModelScope/mamba-370m-hf', 'state-spaces/mamba-370m-hf'),
        ('AI-ModelScope/mamba-390m-hf', 'state-spaces/mamba-390m-hf'),
        ('AI-ModelScope/mamba-790m-hf', 'state-spaces/mamba-790m-hf'),
        ('AI-ModelScope/mamba-1.4b-hf', 'state-spaces/mamba-1.4b-hf'),
        ('AI-ModelScope/mamba-2.8b-hf', 'state-spaces/mamba-2.8b-hf'),
    ]
