# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the OpenBuddy families.

Ported from ``swift/model/models/openbuddy.py``. Plain Llama/Mistral/Mixtral backbones with the
OpenBuddy chat format. ``openbuddy_llama`` legacy mixed two chat templates across its groups
(``openbuddy`` for the v8/v10/deepseek checkpoints, ``openbuddy2`` for the llama3+ ones); that is a
template variant, so it is split into a base loader and an ``architectures=[]`` subclass to keep the
reverse ``LlamaForCausalLM`` lookup from being polluted.
"""
from __future__ import annotations

from .base import ModelLoader, register_model


@register_model
class OpenBuddyLlamaLoader(ModelLoader):

    model_type = 'openbuddy_llama'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['LlamaForCausalLM']
    mcore_model_type = 'gpt'
    template = 'openbuddy'
    models = [
        ('OpenBuddy/openbuddy-llama-65b-v8-bf16', 'OpenBuddy/openbuddy-llama-65b-v8-bf16'),
        ('OpenBuddy/openbuddy-llama2-13b-v8.1-fp16', 'OpenBuddy/openbuddy-llama2-13b-v8.1-fp16'),
        ('OpenBuddy/openbuddy-llama2-70b-v10.1-bf16', 'OpenBuddy/openbuddy-llama2-70b-v10.1-bf16'),
        ('OpenBuddy/openbuddy-deepseek-67b-v15.2', 'OpenBuddy/openbuddy-deepseek-67b-v15.2'),
    ]


@register_model
class OpenBuddyLlama2Loader(OpenBuddyLlamaLoader):
    """The llama3+ OpenBuddy checkpoints, which use the ``openbuddy2`` chat format."""

    model_type = 'openbuddy_llama2'
    architectures = []
    template = 'openbuddy2'
    # Highest floor across the merged groups (llama3.3 needs >=4.45).
    requires = ['transformers>=4.45']
    models = [
        ('OpenBuddy/openbuddy-llama3-8b-v21.1-8k', 'OpenBuddy/openbuddy-llama3-8b-v21.1-8k'),
        ('OpenBuddy/openbuddy-llama3-70b-v21.1-8k', 'OpenBuddy/openbuddy-llama3-70b-v21.1-8k'),
        ('OpenBuddy/openbuddy-yi1.5-34b-v21.3-32k', 'OpenBuddy/openbuddy-yi1.5-34b-v21.3-32k'),
        ('OpenBuddy/openbuddy-llama3.1-8b-v22.1-131k', 'OpenBuddy/openbuddy-llama3.1-8b-v22.1-131k'),
        ('OpenBuddy/openbuddy-nemotron-70b-v23.2-131k', 'OpenBuddy/openbuddy-nemotron-70b-v23.2-131k'),
        ('OpenBuddy/openbuddy-llama3.3-70b-v24.3-131k', 'OpenBuddy/openbuddy-llama3.3-70b-v24.3-131k'),
    ]


@register_model
class OpenBuddyMistralLoader(ModelLoader):

    model_type = 'openbuddy_mistral'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['MistralForCausalLM']
    template = 'openbuddy'
    requires = ['transformers>=4.34']
    models = [
        ('OpenBuddy/openbuddy-mistral-7b-v17.1-32k', 'OpenBuddy/openbuddy-mistral-7b-v17.1-32k'),
        ('OpenBuddy/openbuddy-zephyr-7b-v14.1', 'OpenBuddy/openbuddy-zephyr-7b-v14.1'),
    ]


@register_model
class OpenBuddyMixtralLoader(ModelLoader):

    model_type = 'openbuddy_mixtral'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['MixtralForCausalLM']
    template = 'openbuddy'
    requires = ['transformers>=4.36']
    models = [('OpenBuddy/openbuddy-mixtral-7bx8-v18.1-32k', 'OpenBuddy/openbuddy-mixtral-7bx8-v18.1-32k')]
