# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from .base import ModelLoader, register_model


@register_model
class SeedOssLoader(ModelLoader):

    model_type = 'seed_oss'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['SeedOssForCausalLM']
    template = 'seed_oss'
    requires = ['transformers>=4.56']
    models = [
        'ByteDance-Seed/Seed-OSS-36B-Instruct',
        'ByteDance-Seed/Seed-OSS-36B-Base',
        'ByteDance-Seed/Seed-OSS-36B-Base-woSyn',
    ]
