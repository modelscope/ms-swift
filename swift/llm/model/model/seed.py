# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model

logger = get_logger()

register_model(
    ModelMeta(
        LLMModelType.seed_oss, [
            ModelGroup([
                Model('ByteDance-Seed/Seed-OSS-36B-Instruct', 'ByteDance-Seed/Seed-OSS-36B-Instruct'),
                Model('ByteDance-Seed/Seed-OSS-36B-Base', 'ByteDance-Seed/Seed-OSS-36B-Base'),
                Model('ByteDance-Seed/Seed-OSS-36B-Base-woSyn', 'ByteDance-Seed/Seed-OSS-36B-Base-woSyn'),
            ])
        ],
        TemplateType.seed_oss,
        get_model_tokenizer_with_flash_attn,
        architectures=['SeedOssForCausalLM'],
        requires=['transformers>=4.56']))
