# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.template import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import register_model

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
        template=TemplateType.seed_oss,
        architectures=['SeedOssForCausalLM'],
        requires=['transformers>=4.56']))
