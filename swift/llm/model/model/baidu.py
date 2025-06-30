# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model

logger = get_logger()

register_model(
    ModelMeta(
        LLMModelType.ernie,
        [
            ModelGroup([
                Model('PaddlePaddle/ERNIE-4.5-0.3B-PT', 'baidu/ERNIE-4.5-0.3B-PT'),
            ]),
        ],
        TemplateType.ernie,
        get_model_tokenizer_with_flash_attn,
        architectures=['Ernie4_5_ForCausalLM'],
    ))
