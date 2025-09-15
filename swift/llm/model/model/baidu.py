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
                Model('PaddlePaddle/ERNIE-4.5-0.3B-Base-PT', 'baidu/ERNIE-4.5-0.3B-PT'),
                Model('PaddlePaddle/ERNIE-4.5-0.3B-PT', 'baidu/ERNIE-4.5-0.3B-PT'),
            ]),
            ModelGroup([
                Model('PaddlePaddle/ERNIE-4.5-21B-A3B-Base-PT', 'baidu/ERNIE-4.5-21B-A3B-Base-PT'),
                Model('PaddlePaddle/ERNIE-4.5-21B-A3B-PT', 'baidu/ERNIE-4.5-21B-A3B-PT'),
                Model('PaddlePaddle/ERNIE-4.5-300B-A47B-Base-PT', 'baidu/ERNIE-4.5-300B-A47B-Base-PT'),
                Model('PaddlePaddle/ERNIE-4.5-300B-A47B-PT', 'baidu/ERNIE-4.5-300B-A47B-PT'),
            ]),
        ],
        TemplateType.ernie,
        get_model_tokenizer_with_flash_attn,
        architectures=['Ernie4_5_ForCausalLM', 'Ernie4_5_MoeForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.ernie_thinking,
        [
            ModelGroup([
                Model('PaddlePaddle/ERNIE-4.5-21B-A3B-Thinking', 'baidu/ERNIE-4.5-21B-A3B-Thinking'),
            ]),
        ],
        TemplateType.ernie_thinking,
        get_model_tokenizer_with_flash_attn,
        architectures=['Ernie4_5_ForCausalLM', 'Ernie4_5_MoeForCausalLM'],
    ))
