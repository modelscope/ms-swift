# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
                        get_model_tokenizer_with_flash_attn, register_model)

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


def get_model_tokenizer_ernie_vl(*args, **kwargs):
    model, processor = get_model_tokenizer_multimodal(*args, **kwargs)
    if model is not None:
        model.add_image_preprocess(processor)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.ernie_vl,
        [
            ModelGroup([
                Model('PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Base-PT', 'baidu/ERNIE-4.5-VL-28B-A3B-Base-PT'),
                Model('PaddlePaddle/ERNIE-4.5-VL-28B-A3B-PT', 'baidu/ERNIE-4.5-VL-28B-A3B-PT'),
                Model('PaddlePaddle/ERNIE-4.5-VL-424B-A47B-Base-PT', 'baidu/ERNIE-4.5-VL-424B-A47B-Base-PT'),
                Model('PaddlePaddle/ERNIE-4.5-VL-424B-A47B-PT', 'baidu/ERNIE-4.5-VL-424B-A47B-PT'),
            ]),
        ],
        TemplateType.ernie_vl,
        get_model_tokenizer_ernie_vl,
        model_arch=ModelArch.ernie_vl,
        architectures=['Ernie4_5_VLMoeForConditionalGeneration'],
    ))
