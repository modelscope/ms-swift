# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers.dynamic_module_utils import get_class_from_dynamic_module

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

register_model(
    ModelMeta(
        MLLMModelType.paddle_ocr,
        [
            ModelGroup([
                Model('PaddlePaddle/PaddleOCR-VL', 'PaddlePaddle/PaddleOCR-VL'),
            ]),
        ],
        TemplateType.paddle_ocr,
        get_model_tokenizer_multimodal,
        model_arch=ModelArch.keye_vl,
        architectures=['PaddleOCRVLForConditionalGeneration'],
    ))


def get_model_tokenizer_ernie_vl(model_dir, *args, **kwargs):
    MOEAllGatherLayerV2 = get_class_from_dynamic_module('modeling_ernie4_5_vl.MOEAllGatherLayerV2', model_dir)
    kwargs['leaf_modules'] = MOEAllGatherLayerV2
    model, processor = get_model_tokenizer_multimodal(model_dir, *args, **kwargs)
    if model is not None:
        model.add_image_preprocess(processor)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.ernie_vl,
        [
            ModelGroup([
                Model('PaddlePaddle/ERNIE-4.5-VL-28B-A3B-PT', 'baidu/ERNIE-4.5-VL-28B-A3B-PT'),
                Model('PaddlePaddle/ERNIE-4.5-VL-424B-A47B-PT', 'baidu/ERNIE-4.5-VL-424B-A47B-PT'),
                Model('PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Base-PT', 'baidu/ERNIE-4.5-VL-28B-A3B-Base-PT'),
                Model('PaddlePaddle/ERNIE-4.5-VL-424B-A47B-Base-PT', 'baidu/ERNIE-4.5-VL-424B-A47B-Base-PT'),
            ]),
        ],
        TemplateType.ernie_vl,
        get_model_tokenizer_ernie_vl,
        model_arch=ModelArch.ernie_vl,
        architectures=['Ernie4_5_VLMoeForConditionalGeneration'],
        requires=['transformers>=4.52', 'moviepy'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.ernie_vl_thinking,
        [
            ModelGroup([
                Model('PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Thinking', 'baidu/ERNIE-4.5-VL-28B-A3B-Thinking'),
            ]),
        ],
        TemplateType.ernie_vl_thinking,
        get_model_tokenizer_ernie_vl,
        model_arch=ModelArch.ernie_vl,
        architectures=['Ernie4_5_VLMoeForConditionalGeneration'],
        requires=['transformers>=4.52', 'moviepy'],
    ))
