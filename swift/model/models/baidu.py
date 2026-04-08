# Copyright (c) ModelScope Contributors. All rights reserved.
from transformers import PretrainedConfig, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.template import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model

logger = get_logger()

register_model(
    ModelMeta(
        LLMModelType.ernie4_5,
        [
            ModelGroup([
                Model('PaddlePaddle/ERNIE-4.5-0.3B-Base-PT', 'baidu/ERNIE-4.5-0.3B-PT'),
                Model('PaddlePaddle/ERNIE-4.5-0.3B-PT', 'baidu/ERNIE-4.5-0.3B-PT'),
            ], TemplateType.ernie),
        ],
        architectures=['Ernie4_5_ForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.ernie4_5_moe,
        [
            ModelGroup([
                Model('PaddlePaddle/ERNIE-4.5-21B-A3B-Base-PT', 'baidu/ERNIE-4.5-21B-A3B-Base-PT'),
                Model('PaddlePaddle/ERNIE-4.5-21B-A3B-PT', 'baidu/ERNIE-4.5-21B-A3B-PT'),
                Model('PaddlePaddle/ERNIE-4.5-300B-A47B-Base-PT', 'baidu/ERNIE-4.5-300B-A47B-Base-PT'),
                Model('PaddlePaddle/ERNIE-4.5-300B-A47B-PT', 'baidu/ERNIE-4.5-300B-A47B-PT'),
            ], TemplateType.ernie),
            ModelGroup([
                Model('PaddlePaddle/ERNIE-4.5-21B-A3B-Thinking', 'baidu/ERNIE-4.5-21B-A3B-Thinking'),
            ], TemplateType.ernie_thinking),
        ],
        architectures=['Ernie4_5_MoeForCausalLM'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.paddle_ocr,
        [
            ModelGroup([
                Model('PaddlePaddle/PaddleOCR-VL', 'PaddlePaddle/PaddleOCR-VL'),
            ]),
        ],
        template=TemplateType.paddle_ocr,
        model_arch=ModelArch.keye_vl,
        architectures=['PaddleOCRVLForConditionalGeneration'],
    ))


class ErnieVLLoader(ModelLoader):

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        MOEAllGatherLayerV2 = get_class_from_dynamic_module('modeling_ernie4_5_vl.MOEAllGatherLayerV2', model_dir)
        self.leaf_modules = MOEAllGatherLayerV2
        model = super().get_model(model_dir, config, processor, model_kwargs)
        model.add_image_preprocess(processor)
        return model


register_model(
    ModelMeta(
        MLLMModelType.ernie_vl,
        [
            ModelGroup([
                Model('PaddlePaddle/ERNIE-4.5-VL-28B-A3B-PT', 'baidu/ERNIE-4.5-VL-28B-A3B-PT'),
                Model('PaddlePaddle/ERNIE-4.5-VL-424B-A47B-PT', 'baidu/ERNIE-4.5-VL-424B-A47B-PT'),
                Model('PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Base-PT', 'baidu/ERNIE-4.5-VL-28B-A3B-Base-PT'),
                Model('PaddlePaddle/ERNIE-4.5-VL-424B-A47B-Base-PT', 'baidu/ERNIE-4.5-VL-424B-A47B-Base-PT'),
            ], TemplateType.ernie_vl),
            ModelGroup([
                Model('PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Thinking', 'baidu/ERNIE-4.5-VL-28B-A3B-Thinking'),
            ], TemplateType.ernie_vl_thinking),
        ],
        ErnieVLLoader,
        model_arch=ModelArch.ernie_vl,
        architectures=['Ernie4_5_VLMoeForConditionalGeneration'],
        requires=['transformers>=4.52', 'moviepy'],
    ))


class PaddleOCR1_5Loader(ModelLoader):

    def get_config(self, model_dir: str) -> PretrainedConfig:
        from transformers import AutoConfig
        auto_config_cls = self.auto_config_cls or AutoConfig
        return auto_config_cls.from_pretrained(model_dir)

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import AutoModelForImageTextToText
        self.auto_model_cls = self.auto_model_cls or AutoModelForImageTextToText
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.paddle_ocr_1_5,
        [
            ModelGroup([
                Model('PaddlePaddle/PaddleOCR-VL-1.5', 'PaddlePaddle/PaddleOCR-VL-1.5'),
            ]),
        ],
        PaddleOCR1_5Loader,
        template=TemplateType.paddle_ocr_1_5,
        model_arch=ModelArch.paddle_ocr_1_5,
        architectures=['PaddleOCRVLForConditionalGeneration'],
    ))
