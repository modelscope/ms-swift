# Copyright (c) Alibaba, Inc. and its affiliates.

from transformers import PreTrainedModel

from swift.template import TemplateType
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model


class HunyuanVLLoader(ModelLoader):

    def get_model(self, model_dir: str, config, model_kwargs) -> PreTrainedModel:
        from transformers import HunYuanVLForConditionalGeneration
        self.automodel_class = self.automodel_class or HunYuanVLForConditionalGeneration
        self.attn_impl = self.attn_impl or 'eager'
        return super().get_model(model_dir, config, model_kwargs)


register_model(
    ModelMeta(
        MLLMModelType.hunyuan_ocr,
        [
            ModelGroup([
                Model('Tencent-Hunyuan/HunyuanOCR', 'tencent/HunyuanOCR'),
            ]),
        ],
        HunyuanVLLoader,
        template=TemplateType.hunyuan_ocr,
        architectures=['HunYuanVLForConditionalGeneration'],
        model_arch=ModelArch.hunyuan_vl,
        requires=['transformers>=4.49.0'],
    ))
