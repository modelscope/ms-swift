# Copyright (c) ModelScope Contributors. All rights reserved.

from transformers import PreTrainedModel

from swift.template import TemplateType
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model


class HunyuanVLLoader(ModelLoader):

    def get_config(self, model_dir: str):
        self.attn_impl = self.attn_impl or 'eager'
        return super().get_config(model_dir)

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import HunYuanVLForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or HunYuanVLForConditionalGeneration
        return super().get_model(model_dir, *args, **kwargs)


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
