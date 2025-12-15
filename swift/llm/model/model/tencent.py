from typing import Any, Dict

from swift.llm import TemplateType
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal, register_model
from ..utils import ModelInfo


def get_model_tokenizer_hunyuan_vl(model_dir: str,
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    from transformers import HunYuanVLForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or HunYuanVLForConditionalGeneration
    kwargs['attn_impl'] = kwargs['attn_impl'] or 'eager'
    model, processor = get_model_tokenizer_multimodal(model_dir, model_info, model_kwargs, load_model, **kwargs)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.hunyuan_ocr,
        [
            ModelGroup([
                Model('Tencent-Hunyuan/HunyuanOCR', 'tencent/HunyuanOCR'),
            ]),
        ],
        TemplateType.hunyuan_ocr,
        get_model_tokenizer_hunyuan_vl,
        architectures=['HunYuanVLForConditionalGeneration'],
        model_arch=ModelArch.hunyuan_vl,
        requires=['transformers>=4.49.0'],
    ))
