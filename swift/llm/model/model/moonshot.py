# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.llm import TemplateType
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..patcher import patch_get_input_embeddings
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
                        get_model_tokenizer_with_flash_attn, register_model)


def get_model_tokenizer_kimi_vl(model_dir, *args, **kwargs):
    KimiVLPreTrainedModel = get_class_from_dynamic_module('modeling_kimi_vl.KimiVLPreTrainedModel', model_dir)
    try:
        del KimiVLPreTrainedModel._supports_sdpa
    except AttributeError:
        pass
    model, processor = get_model_tokenizer_multimodal(model_dir, *args, **kwargs)
    if model is not None:
        patch_get_input_embeddings(model.vision_tower, 'patch_embed')
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.kimi_vl,
        [
            ModelGroup([
                Model('moonshotai/Kimi-VL-A3B-Instruct', 'moonshotai/Kimi-VL-A3B-Instruct'),
                Model('moonshotai/Kimi-VL-A3B-Thinking', 'moonshotai/Kimi-VL-A3B-Thinking'),
                Model('moonshotai/Kimi-VL-A3B-Thinking-2506', 'moonshotai/Kimi-VL-A3B-Thinking-2506'),
            ])
        ],
        get_model_tokenizer_kimi_vl,
        template=TemplateType.kimi_vl,
        model_arch=ModelArch.llava_hf_legacy,
        requires=['transformers<4.49'],
    ))
