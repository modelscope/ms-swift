# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: F401

from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal, register_model
from ...template.constant import TemplateType
from .qwen import patch_qwen_vl_utils


def get_model_tokenizer_keye_vl_1_5(model_dir: str, *args, **kwargs):
    model, processor = get_model_tokenizer_multimodal(model_dir, *args, **kwargs)
    from keye_vl_utils import vision_process
    global_vars = patch_qwen_vl_utils(vision_process)
    processor.global_vars = global_vars
    return model, processor


# Ensure registration executed on import
register_model(
    ModelMeta(
        MLLMModelType.keye_vl_1_5,
        [
            ModelGroup([
                Model('Kwai-Keye/Keye-VL-1_5-8B', 'Kwai-Keye/Keye-VL-1_5-8B'),
            ]),
        ],
        TemplateType.keye_vl_1_5,
        get_model_tokenizer_keye_vl_1_5,
        model_arch=ModelArch.keye_vl,
        architectures=['KeyeVL1_5ForConditionalGeneration'],
        tags=['vision'],
        requires=['keye_vl_utils>=1.5.2'],
    ))


