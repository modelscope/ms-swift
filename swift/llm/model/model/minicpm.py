# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import contextmanager
from functools import wraps, partial
from types import MethodType
from typing import Dict, Any, List

from transformers import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.llm import TemplateType
from .deepseek import get_model_tokenizer_deepseek_moe
from ..constant import LLMModelType, MLLMModelType
from ..patcher import patch_fixed_device, patch_output_clone
from ..register import (Model, ModelGroup, ModelMeta, register_model, get_model_tokenizer_with_flash_attn)
from ..utils import _use_submodel_func

register_model(
    ModelMeta(
        LLMModelType.minicpm_moe,
        [
            ModelGroup(
                [
                    Model('OpenBMB/MiniCPM-MoE-8x2B', 'openbmb/MiniCPM-MoE-8x2B'),
                 ],
                requires=['transformers>=4.36.0'],
                tags=['moe'],
            ),
        ],
        TemplateType.minicpm,
        get_model_tokenizer_deepseek_moe,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=True,
    ))


def _patch_minicpm_v_device_map(model) -> None:
    if not hasattr(model, 'hf_device_map') or len(model.hf_device_map.values()) == 1:
        return

    device = list(model.hf_device_map.values())[0]
    if hasattr(model, 'get_vision_embedding') and not hasattr(model, '_old_get_vision_embedding'):
        # minicpm-v-v2-chat; avoid double patching
        _old_get_vision_embedding = model.__class__.get_vision_embedding

        def _get_vision_embedding(self, pixel_values):
            if len(pixel_values) == 0:
                return _old_get_vision_embedding(self, pixel_values)
            output = _old_get_vision_embedding(self, pixel_values)
            return output.to(device=device)

        model.__class__._old_get_vision_embedding = _old_get_vision_embedding
        model.__class__.get_vision_embedding = _get_vision_embedding

    if hasattr(model, 'resampler'):  # minicpm-v-v2_5-chat
        patch_fixed_device(model.resampler, device)


def get_model_tokenizer_minicpm_v(model_dir: str,
                                  config: PretrainedConfig,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, config, model_kwargs, load_model, **kwargs)
    if load_model:
        model.resampler.to(config.torch_dtype)  # fix float32
        _patch_minicpm_v_device_map(model)
        func_list = ['generate', 'get_input_embeddings', 'forward']
        _use_submodel_func(model, 'llm', func_list)
        if hasattr(model, 'get_slice_image_placeholder'):
            tokenizer.get_slice_image_placeholder = MethodType(model.get_slice_image_placeholder, tokenizer)
            tokenizer.transform = MethodType(model.transform, tokenizer)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.minicpmv,
        [
            ModelGroup(
                [
                    Model('OpenBMB/MiniCPM-V', 'openbmb/MiniCPM-V'),
                    Model('OpenBMB/MiniCPM-V-2', 'openbmb/MiniCPM-V-2'),
                ],
                requires=['timm', 'transformers<4.42'],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.minicpmv,
        get_model_tokenizer_minicpm_v,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=True,
    ))


@contextmanager
def ignore_check_imports():
    import transformers.dynamic_module_utils as td

    @wraps(td.check_imports)
    def _check_imports(filename) -> List[str]:
        return td.get_relative_imports(filename)

    td._old_check_imports = td.check_imports
    td.check_imports = _check_imports
    yield
    td.check_imports = td._old_check_imports


def get_model_tokenizer_minicpm_v_2_x(model_dir: str,
                                      config: PretrainedConfig,
                                      model_kwargs: Dict[str, Any],
                                      load_model: bool = True,
                                      **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    version = kwargs.get('version', 'v2.5')
    if load_model and version == 'v2.6':
        with ignore_check_imports():
            model_cls = get_class_from_dynamic_module('modeling_navit_siglip.SiglipVisionTransformer', model_dir)
            model_cls._no_split_modules = []
    model, tokenizer = get_model_tokenizer_minicpm_v(model_dir, config, model_kwargs, load_model, **kwargs)
    tokenizer.processor = processor
    if load_model:
        embedding = model.get_input_embeddings()
        patch_output_clone(embedding)

    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.minicpmv2_6,
        [
            ModelGroup(
                [
                    Model('OpenBMB/MiniCPM-V-2_6', 'openbmb/MiniCPM-V-2_6'),
                ],
                requires=['timm', 'transformers>=4.36'],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.minicpmv2_6,
        partial(get_model_tokenizer_minicpm_v_2_x, version='v2.6'),
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=True,
    ))


register_model(
    ModelMeta(
        MLLMModelType.minicpmv2_5,
        [
            ModelGroup(
                [
                    Model('OpenBMB/MiniCPM-Llama3-V-2_5', 'openbmb/MiniCPM-Llama3-V-2_5'),
                ],
                requires=['timm', 'transformers>=4.36'],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.minicpmv2_5,
        get_model_tokenizer_minicpm_v_2_x,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=True,
    ))
