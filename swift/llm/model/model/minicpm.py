# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from types import MethodType
from typing import Any, Dict

from transformers import AutoConfig
from transformers.utils import strtobool

from swift.llm import TemplateType
from swift.utils import get_env_args
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..patcher import patch_device_map, patch_fixed_device, patch_output_clone
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo, use_submodel_func
from .deepseek import get_model_tokenizer_deepseek_moe

register_model(
    ModelMeta(
        LLMModelType.minicpm_moe,
        [
            ModelGroup([
                Model('OpenBMB/MiniCPM-MoE-8x2B', 'openbmb/MiniCPM-MoE-8x2B'),
            ]),
        ],
        TemplateType.minicpm,
        get_model_tokenizer_deepseek_moe,
        architectures=['MiniCPMForCausalLM'],
        model_arch=ModelArch.llama,
        requires=['transformers>=4.36'],
    ))


def _patch_minicpmv_device_map(model) -> None:
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


def get_model_tokenizer_minicpmv(model_dir: str,
                                 model_info: ModelInfo,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if load_model:
        model.resampler.to(model_info.torch_dtype)  # fix float32
        _patch_minicpmv_device_map(model)
        func_list = ['generate', 'get_input_embeddings', 'forward']
        use_submodel_func(model, 'llm', func_list)
        if hasattr(model, 'get_slice_image_placeholder'):
            tokenizer.get_slice_image_placeholder = MethodType(model.get_slice_image_placeholder, tokenizer)
            tokenizer.transform = MethodType(model.transform, tokenizer)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.minicpmv,
        [
            ModelGroup([
                Model('OpenBMB/MiniCPM-V', 'openbmb/MiniCPM-V'),
                Model('OpenBMB/MiniCPM-V-2', 'openbmb/MiniCPM-V-2'),
            ], ),
        ],
        TemplateType.minicpmv,
        get_model_tokenizer_minicpmv,
        architectures=['MiniCPMV'],
        model_arch=ModelArch.minicpmv,
        requires=['timm', 'transformers<4.42'],
        tags=['vision'],
    ))


def get_model_tokenizer_minicpmv_2_x(model_dir: str,
                                     model_info: ModelInfo,
                                     model_kwargs: Dict[str, Any],
                                     load_model: bool = True,
                                     **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    version = kwargs.get('version')
    if version == 'o2.6':
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        model_config.init_tts = strtobool(get_env_args('init_tts', str, 'false'))
        model_config.init_audio = strtobool(get_env_args('init_audio', str, 'false'))
        kwargs['model_config'] = model_config
    with patch_device_map():
        model, tokenizer = get_model_tokenizer_minicpmv(
            model_dir, model_info, model_kwargs, load_model, tokenizer=processor.tokenizer, **kwargs)
    if load_model:
        embedding = model.get_input_embeddings()
        patch_output_clone(embedding)

    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.minicpmv2_5,
        [
            ModelGroup([
                Model('OpenBMB/MiniCPM-Llama3-V-2_5', 'openbmb/MiniCPM-Llama3-V-2_5'),
            ], ),
        ],
        TemplateType.minicpmv2_5,
        get_model_tokenizer_minicpmv_2_x,
        architectures=['MiniCPMV'],
        model_arch=ModelArch.minicpmv,
        requires=['timm', 'transformers>=4.36'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.minicpmv2_6,
        [
            ModelGroup([
                Model('OpenBMB/MiniCPM-V-2_6', 'openbmb/MiniCPM-V-2_6'),
            ], ),
        ],
        TemplateType.minicpmv2_6,
        get_model_tokenizer_minicpmv_2_x,
        architectures=['MiniCPMV'],
        model_arch=ModelArch.minicpmv,
        requires=['timm', 'transformers>=4.36', 'decord'],
        tags=['vision', 'video'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.minicpmo2_6,
        [
            ModelGroup([
                Model('OpenBMB/MiniCPM-o-2_6', 'openbmb/MiniCPM-o-2_6'),
            ]),
        ],
        TemplateType.minicpmo2_6,
        partial(get_model_tokenizer_minicpmv_2_x, version='o2.6'),
        architectures=['MiniCPMO'],
        model_arch=ModelArch.minicpmv,
        requires=['timm', 'transformers>=4.36', 'decord', 'soundfile'],
        tags=['vision', 'video', 'omni', 'audio'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.minicpmv4,
        [
            ModelGroup([
                Model('OpenBMB/MiniCPM-V-4', 'openbmb/MiniCPM-V-4'),
            ], ),
        ],
        TemplateType.minicpmv4,
        get_model_tokenizer_minicpmv_2_x,
        architectures=['MiniCPMV'],
        model_arch=ModelArch.minicpmv,
        requires=['timm', 'transformers>=4.36', 'decord'],
        tags=['vision', 'video'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.minicpmv4_5,
        [
            ModelGroup([
                Model('OpenBMB/MiniCPM-V-4_5', 'openbmb/MiniCPM-V-4_5'),
            ], ),
        ],
        TemplateType.minicpmv4_5,
        get_model_tokenizer_minicpmv_2_x,
        architectures=['MiniCPMV'],
        model_arch=ModelArch.minicpmv,
        requires=['timm', 'transformers>=4.36', 'decord'],
        tags=['vision', 'video'],
    ))

register_model(
    ModelMeta(
        LLMModelType.minicpm,
        [
            ModelGroup([
                Model('OpenBMB/MiniCPM-2B-sft-fp32', 'openbmb/MiniCPM-2B-sft-fp32'),
                Model('OpenBMB/MiniCPM-2B-dpo-fp32', 'openbmb/MiniCPM-2B-dpo-fp32'),
                Model('OpenBMB/MiniCPM-1B-sft-bf16', 'openbmb/MiniCPM-1B-sft-bf16'),
            ], ),
        ],
        TemplateType.minicpm,
        get_model_tokenizer_with_flash_attn,
        architectures=['MiniCPMForCausalLM'],
        model_arch=ModelArch.llama,
        requires=['transformers>=4.36.0'],
    ))

register_model(
    ModelMeta(
        LLMModelType.minicpm_chatml,
        [
            ModelGroup([
                Model('OpenBMB/MiniCPM-2B-128k', 'openbmb/MiniCPM-2B-128k'),
            ]),
            ModelGroup([
                Model('OpenBMB/MiniCPM4-0.5B', 'openbmb/MiniCPM4-0.5B'),
                Model('OpenBMB/MiniCPM4-8B', 'openbmb/MiniCPM4-8B'),
            ]),
        ],
        TemplateType.chatml,
        get_model_tokenizer_with_flash_attn,
        architectures=['MiniCPMForCausalLM'],
        model_arch=ModelArch.llama,
        requires=['transformers>=4.36'],
    ))

register_model(
    ModelMeta(
        LLMModelType.minicpm3,
        [
            ModelGroup([
                Model('OpenBMB/MiniCPM3-4B', 'openbmb/MiniCPM3-4B'),
            ]),
        ],
        TemplateType.chatml,
        get_model_tokenizer_with_flash_attn,
        architectures=['MiniCPM3ForCausalLM'],
        model_arch=ModelArch.deepseek_v2,
        requires=['transformers>=4.36'],
    ))
