# Copyright (c) ModelScope Contributors. All rights reserved.
from types import MethodType
from typing import Any, Dict

from transformers import PreTrainedModel
from transformers.utils import strtobool

from swift.template import TemplateType
from swift.utils import get_env_args
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..patcher import patch_device_map, patch_fixed_device, patch_output_clone
from ..register import ModelLoader, register_model
from ..utils import use_submodel_func
from .deepseek import DeepseekLoader

register_model(
    ModelMeta(
        LLMModelType.minicpm_moe,
        [
            ModelGroup([
                Model('OpenBMB/MiniCPM-MoE-8x2B', 'openbmb/MiniCPM-MoE-8x2B'),
            ]),
        ],
        DeepseekLoader,
        template=TemplateType.minicpm,
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
            if isinstance(output, list):
                return [x.to(device=device) for x in output]
            else:
                return output.to(device=device)

        model.__class__._old_get_vision_embedding = _old_get_vision_embedding
        model.__class__.get_vision_embedding = _get_vision_embedding

    if hasattr(model, 'resampler'):  # minicpm-v-v2_5-chat
        patch_fixed_device(model.resampler, device)


class MiniCPMVLoader(ModelLoader):

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, config, processor, model_kwargs)
        model.resampler.to(self.torch_dtype)  # fix float32
        _patch_minicpmv_device_map(model)
        func_list = ['generate', 'get_input_embeddings', 'forward']
        use_submodel_func(model, 'llm', func_list)
        if hasattr(model, 'get_slice_image_placeholder'):
            processor.get_slice_image_placeholder = MethodType(model.get_slice_image_placeholder, processor)
            processor.transform = MethodType(model.transform, processor)
        return model


register_model(
    ModelMeta(
        MLLMModelType.minicpmv,
        [
            ModelGroup([
                Model('OpenBMB/MiniCPM-V', 'openbmb/MiniCPM-V'),
                Model('OpenBMB/MiniCPM-V-2', 'openbmb/MiniCPM-V-2'),
            ], ),
        ],
        MiniCPMVLoader,
        template=TemplateType.minicpmv,
        architectures=['MiniCPMV'],
        model_arch=ModelArch.minicpmv,
        requires=['timm', 'transformers<4.42'],
        tags=['vision'],
    ))


class MiniCPMV2Loader(MiniCPMVLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        with patch_device_map():
            model = super().get_model(model_dir, *args, **kwargs)
        embedding = model.get_input_embeddings()
        patch_output_clone(embedding)
        return model


register_model(
    ModelMeta(
        MLLMModelType.minicpmv2_5,
        [
            ModelGroup([
                Model('OpenBMB/MiniCPM-Llama3-V-2_5', 'openbmb/MiniCPM-Llama3-V-2_5'),
            ], ),
        ],
        MiniCPMV2Loader,
        template=TemplateType.minicpmv2_5,
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
        MiniCPMV2Loader,
        template=TemplateType.minicpmv2_6,
        architectures=['MiniCPMV'],
        model_arch=ModelArch.minicpmv,
        requires=['timm', 'transformers>=4.36', 'decord'],
        tags=['vision', 'video'],
    ))


class MiniCPMO2Loader(MiniCPMV2Loader):

    def get_model(self, model_dir: str, config, *args, **kwargs) -> PreTrainedModel:
        config.init_tts = strtobool(get_env_args('init_tts', str, 'false'))
        config.init_audio = strtobool(get_env_args('init_audio', str, 'false'))
        return super().get_model(model_dir, config, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.minicpmo,
        [
            ModelGroup([
                Model('OpenBMB/MiniCPM-o-2_6', 'openbmb/MiniCPM-o-2_6'),
            ], template=TemplateType.minicpmo),
            ModelGroup(
                [
                    Model('OpenBMB/MiniCPM-o-4_5', 'openbmb/MiniCPM-o-4_5'),
                ],
                template=TemplateType.minicpmo4_5,
                requires=['timm', 'transformers==4.51.3', 'decord', 'soundfile'],
            ),
        ],
        MiniCPMO2Loader,
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
        MiniCPMV2Loader,
        template=TemplateType.minicpmv4,
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
        MiniCPMV2Loader,
        template=TemplateType.minicpmv4_5,
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
        template=TemplateType.minicpm,
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
        template=TemplateType.chatml,
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
        template=TemplateType.chatml,
        architectures=['MiniCPM3ForCausalLM'],
        model_arch=ModelArch.deepseek_v2,
        requires=['transformers>=4.36'],
    ))
