# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from functools import partial, wraps
from typing import Any, Dict

from transformers import AutoConfig, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.llm import TemplateType
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model
from ..utils import git_clone_github, safe_snapshot_download


class LlavaLlamaHfLoader(ModelLoader):

    def get_config(self, model_dir: str):
        from transformers import LlavaConfig
        self.autoconfig_class = LlavaConfig
        return super().get_config(model_dir)

    def get_model(self, model_dir: str, config, model_kwargs) -> PreTrainedModel:
        from transformers import LlavaForConditionalGeneration
        self.automodel_class = self.automodel_class or LlavaForConditionalGeneration
        return super().get_model(model_dir, config, model_kwargs)


register_model(
    ModelMeta(
        MLLMModelType.llava_llama3_hf,
        [
            ModelGroup([
                Model('AI-ModelScope/llava-llama-3-8b-v1_1-transformers', 'xtuner/llava-llama-3-8b-v1_1-transformers'),
            ]),
        ],
        LlavaLlamaHfLoader,
        template=TemplateType.llava_llama3_hf,
        architectures=['LlavaForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.36'],
        tags=['vision'],
    ))


def _patch_llava(model):
    if hasattr(model, '__old_generate'):
        return
    generate = model.generate
    model.__old_generate = generate

    @wraps(generate)
    def _new_generate(inputs=None, *args, **kwargs):
        input_ids = kwargs.pop('input_ids', None)
        if inputs is None and input_ids is not None:
            inputs = input_ids
        return generate(inputs, *args, **kwargs)

    model.generate = _new_generate


class LlavahfLoader(ModelLoader):

    def get_model(self, model_dir: str, config, model_kwargs) -> PreTrainedModel:
        from transformers import LlavaForConditionalGeneration
        self.automodel_class = self.automodel_class or LlavaForConditionalGeneration
        return super().get_model(model_dir, config, model_kwargs)


register_model(
    ModelMeta(
        MLLMModelType.llava1_5_hf,
        [
            ModelGroup([
                Model('llava-hf/llava-1.5-7b-hf', 'llava-hf/llava-1.5-7b-hf'),
                Model('llava-hf/llava-1.5-13b-hf', 'llava-hf/llava-1.5-13b-hf'),
            ]),
        ],
        LlavahfLoader,
        template=TemplateType.llava1_5_hf,
        architectures=['LlavaForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.36'],
        tags=['vision'],
    ))


class LlavaOnevisionHfLoader(ModelLoader):

    def get_model(self, model_dir: str, config, model_kwargs) -> PreTrainedModel:
        from transformers import LlavaOnevisionForConditionalGeneration
        self.automodel_class = self.automodel_class or LlavaOnevisionForConditionalGeneration
        return super().get_model(model_dir, config, model_kwargs)


register_model(
    ModelMeta(
        MLLMModelType.llava_onevision_hf,
        [
            ModelGroup([
                Model('llava-hf/llava-onevision-qwen2-0.5b-ov-hf', 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf'),
                Model('llava-hf/llava-onevision-qwen2-7b-ov-hf', 'llava-hf/llava-onevision-qwen2-7b-ov-hf'),
                Model('llava-hf/llava-onevision-qwen2-72b-ov-hf', 'llava-hf/llava-onevision-qwen2-72b-ov-hf'),
            ], ),
        ],
        LlavaOnevisionHfLoader,
        template=TemplateType.llava_onevision_hf,
        architectures=['LlavaOnevisionForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.45'],
        tags=['vision', 'video'],
    ))


class LlavaNextHfLoader(ModelLoader):

    def get_model(self, model_dir: str, config, model_kwargs) -> PreTrainedModel:
        from transformers import LlavaNextForConditionalGeneration
        self.automodel_class = self.automodel_class or LlavaNextForConditionalGeneration
        return super().get_model(model_dir, config, model_kwargs)


register_model(
    ModelMeta(
        MLLMModelType.llava_next_qwen_hf,
        [
            ModelGroup([
                Model('llava-hf/llava-next-72b-hf', 'llava-hf/llava-next-72b-hf'),
                Model('llava-hf/llava-next-110b-hf', 'llava-hf/llava-next-110b-hf'),
            ], ),
        ],
        LlavaNextHfLoader,
        template=TemplateType.llava_next_qwen_hf,
        architectures=['LlavaNextForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.39'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.llama3_llava_next_hf,
        [
            ModelGroup([
                Model('llava-hf/llama3-llava-next-8b-hf', 'llava-hf/llama3-llava-next-8b-hf'),
            ], ),
        ],
        LlavaNextHfLoader,
        template=TemplateType.llama3_llava_next_hf,
        architectures=['LlavaNextForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.39'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.llava1_6_vicuna_hf,
        [
            ModelGroup([
                Model('llava-hf/llava-v1.6-vicuna-7b-hf', 'llava-hf/llava-v1.6-vicuna-7b-hf'),
                Model('llava-hf/llava-v1.6-vicuna-13b-hf', 'llava-hf/llava-v1.6-vicuna-13b-hf'),
            ], ),
        ],
        LlavaNextHfLoader,
        template=TemplateType.llava1_6_vicuna_hf,
        architectures=['LlavaNextForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.39'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.llava1_6_mistral_hf,
        [
            ModelGroup([
                Model('llava-hf/llava-v1.6-mistral-7b-hf', 'llava-hf/llava-v1.6-mistral-7b-hf'),
            ], ),
        ],
        LlavaNextHfLoader,
        template=TemplateType.llava1_6_mistral_hf,
        architectures=['LlavaNextForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.39'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.llava_llama3_1_hf,
        [
            ModelGroup([
                Model('swift/llava-llama3.1-8b'),
            ], ),
        ],
        LlavaNextHfLoader,
        template=TemplateType.llava_llama3_1_hf,
        architectures=['LlavaNextForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.41'],
        tags=['vision'],
    ))

# def get_model_tokenizer_llava_next_yi(*args, **kwargs):
#     model, tokenizer = get_model_tokenizer_llava_next(*args, **kwargs)
#     if model is not None:
#         model.config.image_token_index = 64003
#     return model, tokenizer

register_model(
    ModelMeta(
        MLLMModelType.llava1_6_yi_hf,
        [
            ModelGroup([
                Model('llava-hf/llava-v1.6-34b-hf', 'llava-hf/llava-v1.6-34b-hf'),
            ], ),
        ],
        LlavaNextHfLoader,
        template=TemplateType.llava1_6_yi_hf,
        architectures=['LlavaNextForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.39'],
        tags=['vision'],
    ))


class LlavaNextVideoHfLoader(ModelLoader):

    def get_model(self, model_dir: str, config, model_kwargs) -> PreTrainedModel:
        from transformers import LlavaNextVideoForConditionalGeneration
        self.automodel_class = self.automodel_class or LlavaNextVideoForConditionalGeneration
        return super().get_model(model_dir, config, model_kwargs)


register_model(
    ModelMeta(
        MLLMModelType.llava_next_video_hf,
        [
            ModelGroup([
                Model('llava-hf/LLaVA-NeXT-Video-7B-DPO-hf', 'llava-hf/LLaVA-NeXT-Video-7B-DPO-hf'),
                Model('llava-hf/LLaVA-NeXT-Video-7B-32K-hf', 'llava-hf/LLaVA-NeXT-Video-7B-32K-hf'),
                Model('llava-hf/LLaVA-NeXT-Video-7B-hf', 'llava-hf/LLaVA-NeXT-Video-7B-hf'),
            ], ),
        ],
        LlavaNextVideoHfLoader,
        template=TemplateType.llava_next_video_hf,
        architectures=['LlavaNextVideoForConditionalGeneration'],
        model_arch=ModelArch.llava_next_video_hf,
        requires=['transformers>=4.42', 'av'],
        tags=['video'],
    ))


def get_model_tokenizer_llava_next_video_yi(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_llava_next_video(*args, **kwargs)
    if model is not None:
        model.config.video_token_index = 64003
        model.config.image_token_index = 64004
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.llava_next_video_yi_hf,
        [
            ModelGroup([
                Model('llava-hf/LLaVA-NeXT-Video-34B-hf', 'llava-hf/LLaVA-NeXT-Video-34B-hf'),
            ], ),
        ],
        LlavaNextVideoHfLoader,
        template=TemplateType.llava_next_video_hf,
        architectures=['LlavaNextVideoForConditionalGeneration'],
        model_arch=ModelArch.llava_next_video_hf,
        requires=['transformers>=4.42', 'av'],
        tags=['video'],
    ))


class LlavaLoader(ModelLoader):
    llm_model_type = None

    def get_config(self, model_dir: str):
        return None

    def get_model(self, model_dir: str, config, model_kwargs) -> PreTrainedModel:
        local_repo_path = self.local_repo_path
        if not local_repo_path:
            if 'next' in self.llm_model_type:
                repo_path = 'https://github.com/LLaVA-VL/LLaVA-NeXT'
            else:
                repo_path = 'https://github.com/haotian-liu/LLaVA'
            local_repo_path = git_clone_github(repo_path)
        sys.path.append(local_repo_path)
        if self.llm_model_type == 'mistral':
            from llava.model import LlavaMistralForCausalLM, LlavaMistralConfig
            config = LlavaMistralConfig.from_pretrained(model_dir)
            automodel_class = LlavaMistralForCausalLM
        elif 'llama' in self.llm_model_type:  # llama
            from llava.model import LlavaLlamaForCausalLM, LlavaConfig
            if not hasattr(LlavaLlamaForCausalLM, '__old_forward'):  # Avoid double patching
                forward = LlavaLlamaForCausalLM.forward
                LlavaLlamaForCausalLM.__old_forward = forward

                @wraps(forward)
                def _new_forward(*args, **kwargs):
                    kwargs.pop('cache_position', None)
                    return forward(*args, **kwargs)

                LlavaLlamaForCausalLM.forward = _new_forward
            config = LlavaConfig.from_pretrained(model_dir)
            automodel_class = LlavaLlamaForCausalLM
        else:  # qwen
            from llava.model import LlavaQwenForCausalLM
            automodel_class = LlavaQwenForCausalLM
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

        config.mm_vision_tower = safe_snapshot_download('AI-ModelScope/clip-vit-large-patch14-336', check_local=True)
        self.automodel_class = self.automodel_class or automodel_class
        model = super().get_model(model_dir, config, model_kwargs)
        vision_tower = model.get_vision_tower()
        device_map = str(model_kwargs.get('device_map', str(model.device)))
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if not hasattr(model.config, 'max_sequence_length'):
            model.config.max_sequence_length = 2048
        _patch_llava(model)
        return model


def get_model_tokenizer_llava(model_dir: str,
                              model_info: ModelInfo,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
        tokenizer.image_processor = vision_tower.image_processor
    return model, tokenizer


class Llama3LlavaNextLoader(LlavaLoader):
    llm_model_type = 'next_llama'


register_model(
    ModelMeta(
        MLLMModelType.llama3_llava_next,
        [
            ModelGroup([
                Model('AI-ModelScope/llama3-llava-next-8b', 'lmms-lab/llama3-llava-next-8b'),
            ], ),
        ],
        Llama3LlavaNextLoader,
        template=TemplateType.llama3_llava_next,
        architectures=['LlavaLlamaForCausalLM'],
        model_arch=ModelArch.llava_llama,
        requires=['transformers>=4.42', 'av'],
        tags=['vision'],
    ))


class LlavaMistralLoader(LlavaLoader):
    llm_model_type = 'next_llama'


register_model(
    ModelMeta(
        MLLMModelType.llava1_6_mistral,
        [
            ModelGroup([
                Model('AI-ModelScope/llava-v1.6-mistral-7b', 'liuhaotian/llava-v1.6-mistral-7b'),
            ], ),
        ],
        LlavaMistralLoader,
        template=TemplateType.llava1_6_mistral,
        requires=['transformers>=4.34'],
        architectures=['LlavaMistralForCausalLM'],
        model_arch=ModelArch.llava_mistral,
        tags=['vision'],
    ))


class LlavaLlamaLoader(LlavaLoader):
    llm_model_type = 'llama'


register_model(
    ModelMeta(
        MLLMModelType.llava1_6_yi, [
            ModelGroup([
                Model('AI-ModelScope/llava-v1.6-34b', 'liuhaotian/llava-v1.6-34b'),
            ], ),
        ],
        LlavaLlamaLoader,
        template=TemplateType.llava1_6_yi,
        requires=['transformers>=4.34'],
        architectures=['LlavaLlamaForCausalLM'],
        tags=['vision'],
        model_arch=None))


class LlavaNextQwenLoader(LlavaLoader):
    llm_model_type = 'next_qwen'


register_model(
    ModelMeta(
        MLLMModelType.llava_next_qwen, [
            ModelGroup([
                Model('AI-ModelScope/llava-next-72b', 'lmms-lab/llava-next-72b'),
                Model('AI-ModelScope/llava-next-110b', 'lmms-lab/llava-next-110b'),
            ], ),
        ],
        LlavaNextQwenLoader,
        template=TemplateType.llava_next_qwen,
        architectures=['LlavaQwenForCausalLM'],
        requires=['transformers>=4.42', 'av'],
        tags=['vision'],
        model_arch=None))


class LlavaOnevisionLoader(ModelLoader):

    def get_model(self, model_dir: str, config, model_kwargs) -> PreTrainedModel:
        model_cls = get_class_from_dynamic_module(
            'modeling_llavaonevision1_5.LLaVAOneVision1_5_ForConditionalGeneration', model_dir)
        model_cls._no_split_modules = ['LLaVAOneVision1_5_DecoderLayer', 'RiceBlock']
        config.vision_start_token_id = 151652
        return super().get_model(model_dir, config, model_kwargs)


register_model(
    ModelMeta(
        MLLMModelType.llava_onevision1_5,
        [
            ModelGroup([
                Model('lmms-lab/LLaVA-OneVision-1.5-4B-Instruct', 'lmms-lab/LLaVA-OneVision-1.5-4B-Instruct'),
                Model('lmms-lab/LLaVA-OneVision-1.5-8B-Instruct', 'lmms-lab/LLaVA-OneVision-1.5-8B-Instruct'),
                Model('lmms-lab/LLaVA-OneVision-1.5-4B-Base', 'lmms-lab/LLaVA-OneVision-1.5-4B-Base'),
                Model('lmms-lab/LLaVA-OneVision-1.5-8B-Base', 'lmms-lab/LLaVA-OneVision-1.5-8B-Base'),
            ], ),
        ],
        LlavaOnevisionLoader,
        template=TemplateType.llava_onevision1_5,
        architectures=['LLaVAOneVision1_5_ForConditionalGeneration'],
        model_arch=ModelArch.llava_onevision1_5,
        requires=['transformers>=4.53.0', 'qwen_vl_utils'],
        tags=['vision'],
    ))
