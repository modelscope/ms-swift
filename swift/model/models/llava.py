# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
from functools import wraps

from transformers import PretrainedConfig, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.template import TemplateType
from swift.utils import git_clone_github, safe_snapshot_download
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model


class LlavaLlamaHfLoader(ModelLoader):

    def get_config(self, model_dir: str):
        from transformers import LlavaConfig
        self.autoconfig_class = LlavaConfig
        return super().get_config(model_dir)

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import LlavaForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or LlavaForConditionalGeneration
        return super().get_model(model_dir, *args, **kwargs)


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

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import LlavaForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or LlavaForConditionalGeneration
        return super().get_model(model_dir, *args, **kwargs)


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

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import LlavaOnevisionForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or LlavaOnevisionForConditionalGeneration
        return super().get_model(model_dir, *args, **kwargs)


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

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import LlavaNextForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or LlavaNextForConditionalGeneration
        return super().get_model(model_dir, *args, **kwargs)


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


class LlavaNextYiHfLoader(LlavaNextHfLoader):

    def get_config(self, model_dir: str) -> PretrainedConfig:
        config = super().get_config(model_dir)
        config.image_token_index = 64003
        return config


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

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import LlavaNextVideoForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or LlavaNextVideoForConditionalGeneration
        return super().get_model(model_dir, *args, **kwargs)


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


class LlavaNextVideoYiHfLoader(LlavaNextVideoHfLoader):

    def get_config(self, model_dir: str) -> PretrainedConfig:
        config = super().get_config(model_dir)
        config.video_token_index = 64003
        config.image_token_index = 64004
        return config


register_model(
    ModelMeta(
        MLLMModelType.llava_next_video_yi_hf,
        [
            ModelGroup([
                Model('llava-hf/LLaVA-NeXT-Video-34B-hf', 'llava-hf/LLaVA-NeXT-Video-34B-hf'),
            ], ),
        ],
        LlavaNextVideoYiHfLoader,
        template=TemplateType.llava_next_video_hf,
        architectures=['LlavaNextVideoForConditionalGeneration'],
        model_arch=ModelArch.llava_next_video_hf,
        requires=['transformers>=4.42', 'av'],
        tags=['video'],
    ))


class LlavaLoader(ModelLoader):
    llm_model_type = None

    def get_config(self, model_dir: str):
        local_repo_path = self.local_repo_path
        if not local_repo_path:
            if 'next' in self.llm_model_type:
                repo_path = 'https://github.com/LLaVA-VL/LLaVA-NeXT'
            else:
                repo_path = 'https://github.com/haotian-liu/LLaVA'
            local_repo_path = git_clone_github(repo_path)
        sys.path.append(local_repo_path)
        if self.llm_model_type == 'mistral':
            from llava.model import LlavaMistralConfig
            self.auto_config_cls = LlavaMistralConfig
        elif 'llama' in self.llm_model_type:  # llama
            from llava.model import LlavaConfig
            self.auto_config_cls = LlavaConfig
        config = super().get_config(model_dir)
        if not hasattr(config, 'max_sequence_length'):
            config.max_sequence_length = 2048
        return config

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        if self.llm_model_type == 'mistral':
            from llava.model import LlavaMistralForCausalLM
            auto_model_cls = LlavaMistralForCausalLM
        elif 'llama' in self.llm_model_type:  # llama
            from llava.model import LlavaLlamaForCausalLM
            if not hasattr(LlavaLlamaForCausalLM, '__old_forward'):  # Avoid double patching
                forward = LlavaLlamaForCausalLM.forward
                LlavaLlamaForCausalLM.__old_forward = forward

                @wraps(forward)
                def _new_forward(*args, **kwargs):
                    kwargs.pop('cache_position', None)
                    return forward(*args, **kwargs)

                LlavaLlamaForCausalLM.forward = _new_forward
            auto_model_cls = LlavaLlamaForCausalLM
        else:  # qwen
            from llava.model import LlavaQwenForCausalLM
            auto_model_cls = LlavaQwenForCausalLM

        config.mm_vision_tower = safe_snapshot_download('AI-ModelScope/clip-vit-large-patch14-336', check_local=True)
        self.auto_model_cls = self.auto_model_cls or auto_model_cls
        model = super().get_model(model_dir, config, processor, model_kwargs)
        vision_tower = model.get_vision_tower()
        device_map = str(model_kwargs.get('device_map', str(model.device)))
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        _patch_llava(model)
        model.resize_token_embeddings(len(processor))
        processor.image_processor = vision_tower.image_processor
        return model


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

    def get_config(self, model_dir: str) -> PretrainedConfig:
        config = super().get_config(model_dir)
        config.vision_start_token_id = 151652
        return config

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        model_cls = get_class_from_dynamic_module(
            'modeling_llavaonevision1_5.LLaVAOneVision1_5_ForConditionalGeneration', model_dir)
        model_cls._no_split_modules = ['LLaVAOneVision1_5_DecoderLayer', 'RiceBlock']
        return super().get_model(model_dir, *args, **kwargs)


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
