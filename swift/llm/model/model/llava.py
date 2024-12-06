# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from functools import partial, wraps
from typing import Any, Dict

from transformers import AutoConfig

from swift.llm import TemplateType
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import ModelInfo, git_clone_github, safe_snapshot_download


def get_model_tokenizer_llava_llama(model_dir: str,
                                    model_info: ModelInfo,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    from transformers import LlavaForConditionalGeneration, LlavaConfig

    kwargs['model_config'] = LlavaConfig.from_pretrained(model_dir)
    kwargs['automodel_class'] = kwargs['automodel_class'] or LlavaForConditionalGeneration
    model, processor = get_model_tokenizer_multimodal(model_dir, model_info, model_kwargs, load_model, **kwargs)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.llava_llama3_hf,
        [
            ModelGroup([
                Model('AI-ModelScope/llava-llama-3-8b-v1_1-transformers', 'xtuner/llava-llama-3-8b-v1_1-transformers'),
            ]),
        ],
        TemplateType.llava_llama3_hf,
        get_model_tokenizer_llava_llama,
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


def get_model_tokenizer_llava_hf(model_dir: str, *args, **kwargs):
    from transformers import LlavaForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or LlavaForConditionalGeneration
    model, processor = get_model_tokenizer_multimodal(model_dir, *args, **kwargs)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.llava1_5_hf,
        [
            ModelGroup([
                Model('swift/llava-1.5-7b-hf', 'llava-hf/llava-1.5-7b-hf'),
                Model('swift/llava-1.5-13b-hf', 'llava-hf/llava-1.5-13b-hf'),
            ]),
        ],
        TemplateType.llava1_5_hf,
        get_model_tokenizer_llava_hf,
        architectures=['LlavaForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.36'],
        tags=['vision'],
    ))


def get_model_tokenizer_llava_onevision(*args, **kwargs):
    from transformers import LlavaOnevisionForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or LlavaOnevisionForConditionalGeneration
    return get_model_tokenizer_llava_hf(*args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.llava_onevision_hf,
        [
            ModelGroup([
                Model('AI-ModelScope/llava-onevision-qwen2-0.5b-ov-hf', 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf'),
                Model('AI-ModelScope/llava-onevision-qwen2-7b-ov-hf', 'llava-hf/llava-onevision-qwen2-7b-ov-hf'),
                Model('AI-ModelScope/llava-onevision-qwen2-72b-ov-hf', 'llava-hf/llava-onevision-qwen2-72b-ov-hf'),
            ], ),
        ],
        TemplateType.llava_onevision_hf,
        get_model_tokenizer_llava_onevision,
        architectures=['LlavaOnevisionForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.45'],
        tags=['vision', 'video'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.llava_next_qwen_hf,
        [
            ModelGroup([
                Model('AI-ModelScope/llava-next-72b-hf', 'llava-hf/llava-next-72b-hf'),
                Model('AI-ModelScope/llava-next-110b-hf', 'llava-hf/llava-next-110b-hf'),
            ], ),
        ],
        TemplateType.llava_next_qwen_hf,
        get_model_tokenizer_llava_onevision,
        architectures=['LlavaNextForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.39'],
        tags=['vision'],
    ))


def get_model_tokenizer_llava_next(*args, **kwargs):
    from transformers import LlavaNextForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or LlavaNextForConditionalGeneration
    return get_model_tokenizer_llava_hf(*args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.llama3_llava_next_hf,
        [
            ModelGroup([
                Model('swift/llama3-llava-next-8b-hf', 'llava-hf/llama3-llava-next-8b-hf'),
            ], ),
        ],
        TemplateType.llama3_llava_next_hf,
        get_model_tokenizer_llava_next,
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
                Model('swift/llava-v1.6-vicuna-7b-hf', 'llava-hf/llava-v1.6-vicuna-7b-hf'),
                Model('swift/llava-v1.6-vicuna-13b-hf', 'llava-hf/llava-v1.6-vicuna-13b-hf'),
            ], ),
        ],
        TemplateType.llava1_6_vicuna_hf,
        get_model_tokenizer_llava_next,
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
                Model('swift/llava-v1.6-mistral-7b-hf', 'llava-hf/llava-v1.6-mistral-7b-hf'),
            ], ),
        ],
        TemplateType.llava1_6_mistral_hf,
        get_model_tokenizer_llava_next,
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
        TemplateType.llava_llama3_1_hf,
        get_model_tokenizer_llava_next,
        architectures=['LlavaNextForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.41'],
        tags=['vision'],
    ))


def get_model_tokenizer_llava_next_yi(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_llava_next(*args, **kwargs)
    if model is not None:
        model.config.image_token_index = 64003
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.llava1_6_yi_hf,
        [
            ModelGroup([
                Model('swift/llava-v1.6-34b-hf', 'llava-hf/llava-v1.6-34b-hf'),
            ], ),
        ],
        TemplateType.llava1_6_yi_hf,
        get_model_tokenizer_llava_next_yi,
        architectures=['LlavaNextForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.39'],
        tags=['vision'],
    ))


def get_model_tokenizer_llava_next_video(*args, **kwargs):
    from transformers import LlavaNextVideoForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or LlavaNextVideoForConditionalGeneration
    return get_model_tokenizer_llava_hf(*args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.llava_next_video_hf,
        [
            ModelGroup([
                Model('swift/LLaVA-NeXT-Video-7B-DPO-hf', 'llava-hf/LLaVA-NeXT-Video-7B-DPO-hf'),
                Model('swift/LLaVA-NeXT-Video-7B-32K-hf', 'llava-hf/LLaVA-NeXT-Video-7B-32K-hf'),
                Model('swift/LLaVA-NeXT-Video-7B-hf', 'llava-hf/LLaVA-NeXT-Video-7B-hf'),
            ], ),
        ],
        TemplateType.llava_next_video_hf,
        get_model_tokenizer_llava_next_video,
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
                Model('swift/LLaVA-NeXT-Video-34B-hf', 'llava-hf/LLaVA-NeXT-Video-34B-hf'),
            ], ),
        ],
        TemplateType.llava_next_video_hf,
        get_model_tokenizer_llava_next_video_yi,
        architectures=['LlavaNextVideoForConditionalGeneration'],
        model_arch=ModelArch.llava_next_video_hf,
        requires=['transformers>=4.42', 'av'],
        tags=['video'],
    ))


def get_model_tokenizer_llava(model_dir: str,
                              model_info: ModelInfo,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    llm_model_type = kwargs.pop('llm_model_type')
    local_repo_path = kwargs.get('local_repo_path')
    if not local_repo_path:
        if 'next' in llm_model_type:
            repo_path = 'https://github.com/LLaVA-VL/LLaVA-NeXT'
        else:
            repo_path = 'https://github.com/haotian-liu/LLaVA'
        local_repo_path = git_clone_github(repo_path)
    sys.path.append(os.path.join(local_repo_path))

    if llm_model_type == 'mistral':
        from llava.model import LlavaMistralForCausalLM, LlavaMistralConfig
        model_config = LlavaMistralConfig.from_pretrained(model_dir)
        automodel_class = LlavaMistralForCausalLM
    elif 'llama' in llm_model_type:  # llama
        from llava.model import LlavaLlamaForCausalLM, LlavaConfig
        if not hasattr(LlavaLlamaForCausalLM, '__old_forward'):  # Avoid double patching
            forward = LlavaLlamaForCausalLM.forward
            LlavaLlamaForCausalLM.__old_forward = forward

            @wraps(forward)
            def _new_forward(*args, **kwargs):
                kwargs.pop('cache_position', None)
                return forward(*args, **kwargs)

            LlavaLlamaForCausalLM.forward = _new_forward
        model_config = LlavaConfig.from_pretrained(model_dir)
        automodel_class = LlavaLlamaForCausalLM
    else:  # qwen
        from llava.model import LlavaQwenForCausalLM
        automodel_class = LlavaQwenForCausalLM
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    model_config.mm_vision_tower = safe_snapshot_download('AI-ModelScope/clip-vit-large-patch14-336')
    kwargs['model_config'] = model_config
    kwargs['automodel_class'] = automodel_class
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)

    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        device_map = str(model_kwargs.get('device_map', str(model.device)))
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if not hasattr(model.config, 'max_sequence_length'):
            model.config.max_sequence_length = 2048
        _patch_llava(model)
        tokenizer.image_processor = vision_tower.image_processor
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.llama3_llava_next,
        [
            ModelGroup([
                Model('AI-Modelscope/llama3-llava-next-8b', 'lmms-lab/llama3-llava-next-8b'),
            ], ),
        ],
        TemplateType.llama3_llava_next,
        partial(get_model_tokenizer_llava, llm_model_type='next_llama'),
        architectures=['LlavaLlamaForCausalLM'],
        model_arch=ModelArch.llava_llama,
        requires=['transformers>=4.42', 'av'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.llava1_6_mistral,
        [
            ModelGroup([
                Model('AI-ModelScope/llava-v1.6-mistral-7b', 'liuhaotian/llava-v1.6-mistral-7b'),
            ], ),
        ],
        TemplateType.llava1_6_mistral,
        partial(get_model_tokenizer_llava, llm_model_type='mistral'),
        requires=['transformers>=4.34'],
        architectures=['LlavaMistralForCausalLM'],
        model_arch=ModelArch.llava_mistral,
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.llava1_6_yi, [
            ModelGroup([
                Model('AI-ModelScope/llava-v1.6-34b', 'liuhaotian/llava-v1.6-34b'),
            ], ),
        ],
        TemplateType.llava1_6_yi,
        partial(get_model_tokenizer_llava, llm_model_type='llama'),
        requires=['transformers>=4.34'],
        architectures=['LlavaLlamaForCausalLM'],
        tags=['vision'],
        model_arch=None))

register_model(
    ModelMeta(
        MLLMModelType.llava_next_qwen, [
            ModelGroup([
                Model('AI-Modelscope/llava-next-72b', 'lmms-lab/llava-next-72b'),
                Model('AI-Modelscope/llava-next-110b', 'lmms-lab/llava-next-110b'),
            ], ),
        ],
        TemplateType.llava_next_qwen,
        partial(get_model_tokenizer_llava, llm_model_type='next_qwen'),
        architectures=['LlavaQwenForCausalLM'],
        requires=['transformers>=4.42', 'av'],
        tags=['vision'],
        model_arch=None))
