# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from functools import partial, wraps
from typing import Any, Dict

from modelscope import AutoConfig
from transformers import PretrainedConfig

from swift.llm import TemplateType
from ..constant import MLLMModelType
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo, git_clone_github, safe_snapshot_download


def get_model_tokenizer_llava_llama(model_dir: str,
                                    model_info: ModelInfo,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    from transformers import LlavaForConditionalGeneration, LlavaConfig, AutoProcessor

    model_config = LlavaConfig.from_pretrained(model_dir)  # check
    processor = AutoProcessor.from_pretrained(model_dir)
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir,
        model_info,
        model_kwargs,
        load_model,
        model_config=model_config,
        automodel_class=LlavaForConditionalGeneration,
        **kwargs)
    tokenizer.processor = processor
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.llava_llama,
        [
            ModelGroup([
                Model('AI-ModelScope/llava-llama-3-8b-v1_1-transformers', 'xtuner/llava-llama-3-8b-v1_1-transformers'),
            ],
                       requires=['transformers>=4.36'],
                       tags=['multi-modal', 'vision']),
        ],
        TemplateType.llava_llama_instruct,
        get_model_tokenizer_llava_llama,
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=True,
        support_lmdeploy=True,
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
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir)
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    tokenizer.processor = processor
    return model, tokenizer


def get_model_tokenizer_llava_1_5(*args, **kwargs):
    from transformers import LlavaForConditionalGeneration
    kwargs['automodel_class'] = LlavaForConditionalGeneration
    return get_model_tokenizer_llava_hf(*args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.llava1_5,
        [
            ModelGroup([
                Model('swift/llava-1.5-13b-hf', 'llava-hf/llava-1.5-13b-hf'),
                Model('swift/llava-1.5-7b-hf', 'llava-hf/llava-1.5-7b-hf'),
            ],
                       requires=['transformers>=4.36'],
                       tags=['multi-modal', 'vision']),
        ],
        TemplateType.llava1_5,
        get_model_tokenizer_llava_1_5,
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=True,
    ))


def get_model_tokenizer_llava_onevision(*args, **kwargs):
    from transformers import LlavaOnevisionForConditionalGeneration
    kwargs['automodel_class'] = LlavaOnevisionForConditionalGeneration
    return get_model_tokenizer_llava_hf(*args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.llava1_5,
        [
            ModelGroup(
                [
                    Model('AI-ModelScope/llava-onevision-qwen2-0.5b-ov-hf',
                          'llava-hf/llava-onevision-qwen2-0.5b-ov-hf'),
                    Model('AI-ModelScope/llava-onevision-qwen2-7b-ov-hf', 'llava-hf/llava-onevision-qwen2-7b-ov-hf'),
                    Model('AI-ModelScope/llava-onevision-qwen2-72b-ov-hf', 'llava-hf/llava-onevision-qwen2-72b-ov-hf'),
                ],
                requires=['transformers>=4.45.0.dev0'],
                tags=['multi-modal', 'vision', 'video'],
                ignore_file_pattern=['onnx'],
            ),
        ],
        TemplateType.llava_onevision_qwen,
        get_model_tokenizer_llava_onevision,
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=True,
    ))

register_model(
    ModelMeta(
        MLLMModelType.llava_next,
        [
            ModelGroup(
                [
                    Model('AI-ModelScope/llava-next-72b-hf', 'llava-hf/llava-next-72b-hf'),
                    Model('AI-ModelScope/llava-next-110b-hf', 'llava-hf/llava-next-110b-hf'),
                ],
                requires=['transformers>=4.39'],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.llava_qwen_hf,
        get_model_tokenizer_llava_onevision,
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=True,
    ))


def get_model_tokenizer_llava_next(*args, **kwargs):
    from transformers import LlavaNextForConditionalGeneration
    kwargs['automodel_class'] = LlavaNextForConditionalGeneration
    return get_model_tokenizer_llava_hf(*args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.llama3_llava_next_hf,
        [
            ModelGroup(
                [
                    Model('swift/llama3-llava-next-8b-hf', 'llava-hf/llama3-llava-next-8b-hf'),
                ],
                requires=['transformers>=4.39'],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.llama3_llava_next_hf,
        get_model_tokenizer_llava_next,
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=True,
    ))

register_model(
    ModelMeta(
        MLLMModelType.llava1_6_vicuna,
        [
            ModelGroup(
                [
                    Model('swift/llava-v1.6-vicuna-7b-hf', 'llava-hf/llava-v1.6-vicuna-7b-hf'),
                    Model('swift/llava-v1.6-vicuna-13b-hf', 'llava-hf/llava-v1.6-vicuna-13b-hf'),
                ],
                requires=['transformers>=4.39'],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.llava_vicuna,
        get_model_tokenizer_llava_next,
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=True,
    ))

register_model(
    ModelMeta(
        MLLMModelType.llava1_6_mistral,
        [
            ModelGroup(
                [
                    Model('swift/llava-v1.6-mistral-7b-hf', 'llava-hf/llava-v1.6-mistral-7b-hf'),
                ],
                requires=['transformers>=4.39'],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.llava_mistral,
        get_model_tokenizer_llava_next,
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=True,
    ))

register_model(
    ModelMeta(
        MLLMModelType.llava1_6_llama3_1,
        [
            ModelGroup(
                [
                    Model('DaozeZhang/llava-llama3.1-8b'),
                ],
                requires=['transformers>=4.41'],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.llava_next_llama3,
        get_model_tokenizer_llava_next,
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=False,
    ))


def get_model_tokenizer_llava_next_yi(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_llava_next(*args, **kwargs)
    if model is not None:
        model.config.image_token_index = 64003
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.llava1_6_yi,
        [
            ModelGroup(
                [
                    Model('swift/llava-v1.6-34b-hf', 'llava-hf/llava-v1.6-34b-hf'),
                ],
                requires=['transformers>=4.39'],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.llava_yi,
        get_model_tokenizer_llava_next_yi,
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=True,
    ))


def get_model_tokenizer_llava_next_video(*args, **kwargs):
    from transformers import LlavaNextVideoForConditionalGeneration
    kwargs['automodel_class'] = LlavaNextVideoForConditionalGeneration
    return get_model_tokenizer_llava_hf(*args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.llava_next_video,
        [
            ModelGroup(
                [
                    Model('swift/LLaVA-NeXT-Video-7B-DPO-hf', 'llava-hf/LLaVA-NeXT-Video-7B-DPO-hf'),
                    Model('swift/LLaVA-NeXT-Video-7B-32K-hf', 'llava-hf/LLaVA-NeXT-Video-7B-32K-hf'),
                    Model('swift/LLaVA-NeXT-Video-7B-hf', 'llava-hf/LLaVA-NeXT-Video-7B-hf'),
                ],
                requires=['transformers>=4.42', 'av'],
                tags=['multi-modal', 'video'],
            ),
        ],
        TemplateType.llava_next_video,
        get_model_tokenizer_llava_next_video,
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
    ))


def get_model_tokenizer_llava_next_video_yi(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_llava_next_video(*args, **kwargs)
    if model is not None:
        model.config.video_token_index = 64003
        model.config.image_token_index = 64004
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.llava_next_video_yi,
        [
            ModelGroup(
                [
                    Model('swift/LLaVA-NeXT-Video-34B-hf', 'llava-hf/LLaVA-NeXT-Video-34B-hf'),
                ],
                requires=['transformers>=4.42', 'av'],
                tags=['multi-modal', 'video'],
            ),
        ],
        TemplateType.llava_next_video_yi,
        get_model_tokenizer_llava_next_video_yi,
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
    ))


def get_model_tokenizer_llava(model_dir: str,
                              config: PretrainedConfig,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    llm_model_type = kwargs.pop('llm_model_type')
    if 'local_repo_path' in kwargs:
        local_repo_path = kwargs['local_repo_path']
    elif 'next' in llm_model_type:
        repo_path = 'https://github.com/LLaVA-VL/LLaVA-NeXT'
        local_repo_path = git_clone_github(repo_path)
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
        model_config = AutoConfig.from_pretrained(model_dir)

    model_config.mm_vision_tower = safe_snapshot_download('AI-ModelScope/clip-vit-large-patch14-336')
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir, model_config, model_kwargs, load_model, automodel_class=automodel_class, **kwargs)

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
            ModelGroup(
                [
                    Model('AI-Modelscope/llama3-llava-next-8b', 'lmms-lab/llama3-llava-next-8b'),
                ],
                requires=['transformers>=4.42', 'av'],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.llama3_llava_next,
        partial(get_model_tokenizer_llava, llm_model_type='next_llama'),
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
    ))

register_model(
    ModelMeta(
        MLLMModelType.llama3_llava_next,
        [
            ModelGroup(
                [
                    Model('AI-Modelscope/llava-next-72b', 'lmms-lab/llava-next-72b'),
                    Model('AI-Modelscope/llava-next-110b', 'lmms-lab/llava-next-110b'),
                ],
                requires=['transformers>=4.42', 'av'],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.llava_qwen,
        partial(get_model_tokenizer_llava, llm_model_type='next_qwen'),
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
    ))

register_model(
    ModelMeta(
        MLLMModelType.llama3_llava_next,
        [
            ModelGroup(
                [
                    Model('AI-ModelScope/llava-onevision-qwen2-72b-ov-hf', 'llava-hf/llava-onevision-qwen2-72b-ov-hf'),
                ],
                requires=['transformers>=4.45'],
                tags=['multi-modal', 'vision', 'video'],
            ),
        ],
        TemplateType.llava_onevision_qwen,
        get_model_tokenizer_llava_onevision,
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
    ))
