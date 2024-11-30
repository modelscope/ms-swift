# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from typing import Any, Dict

from swift.llm import TemplateType
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..patcher import patch_output_clone, patch_output_to_input_device
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo, git_clone_github, use_submodel_func


def get_model_tokenizer_deepseek_moe(model_dir: str,
                                     model_info: ModelInfo,
                                     model_kwargs: Dict[str, Any],
                                     load_model: bool = True,
                                     **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if model is not None:
        # fix dtype bug
        mlp_cls = model.model.layers[1].mlp.__class__

        def _dtype_hook(module, input, output):
            return output.to(input[0].dtype)

        for module in model.modules():
            if isinstance(module, mlp_cls):
                module.register_forward_hook(_dtype_hook)
    return model, tokenizer


def get_model_tokenizer_deepseek2(model_dir: str,
                                  model_info: ModelInfo,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    model, tokenizer = get_model_tokenizer_deepseek_moe(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if model is not None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.deepseek_moe,
        [
            ModelGroup(
                [
                    Model('deepseek-ai/deepseek-moe-16b-chat', 'deepseek-ai/deepseek-moe-16b-chat'),
                    Model('deepseek-ai/deepseek-moe-16b-base', 'deepseek-ai/deepseek-moe-16b-base'),
                ],
                tags=['moe'],
            ),
        ],
        TemplateType.deepseek,
        get_model_tokenizer_deepseek_moe,
        architectures=['DeepseekForCausalLM'],
        model_arch=ModelArch.llama,
    ))

register_model(
    ModelMeta(
        LLMModelType.deepseek,
        [
            ModelGroup(
                [
                    Model('deepseek-ai/DeepSeek-Coder-V2-Instruct', 'deepseek-ai/DeepSeek-Coder-V2-Instruct'),
                    Model('deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'),
                    Model('deepseek-ai/DeepSeek-Coder-V2-Base', 'deepseek-ai/DeepSeek-Coder-V2-Base'),
                    Model('deepseek-ai/DeepSeek-Coder-V2-Lite-Base', 'deepseek-ai/DeepSeek-Coder-V2-Lite-Base'),
                    Model('deepseek-ai/DeepSeek-V2-Lite', 'deepseek-ai/DeepSeek-V2-Lite'),
                    Model('deepseek-ai/DeepSeek-V2-Lite-Chat', 'deepseek-ai/DeepSeek-V2-Lite-Chat'),
                    Model('deepseek-ai/DeepSeek-V2', 'deepseek-ai/DeepSeek-V2'),
                    Model('deepseek-ai/DeepSeek-V2-Chat', 'deepseek-ai/DeepSeek-V2-Chat'),
                ],
                requires=['transformers>=4.39.3'],
                tags=['moe', 'skip_test'],
            ),
        ],
        TemplateType.deepseek,
        get_model_tokenizer_deepseek_moe,
        architectures=['DeepseekV2ForCausalLM'],
        model_arch=ModelArch.deepseek_v2,
    ))

register_model(
    ModelMeta(
        LLMModelType.deepseek2_5,
        [
            ModelGroup(
                [
                    Model('deepseek-ai/DeepSeek-V2.5', 'deepseek-ai/DeepSeek-V2.5'),
                ],
                requires=['transformers>=4.39.3'],
                tags=['moe', 'skip_test'],
            ),
        ],
        TemplateType.deepseek2_5,
        get_model_tokenizer_deepseek_moe,
        architectures=['DeepseekV2ForCausalLM'],
        model_arch=ModelArch.deepseek_v2,
    ))


def get_model_tokenizer_deepseek_vl(model_dir: str,
                                    model_info: ModelInfo,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    # compat with python==3.10
    if sys.version_info.minor >= 10:
        import collections
        import collections.abc
        for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))
    if 'local_repo_path' in kwargs:
        local_repo_path = kwargs['local_repo_path']
    else:
        local_repo_path = git_clone_github('https://github.com/deepseek-ai/DeepSeek-VL')
    sys.path.append(os.path.join(local_repo_path))
    from deepseek_vl.models import VLChatProcessor
    processor = VLChatProcessor.from_pretrained(model_dir)
    tokenizer = processor.tokenizer

    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)
    if load_model:
        patch_output_clone(model.language_model.model.embed_tokens)
        patch_output_to_input_device(model.language_model.model.embed_tokens)
        func_list = ['generate', 'get_input_embeddings', 'gradient_checkpointing_enable', 'forward']
        use_submodel_func(model, 'language_model', func_list)
        model.generation_config = model.language_model.generation_config
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.deepseek_vl, [
            ModelGroup(
                [
                    Model('deepseek-ai/deepseek-vl-1.3b-chat', 'deepseek-ai/deepseek-vl-1.3b-chat'),
                    Model('deepseek-ai/deepseek-vl-7b-chat', 'deepseek-ai/deepseek-vl-7b-chat'),
                ],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.deepseek_vl,
        get_model_tokenizer_deepseek_vl,
        architectures=['MultiModalityCausalLM'],
        model_arch=ModelArch.deepseek_vl))

register_model(
    ModelMeta(
        LLMModelType.deepseek_math, [
            ModelGroup(
                [
                    Model('deepseek-ai/deepseek-math-7b-instruct', 'deepseek-ai/deepseek-math-7b-instruct'),
                    Model('deepseek-ai/deepseek-math-7b-base', 'deepseek-ai/deepseek-math-7b-base'),
                    Model('deepseek-ai/deepseek-math-7b-rl', 'deepseek-ai/deepseek-math-7b-rl'),
                ],
                tags=['math'],
            ),
            ModelGroup(
                [
                    Model('deepseek-ai/deepseek-coder-1.3b-instruct', 'deepseek-ai/deepseek-coder-1.3b-instruct'),
                    Model('deepseek-ai/deepseek-coder-1.3b-base', 'deepseek-ai/deepseek-coder-1.3b-base'),
                    Model('deepseek-ai/deepseek-coder-6.7b-base', 'deepseek-ai/deepseek-coder-6.7b-base'),
                    Model('deepseek-ai/deepseek-coder-6.7b-instruct', 'deepseek-ai/deepseek-coder-6.7b-instruct'),
                    Model('deepseek-ai/deepseek-coder-33b-base', 'deepseek-ai/deepseek-coder-33b-base'),
                    Model('deepseek-ai/deepseek-coder-33b-instruct', 'deepseek-ai/deepseek-coder-33b-instruct'),
                ],
                tags=['coding'],
            ),
            ModelGroup([
                Model('deepseek-ai/deepseek-llm-7b-chat', 'deepseek-ai/deepseek-llm-7b-chat'),
                Model('deepseek-ai/deepseek-llm-7b-base', 'deepseek-ai/deepseek-llm-7b-base'),
                Model('deepseek-ai/deepseek-llm-67b-base', 'deepseek-ai/deepseek-llm-67b-base'),
                Model('deepseek-ai/deepseek-llm-67b-chat', 'deepseek-ai/deepseek-llm-67b-chat'),
            ], ),
        ],
        TemplateType.deepseek,
        get_model_tokenizer_with_flash_attn,
        architectures=['LlamaForCausalLM'],
        model_arch=ModelArch.llama))


def get_model_tokenizer_deepseek_janus(model_dir: str, *args, **kwargs):
    if 'local_repo_path' in kwargs:
        local_repo_path = kwargs['local_repo_path']
    else:
        local_repo_path = git_clone_github('https://github.com/deepseek-ai/Janus')
    sys.path.append(os.path.join(local_repo_path))
    from janus.models import MultiModalityCausalLM, VLChatProcessor

    processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_dir)
    tokenizer = processor.tokenizer
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, *args, tokenizer=tokenizer, **kwargs)
    if model:
        model.language_model.model.embed_tokens.register_forward_hook(patch_output_clone)
        model.language_model.model.embed_tokens.register_forward_hook(patch_output_to_input_device)
        func_list = ['generate', 'get_input_embeddings', 'forward', 'gradient_checkpointing_enable']
        use_submodel_func(model, 'language_model', func_list)
        model.generation_config = model.language_model.generation_config
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.janus, [
            ModelGroup(
                [
                    Model('deepseek-ai/Janus-1.3B', 'deepseek-ai/Janus-1.3B'),
                ],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.janus,
        get_model_tokenizer_deepseek_janus,
        model_arch=ModelArch.janus))
