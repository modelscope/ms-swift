# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from functools import partial
from typing import Any, Dict

from modelscope import AutoModelForCausalLM, AutoConfig
from transformers import AutoTokenizer, PretrainedConfig

from swift.llm import TemplateType
from .model import _use_submodel_func
from .qwen import get_model_tokenizer_qwen
from ..constant import LLMModelType, MLLMModelType
from ..register import (Model, ModelGroup, ModelMeta, register_model, get_model_tokenizer_from_local,
                        get_model_tokenizer_with_flash_attn)
from ..utils import git_clone_github


def get_model_tokenizer_grok(model_dir: str,
                             model_config: PretrainedConfig,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             tokenizer=None,
                             automodel_class=AutoModelForCausalLM,
                             **kwargs):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            'AI-ModelScope/grok-1-tokenizer', revision='master', trust_remote_code=True)
    eos_token = kwargs.get('eos_token')
    if eos_token is not None:
        tokenizer.eos_token = eos_token
    model = None
    if load_model:
        model = automodel_class.from_pretrained(
            model_dir, config=model_config, trust_remote_code=True, **model_kwargs)
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.grok,
        [
            # llama2
            ModelGroup(
                [
                    # base
                    Model('colossalai/grok-1-pytorch', 'hpcai-tech/grok-1'),
                ],
                requires=['transformers>=4.36'],
                tags=['multi-modal', 'vision'],
                ignore_file_pattern=[r'.+\.bin$']),
        ],
        TemplateType.default,
        get_model_tokenizer_grok,
        architectures=['LlavaForConditionalGeneration'],
        support_vllm=False,
        support_flash_attn=False,
    ))


def get_model_tokenizer_mplug_owl3(model_dir: str,
                                   model_config: PretrainedConfig,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_config, model_kwargs, load_model, **kwargs)
    processor = model.init_processor(tokenizer)
    tokenizer.processor = processor
    func_list = ['generate', 'forward']
    _use_submodel_func(model, 'language_model', func_list)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.mplug,
        [
            # llama2
            ModelGroup(
                [
                    # base
                    Model('iic/mPLUG-Owl3-7B-240728', 'mPLUG/mPLUG-Owl3-7B-240728'),
                ],
                requires=['transformers>=4.36', 'icecream'],
                tags=['multi-modal', 'vision'],
                ignore_file_pattern=[r'.+\.bin$']),
        ],
        TemplateType.mplug_owl3,
        get_model_tokenizer_mplug_owl3,
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
    ))


def get_model_tokenizer_polylm(model_dir: str,
                               config: PretrainedConfig,
                               model_kwargs: Dict[str, Any],
                               load_model: bool = True,
                               **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False, legacy=True)
    return get_model_tokenizer_from_local(
        model_dir, config, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.polylm,
        [
            ModelGroup(
                [
                    # base
                    Model('damo/nlp_polylm_13b_text_generation', 'DAMO-NLP-MT/polylm-13b'),
                ],),
        ],
        TemplateType.default,
        get_model_tokenizer_polylm,
        architectures=['LlavaForConditionalGeneration'],
        support_flash_attn=True,
    ))


def get_skywork_model_tokenizer(model_dir: str,
                                config: PretrainedConfig,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):
    model, tokenizer = get_model_tokenizer_from_local(model_dir, config, model_kwargs, load_model, **kwargs)
    if 'chat' in model_dir:
        tokenizer.add_tokens('[USER]')
        tokenizer.add_tokens('[BOT]')
        tokenizer.add_tokens('[SEP]')
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.skywork,
        [
            ModelGroup(
                [
                    Model('skywork/Skywork-13B-chat'),
                    Model('skywork/Skywork-13B-base'),
                ]),
        ],
        TemplateType.skywork,
        get_skywork_model_tokenizer,
        architectures=['LlavaForConditionalGeneration'],
    ))


def get_model_tokenizer_codellama(model_dir: str,
                                  config: PretrainedConfig,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False, legacy=False)
    return get_model_tokenizer_with_flash_attn(
        model_dir, config, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.codefuse_codellama,
        [
            ModelGroup(
                [
                    Model('codefuse-ai/CodeFuse-CodeLlama-34B', 'codefuse-ai/CodeFuse-CodeLlama-34B'),
                ],
            tags=['coding'],),
        ],
        TemplateType.codefuse_codellama,
        get_model_tokenizer_codellama,
        support_flash_attn=True,
        support_vllm=True,
        support_lmdeploy=True,
        architectures=['LlavaForConditionalGeneration'],
    ))


def get_model_tokenizer_yuan(model_dir: str,
                             config: PretrainedConfig,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             **kwargs):
    attn_type = AttentionImpl(kwargs.pop('use_flash_attn', None), kwargs.pop('attn_type', None))
    config.use_flash_attention = attn_type.to_bool()
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, add_eos_token=False, add_bos_token=False, eos_token='<eod>', legacy=True)
    addi_tokens = [
        '<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>',
        '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>',
        '<empty_output>'
    ]
    tokenizer.add_tokens(addi_tokens, special_tokens=True)
    model, tokenizer = get_model_tokenizer_from_local(
        model_dir, config, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.yuan2,
        [
            ModelGroup(
                [
                    Model('YuanLLM/Yuan2.0-2B-hf', 'IEITYuan/Yuan2-2B-hf'),
                    Model('YuanLLM/Yuan2.0-51B-hf', 'IEITYuan/Yuan2-51B-hf'),
                    Model('YuanLLM/Yuan2.0-102B-hf', 'IEITYuan/Yuan2-102B-hf'),
                    Model('YuanLLM/Yuan2-2B-Janus-hf', 'IEITYuan/Yuan2-2B-Janus-hf'),
                ]),
            ModelGroup(
                [
                    Model('YuanLLM/Yuan2-M32-hf', 'IEITYuan/Yuan2-M32-hf'),
                ], tags=['moe']),
        ],
        TemplateType.yuan,
        get_model_tokenizer_codellama,
        support_flash_attn=True,
        architectures=['LlavaForConditionalGeneration'],
    ))


def get_model_tokenizer_orion(model_dir: str,
                              config: PretrainedConfig,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    attn_type = AttentionImpl(kwargs.pop('use_flash_attn', None), kwargs.pop('attn_type', None))
    config._flash_attn_2_enabled = attn_type.to_bool()
    return get_model_tokenizer_from_local(
        model_dir, config, model_kwargs, load_model, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.yuan2,
        [
            ModelGroup(
                [
                    Model('OrionStarAI/Orion-14B-Base', 'OrionStarAI/Orion-14B-Base'),
                    Model('OrionStarAI/Orion-14B-Chat', 'OrionStarAI/Orion-14B-Chat'),
                ], ignore_file_pattern=[r'.+\.gguf$']),
        ],
        TemplateType.orion,
        get_model_tokenizer_codellama,
        support_flash_attn=True,
        architectures=['LlavaForConditionalGeneration'],
    ))


def get_model_tokenizer_idefics(model_dir: str, *args, **kwargs):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    processor = AutoProcessor.from_pretrained(model_dir)
    kwargs['automodel_class'] = AutoModelForVision2Seq
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    tokenizer.processor = processor
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.idefics3,
        [
            ModelGroup(
                [
                    Model('AI-ModelScope/Idefics3-8B-Llama3', 'HuggingFaceM4/Idefics3-8B-Llama3'),
                ], tags=['multi-modal', 'vision'], requires=['transformers>=4.45.0.dev0']),
        ],
        TemplateType.idefics3,
        get_model_tokenizer_idefics,
        support_flash_attn=True,
        architectures=['LlavaForConditionalGeneration'],
    ))


def get_model_tokenizer_mplug_owl2(model_dir: str,
                                   config: PretrainedConfig,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    if 'local_repo_path' in kwargs:
        local_repo_path = kwargs['local_repo_path']
    else:
        local_repo_path = git_clone_github('https://github.com/X-PLUG/mPLUG-Owl')
    local_repo_path = os.path.join(local_repo_path, 'mPLUG-Owl2')
    sys.path.append(os.path.join(local_repo_path))

    # register
    # https://github.com/X-PLUG/mPLUG-Owl/blob/main/mPLUG-Owl2/mplug_owl2/model/modeling_mplug_owl2.py#L447
    from transformers.models.clip.image_processing_clip import CLIPImageProcessor
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    vocab_size = kwargs.pop('vocab_size', None)
    if vocab_size is not None:
        model_config.vocab_size = vocab_size
    get_model_tokenizer_function = kwargs.pop('get_model_tokenizer_function')
    model, tokenizer = get_model_tokenizer_function(
        model_dir, config, model_kwargs, load_model, **kwargs)
    logger.info('Please ignore the unimported warning.')
    processor = CLIPImageProcessor.from_pretrained(model_dir)
    tokenizer.processor = processor
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.mplug2,
        [
            ModelGroup(
                [
                    Model('iic/mPLUG-Owl2', 'MAGAer13/mplug-owl2-llama2-7b'),
                ], tags=['multi-modal', 'vision'], requires=['transformers<4.35', 'icecream']),
        ],
        TemplateType.mplug_owl2,
        partial(get_model_tokenizer_mplug_owl2, get_model_tokenizer_function=get_model_tokenizer_with_flash_attn),
        support_flash_attn=True,
        architectures=['LlavaForConditionalGeneration'],
    ))


register_model(
    ModelMeta(
        MLLMModelType.mplug2_1,
        [
            ModelGroup(
                [
                    Model('iic/mPLUG-Owl2.1', 'Mizukiluke/mplug_owl_2_1'),
                ], tags=['multi-modal', 'vision'], requires=['transformers<4.35', 'icecream']),
        ],
        TemplateType.mplug_owl2,
        partial(get_model_tokenizer_mplug_owl2, vocab_size=151851, get_model_tokenizer_function=get_model_tokenizer_qwen),
        support_flash_attn=True,
        architectures=['LlavaForConditionalGeneration'],
    ))
