# Copyright (c) Alibaba, Inc. and its affiliates.
from types import MethodType
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType
from ..model_arch import ModelArch
from ..patcher import patch_output_clone
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo, git_clone_github, use_submodel_func

logger = get_logger()


def get_model_tokenizer_grok(model_dir: str,
                             model_info: ModelInfo,
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
        model = automodel_class.from_pretrained(model_dir, trust_remote_code=True, **model_kwargs)
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.grok, [
            ModelGroup([
                Model('colossalai/grok-1-pytorch', 'hpcai-tech/grok-1'),
            ]),
        ],
        TemplateType.default,
        get_model_tokenizer_grok,
        architectures=['Grok1ModelForCausalLM'],
        model_arch=ModelArch.llama
        # TODO
    ))


def get_model_tokenizer_polylm(model_dir: str,
                               model_info: ModelInfo,
                               model_kwargs: Dict[str, Any],
                               load_model: bool = True,
                               **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False, legacy=True)
    return get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.polylm,
        [
            ModelGroup(
                [
                    # base
                    Model('damo/nlp_polylm_13b_text_generation', 'DAMO-NLP-MT/polylm-13b'),
                ], ),
        ],
        TemplateType.default,
        get_model_tokenizer_polylm,
        architectures=['GPT2LMHeadModel'],
        model_arch=ModelArch.qwen))


def get_skywork_model_tokenizer(model_dir: str,
                                model_info: ModelInfo,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if 'chat' in model_dir:
        tokenizer.add_tokens('[USER]')
        tokenizer.add_tokens('[BOT]')
        tokenizer.add_tokens('[SEP]')
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.skywork,
        [
            ModelGroup([
                Model('skywork/Skywork-13B-chat'),
                Model('skywork/Skywork-13B-base'),
            ]),
        ],
        TemplateType.skywork,
        get_skywork_model_tokenizer,
        architectures=['SkyworkForCausalLM'],
        model_arch=ModelArch.llama,
    ))


def get_model_tokenizer_yuan(model_dir: str,
                             model_info: ModelInfo,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, add_eos_token=False, add_bos_token=False, eos_token='<eod>', legacy=True)
    addi_tokens = [
        '<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>',
        '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>',
        '<empty_output>'
    ]
    tokenizer.add_tokens(addi_tokens, special_tokens=True)
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.yuan2,
        [
            ModelGroup([
                Model('IEITYuan/Yuan2.0-2B-hf', 'IEITYuan/Yuan2-2B-hf'),
                Model('IEITYuan/Yuan2.0-51B-hf', 'IEITYuan/Yuan2-51B-hf'),
                Model('IEITYuan/Yuan2.0-102B-hf', 'IEITYuan/Yuan2-102B-hf'),
                Model('IEITYuan/Yuan2-2B-Janus-hf', 'IEITYuan/Yuan2-2B-Janus-hf'),
            ]),
            ModelGroup([
                Model('IEITYuan/Yuan2-M32-hf', 'IEITYuan/Yuan2-M32-hf'),
            ]),
        ],
        TemplateType.yuan,
        get_model_tokenizer_yuan,
        model_arch=ModelArch.llama,
        architectures=['YuanForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.orion,
        [
            ModelGroup([
                Model('OrionStarAI/Orion-14B-Chat', 'OrionStarAI/Orion-14B-Chat'),
                Model('OrionStarAI/Orion-14B-Base', 'OrionStarAI/Orion-14B-Base'),
            ],
                       requires=['transformers==4.34.1']),
        ],
        TemplateType.orion,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['OrionForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.dbrx,
        [
            ModelGroup([
                Model('AI-ModelScope/dbrx-instruct', 'databricks/dbrx-instruct'),
                Model('AI-ModelScope/dbrx-base', 'databricks/dbrx-base'),
            ],
                       requires=['transformers>=4.36']),
        ],
        TemplateType.dbrx,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.dbrx,
        architectures=['DbrxForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.bluelm,
        [
            ModelGroup([
                Model('vivo-ai/BlueLM-7B-Chat-32K', 'vivo-ai/BlueLM-7B-Chat-32K'),
                Model('vivo-ai/BlueLM-7B-Chat', 'vivo-ai/BlueLM-7B-Chat'),
                Model('vivo-ai/BlueLM-7B-Base-32K', 'vivo-ai/BlueLM-7B-Base-32K'),
                Model('vivo-ai/BlueLM-7B-Base', 'vivo-ai/BlueLM-7B-Base'),
            ]),
        ],
        TemplateType.bluelm,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['BlueLMForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.seggpt,
        [
            ModelGroup([
                Model('damo/nlp_seqgpt-560m', 'DAMO-NLP/SeqGPT-560M'),
            ], tags=['audio']),
        ],
        TemplateType.default,
        get_model_tokenizer_with_flash_attn,
        model_arch=None,
        architectures=['BloomForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.xverse,
        [
            ModelGroup([
                Model('xverse/XVERSE-7B-Chat', 'xverse/XVERSE-7B-Chat'),
                Model('xverse/XVERSE-7B', 'xverse/XVERSE-7B'),
                Model('xverse/XVERSE-13B', 'xverse/XVERSE-13B'),
                Model('xverse/XVERSE-13B-Chat', 'xverse/XVERSE-13B-Chat'),
                Model('xverse/XVERSE-65B', 'xverse/XVERSE-65B'),
                Model('xverse/XVERSE-65B-2', 'xverse/XVERSE-65B-2'),
                Model('xverse/XVERSE-65B-Chat', 'xverse/XVERSE-65B-Chat'),
                Model('xverse/XVERSE-13B-256K', 'xverse/XVERSE-13B-256K', ms_revision='v1.0.0'),
            ],
                       requires=['transformers==4.38.2']),
        ],
        TemplateType.xverse,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['XverseForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.xverse_moe,
        [
            ModelGroup([
                Model('xverse/XVERSE-MoE-A4.2B', 'xverse/XVERSE-MoE-A4.2B'),
            ],
                       requires=['transformers==4.38.2']),
        ],
        TemplateType.xverse,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['XverseForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.c4ai,
        [
            ModelGroup([
                Model('AI-ModelScope/c4ai-command-r-v01', 'CohereForAI/c4ai-command-r-v01'),
                Model('AI-ModelScope/c4ai-command-r-plus', 'CohereForAI/c4ai-command-r-plus'),
            ],
                       requires=['transformers>=4.39']),
        ],
        TemplateType.c4ai,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['CohereForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.aya,
        [
            ModelGroup([
                Model('AI-ModelScope/aya-expanse-8b', 'CohereForAI/aya-expanse-8b'),
                Model('AI-ModelScope/aya-expanse-32b', 'CohereForAI/aya-expanse-32b'),
            ],
                       requires=['transformers>=4.44.0']),
        ],
        TemplateType.aya,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['CohereForCausalLM'],
    ))
