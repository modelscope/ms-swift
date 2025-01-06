# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from transformers import AutoTokenizer

from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType
from ..model_arch import ModelArch
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo, safe_snapshot_download

logger = get_logger()


def get_model_tokenizer_grok(*args, **kwargs):
    tokenizer_dir = safe_snapshot_download('AI-ModelScope/grok-1-tokenizer', download_model=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    kwargs['tokenizer'] = tokenizer
    model, _ = get_model_tokenizer_with_flash_attn(*args, **kwargs)
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
            ]),
        ],
        TemplateType.orion,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['OrionForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.dbrx, [
            ModelGroup([
                Model('AI-ModelScope/dbrx-base', 'databricks/dbrx-base'),
                Model('AI-ModelScope/dbrx-instruct', 'databricks/dbrx-instruct'),
            ]),
        ],
        TemplateType.dbrx,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.dbrx,
        architectures=['DbrxForCausalLM'],
        requires=['transformers>=4.36']))

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
            ]),
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
            ]),
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
            ]),
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
            ]),
        ],
        TemplateType.c4ai,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['CohereForCausalLM'],
        requires=['transformers>=4.39'],
    ))

register_model(
    ModelMeta(
        LLMModelType.aya, [
            ModelGroup([
                Model('AI-ModelScope/aya-expanse-8b', 'CohereForAI/aya-expanse-8b'),
                Model('AI-ModelScope/aya-expanse-32b', 'CohereForAI/aya-expanse-32b'),
            ]),
        ],
        TemplateType.aya,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['CohereForCausalLM'],
        requires=['transformers>=4.44.0']))
