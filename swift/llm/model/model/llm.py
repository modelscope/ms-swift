# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from transformers import AutoTokenizer

from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType
from ..model_arch import ModelArch
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_sentence_transformers,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import ModelInfo, safe_snapshot_download

logger = get_logger()


def get_model_tokenizer_grok(*args, **kwargs):
    tokenizer_dir = safe_snapshot_download('AI-ModelScope/grok-1-tokenizer', download_model=False, check_local=True)
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

register_model(
    ModelMeta(
        LLMModelType.ling,
        [
            ModelGroup([
                Model('inclusionAI/Ling-lite', 'inclusionAI/Ling-lite'),
                Model('inclusionAI/Ling-plus', 'inclusionAI/Ling-plus'),
                Model('inclusionAI/Ling-lite-base', 'inclusionAI/Ling-lite-base'),
                Model('inclusionAI/Ling-plus-base', 'inclusionAI/Ling-plus-base'),
            ]),
        ],
        TemplateType.ling,
        get_model_tokenizer_with_flash_attn,
        architectures=['BailingMoeForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.qwen2_gte, [
            ModelGroup([
                Model('iic/gte_Qwen2-1.5B-instruct', 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'),
                Model('iic/gte_Qwen2-7B-instruct', 'Alibaba-NLP/gte-Qwen2-7B-instruct'),
            ]),
        ],
        None,
        get_model_tokenizer_sentence_transformers,
        architectures=['Qwen2ForCausalLM']))

register_model(
    ModelMeta(
        LLMModelType.mimo, [
            ModelGroup([
                Model('XiaomiMiMo/MiMo-7B-Base', 'XiaomiMiMo/MiMo-7B-Base'),
                Model('XiaomiMiMo/MiMo-7B-SFT', 'XiaomiMiMo/MiMo-7B-SFT'),
                Model('XiaomiMiMo/MiMo-7B-RL-Zero', 'XiaomiMiMo/MiMo-7B-RL-Zero'),
                Model('XiaomiMiMo/MiMo-7B-RL', 'XiaomiMiMo/MiMo-7B-RL'),
            ])
        ],
        TemplateType.qwen,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['MiMoForCausalLM'],
        requires=['transformers>=4.37']))

register_model(
    ModelMeta(
        LLMModelType.mimo_rl, [ModelGroup([
            Model('XiaomiMiMo/MiMo-7B-RL-0530', 'XiaomiMiMo/MiMo-7B-RL-0530'),
        ])],
        TemplateType.mimo_rl,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['MiMoForCausalLM'],
        requires=['transformers>=4.37']))

register_model(
    ModelMeta(
        LLMModelType.dots1,
        [
            ModelGroup([
                Model('rednote-hilab/dots.llm1.base', 'rednote-hilab/dots.llm1.base'),
                Model('rednote-hilab/dots.llm1.inst', 'rednote-hilab/dots.llm1.inst'),
            ])
        ],
        TemplateType.dots1,
        get_model_tokenizer_with_flash_attn,
        architectures=['Dots1ForCausalLM'],
        requires=['transformers>=4.53'],
    ))

register_model(
    ModelMeta(
        LLMModelType.hunyuan_moe,
        [ModelGroup([
            Model('Tencent-Hunyuan/Hunyuan-A13B-Instruct', 'tencent/Hunyuan-A13B-Instruct'),
        ])],
        TemplateType.hunyuan_moe,
        get_model_tokenizer_with_flash_attn,
        architectures=['HunYuanMoEV1ForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.hunyuan,
        [
            ModelGroup([
                Model('Tencent-Hunyuan/Hunyuan-0.5B-Instruct', 'tencent/Hunyuan-0.5B-Instruct'),
                Model('Tencent-Hunyuan/Hunyuan-1.8B-Instruct', 'tencent/Hunyuan-1.8B-Instruct'),
                Model('Tencent-Hunyuan/Hunyuan-4B-Instruct', 'tencent/Hunyuan-4B-Instruct'),
                Model('Tencent-Hunyuan/Hunyuan-7B-Instruct', 'tencent/Hunyuan-7B-Instruct'),
                # pretrain
                Model('Tencent-Hunyuan/Hunyuan-0.5B-Pretrain', 'tencent/Hunyuan-0.5B-Pretrain'),
                Model('Tencent-Hunyuan/Hunyuan-1.8B-Pretrain', 'tencent/Hunyuan-1.8B-Pretrain'),
                Model('Tencent-Hunyuan/Hunyuan-4B-Pretrain', 'tencent/Hunyuan-4B-Pretrain'),
                Model('Tencent-Hunyuan/Hunyuan-7B-Pretrain', 'tencent/Hunyuan-7B-Pretrain'),
                # fp8
                Model('Tencent-Hunyuan/Hunyuan-0.5B-Instruct-FP8', 'tencent/Hunyuan-0.5B-Instruct-FP8'),
                Model('Tencent-Hunyuan/Hunyuan-1.8B-Instruct-FP8', 'tencent/Hunyuan-1.8B-Instruct-FP8'),
                Model('Tencent-Hunyuan/Hunyuan-4B-Instruct-FP8', 'tencent/Hunyuan-4B-Instruct-FP8'),
                Model('Tencent-Hunyuan/Hunyuan-7B-Instruct-FP8', 'tencent/Hunyuan-7B-Instruct-FP8'),
                # awq
                Model('Tencent-Hunyuan/Hunyuan-0.5B-Instruct-AWQ-Int4', 'tencent/Hunyuan-0.5B-Instruct-AWQ-Int4'),
                Model('Tencent-Hunyuan/Hunyuan-1.8B-Instruct-AWQ-Int4', 'tencent/Hunyuan-1.8B-Instruct-AWQ-Int4'),
                Model('Tencent-Hunyuan/Hunyuan-4B-Instruct-AWQ-Int4', 'tencent/Hunyuan-4B-Instruct-AWQ-Int4'),
                Model('Tencent-Hunyuan/Hunyuan-7B-Instruct-AWQ-Int4', 'tencent/Hunyuan-7B-Instruct-AWQ-Int4'),
                # gptq
                Model('Tencent-Hunyuan/Hunyuan-0.5B-Instruct-GPTQ-Int4', 'tencent/Hunyuan-0.5B-Instruct-GPTQ-Int4'),
                Model('Tencent-Hunyuan/Hunyuan-1.8B-Instruct-GPTQ-Int4', 'tencent/Hunyuan-1.8B-Instruct-GPTQ-Int4'),
                Model('Tencent-Hunyuan/Hunyuan-4B-Instruct-GPTQ-Int4', 'tencent/Hunyuan-4B-Instruct-GPTQ-Int4'),
                Model('Tencent-Hunyuan/Hunyuan-7B-Instruct-GPTQ-Int4', 'tencent/Hunyuan-7B-Instruct-GPTQ-Int4'),
            ])
        ],
        TemplateType.hunyuan,
        get_model_tokenizer_with_flash_attn,
        requires=['transformers>=4.55.0.dev0'],
        architectures=['HunYuanDenseV1ForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.gpt_oss, [
            ModelGroup([
                Model('openai-mirror/gpt-oss-20b', 'openai/gpt-oss-20b'),
                Model('openai-mirror/gpt-oss-120b', 'openai/gpt-oss-120b'),
            ])
        ],
        TemplateType.gpt_oss,
        get_model_tokenizer_with_flash_attn,
        architectures=['GptOssForCausalLM'],
        ignore_patterns=['metal/', 'original/'],
        requires=['transformers>=4.55']))

register_model(
    ModelMeta(
        LLMModelType.longchat,
        [
            ModelGroup([
                Model('meituan-longcat/LongCat-Flash-Chat', 'meituan-longcat/LongCat-Flash-Chat'),
                Model('meituan-longcat/LongCat-Flash-Chat-FP8', 'meituan-longcat/LongCat-Flash-Chat-FP8'),
            ])
        ],
        TemplateType.longchat,
        get_model_tokenizer_with_flash_attn,
        architectures=['LongcatFlashForCausalLM'],
        requires=['transformers>=4.54,<4.56'],
    ))

register_model(
    ModelMeta(
        LLMModelType.ling2,
        [
            ModelGroup([
                Model('inclusionAI/Ling-mini-2.0', 'inclusionAI/Ling-mini-2.0'),
                Model('inclusionAI/Ling-mini-base-2.0', 'inclusionAI/Ling-mini-base-2.0'),
            ])
        ],
        TemplateType.ling2,
        get_model_tokenizer_with_flash_attn,
        architectures=['BailingMoeV2ForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.ring2,
        [ModelGroup([
            Model('inclusionAI/Ring-mini-2.0', 'inclusionAI/Ring-mini-2.0'),
        ])],
        TemplateType.ring2,
        get_model_tokenizer_with_flash_attn,
        architectures=['BailingMoeV2ForCausalLM'],
    ))
