# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict

from transformers import AutoTokenizer, PretrainedConfig

from swift.template import TemplateType
from swift.utils import Processor, get_logger, safe_snapshot_download
from ..constant import LLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, SentenceTransformersLoader, register_model

logger = get_logger()


class GrokLoader(ModelLoader):

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        tokenizer_dir = safe_snapshot_download('AI-ModelScope/grok-1-tokenizer', download_model=False, check_local=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        return tokenizer


register_model(
    ModelMeta(
        LLMModelType.grok, [
            ModelGroup([
                Model('colossalai/grok-1-pytorch', 'hpcai-tech/grok-1'),
            ]),
        ],
        GrokLoader,
        template=TemplateType.default,
        architectures=['Grok1ModelForCausalLM'],
        model_arch=ModelArch.llama))


class PolyLMLoader(ModelLoader):

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False, legacy=True)


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
        PolyLMLoader,
        template=TemplateType.default,
        architectures=['GPT2LMHeadModel'],
        model_arch=ModelArch.qwen))


class YuanLoader(ModelLoader):

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, add_eos_token=False, add_bos_token=False, eos_token='<eod>', legacy=True)
        addi_tokens = [
            '<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>',
            '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>',
            '<empty_output>'
        ]
        tokenizer.add_tokens(addi_tokens, special_tokens=True)
        return tokenizer


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
        YuanLoader,
        template=TemplateType.yuan,
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
        template=TemplateType.orion,
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
        template=TemplateType.dbrx,
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
        template=TemplateType.bluelm,
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
        template=TemplateType.default,
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
        template=TemplateType.xverse,
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
        template=TemplateType.xverse,
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
        template=TemplateType.c4ai,
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
        template=TemplateType.aya,
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
        template=TemplateType.ling,
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
        SentenceTransformersLoader,
        template=TemplateType.dummy,
        architectures=['Qwen2ForCausalLM']))

register_model(
    ModelMeta(
        LLMModelType.mimo, [
            ModelGroup([
                Model('XiaomiMiMo/MiMo-7B-Base', 'XiaomiMiMo/MiMo-7B-Base'),
                Model('XiaomiMiMo/MiMo-7B-SFT', 'XiaomiMiMo/MiMo-7B-SFT'),
                Model('XiaomiMiMo/MiMo-7B-RL-Zero', 'XiaomiMiMo/MiMo-7B-RL-Zero'),
                Model('XiaomiMiMo/MiMo-7B-RL', 'XiaomiMiMo/MiMo-7B-RL'),
            ], TemplateType.qwen),
            ModelGroup([
                Model('XiaomiMiMo/MiMo-7B-RL-0530', 'XiaomiMiMo/MiMo-7B-RL-0530'),
            ], TemplateType.mimo_rl),
        ],
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
        template=TemplateType.dots1,
        architectures=['Dots1ForCausalLM'],
        requires=['transformers>=4.53'],
    ))

register_model(
    ModelMeta(
        LLMModelType.hunyuan,
        [ModelGroup([
            Model('Tencent-Hunyuan/Hunyuan-A13B-Instruct', 'tencent/Hunyuan-A13B-Instruct'),
        ])],
        template=TemplateType.hunyuan_moe,
        architectures=['HunYuanMoEV1ForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.hunyuan_v1_dense,
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
        template=TemplateType.hunyuan,
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
        template=TemplateType.gpt_oss,
        ignore_patterns=['metal/', 'original/'],
        architectures=['GptOssForCausalLM'],
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
        template=TemplateType.longchat,
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
        template=TemplateType.ling2,
        architectures=['BailingMoeV2ForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.ring2,
        [ModelGroup([
            Model('inclusionAI/Ring-mini-2.0', 'inclusionAI/Ring-mini-2.0'),
        ])],
        template=TemplateType.ring2,
        architectures=['BailingMoeV2ForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.iquestcoder,
        [
            ModelGroup([
                Model('IQuestLab/IQuest-Coder-V1-40B-Base-Stage1', 'IQuestLab/IQuest-Coder-V1-40B-Base-Stage1'),
                Model('IQuestLab/IQuest-Coder-V1-40B-Base', 'IQuestLab/IQuest-Coder-V1-40B-Base'),
                Model('IQuestLab/IQuest-Coder-V1-40B-Instruct', 'IQuestLab/IQuest-Coder-V1-40B-Instruct'),
            ])
        ],
        template=TemplateType.iquestcoder,
        requires=['transformers==4.52.4'],
        architectures=['IQuestCoderForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.youtu_llm,
        [
            ModelGroup([
                Model('Tencent-YouTu-Research/Youtu-LLM-2B', 'tencent/Youtu-LLM-2B'),
                Model('Tencent-YouTu-Research/Youtu-LLM-2B-Base', 'tencent/Youtu-LLM-2B-Base'),
            ])
        ],
        template=TemplateType.youtu_llm,
        architectures=['YoutuForCausalLM'],
        requires=['transformers>=4.56'],
    ))

register_model(
    ModelMeta(
        LLMModelType.olmoe,
        [
            ModelGroup([
                Model('allenai/OLMoE-1B-7B-0125', 'allenai/OLMoE-1B-7B-0125'),
                Model('allenai/OLMoE-1B-7B-0125-Instruct', 'allenai/OLMoE-1B-7B-0125-Instruct'),
            ],
                       template=TemplateType.olmoe),
            ModelGroup([
                Model('allenai/OLMoE-1B-7B-0924', 'allenai/OLMoE-1B-7B-0924'),
                Model('allenai/OLMoE-1B-7B-0924-Instruct', 'allenai/OLMoE-1B-7B-0924-Instruct'),
                Model('allenai/OLMoE-1B-7B-0924-SFT', 'allenai/OLMoE-1B-7B-0924-SFT'),
            ],
                       template=TemplateType.olmoe_0924)
        ],
        architectures=['OlmoeForCausalLM'],
    ))
