# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the assorted single-family text LLMs (from ``swift/model/models/llm.py``).

These are the "one register_model each" text models that do not warrant their own file. Almost all are
plain ``AutoModelForCausalLM`` with no legacy loader at all; the three that had one keep only its real
behaviour (a processor tweak), see ``GrokLoader`` / ``PolyLMLoader`` / ``Yuan2Loader`` below. Text
models keep the default empty ``ModelArch``; MoE families are flagged ``is_moe`` (no ``moe_block``
yet -- same call as qwen3_moe / deepseek: z3 is unwired and the leaf class name is not guessed).

Architectures absent from transformers 5.5 (``BailingMoe*``, ``BlueLM``, ``Orion``, ``Xverse``,
``Yuan``, ``Grok1Model``, ``HYV3``, ``MiMo``) are remote-code checkpoints loaded through the g7
``trust_remote_code`` flag; the in-tree ones (GptOss/Dbrx/Olmoe/Dots1/HunYuan*/Cohere/Longcat/Youtu/
Bloom/GPT2) leave it False.

Legacy per-``ModelGroup`` template differences become ``architectures=[]`` template-variant subclasses
(the llama/qwen pattern): olmoe/olmoe_0924, ling2/ring2, ling2/ring2_5, hy_v3_preview/hy_v3, qwen/mimo_rl.

Not migrated here (see MODEL_MIGRATION.md):
  * ``iquestcoder`` -- pinned ``transformers==4.52.4`` (dead on dev's 5.5).
  * ``qwen2_gte`` -- a ``SentenceTransformersLoader`` (embedding backbone via sentence-transformers),
    which is a different load path than ``ModelLoader``; belongs with the task-head/embedding pass.
"""
from __future__ import annotations

from .base import ModelLoader, register_model


@register_model
class GrokLoader(ModelLoader):
    """Grok-1: the checkpoint ships no tokenizer, so legacy pulled one from a separate repo."""

    model_type = 'grok'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['Grok1ModelForCausalLM']
    template = 'default'
    is_moe = True
    models = [('colossalai/grok-1-pytorch', 'hpcai-tech/grok-1')]

    def build_processor(self, model_dir: str, config, **kwargs):
        from swift.dev.utils import safe_snapshot_download
        tokenizer_dir = safe_snapshot_download(
            'AI-ModelScope/grok-1-tokenizer', download_model=False, check_local=True)
        return super().build_processor(tokenizer_dir, config, **kwargs)


@register_model
class PolyLMLoader(ModelLoader):
    """PolyLM (GPT2 arch): needs the slow tokenizer (``use_fast=False``, legacy sentencepiece)."""

    model_type = 'polylm'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['GPT2LMHeadModel']
    template = 'default'
    models = [('damo/nlp_polylm_13b_text_generation', 'DAMO-NLP-MT/polylm-13b')]

    def build_processor(self, model_dir: str, config, **kwargs):
        kwargs.setdefault('use_fast', False)
        kwargs.setdefault('legacy', True)
        return super().build_processor(model_dir, config, **kwargs)


@register_model
class Yuan2Loader(ModelLoader):
    """Yuan2: eos is ``<eod>`` with no added bos/eos, plus a fixed set of sentinel/FIM tokens."""

    model_type = 'yuan2'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['YuanForCausalLM']
    template = 'yuan'
    models = [
        ('IEITYuan/Yuan2.0-2B-hf', 'IEITYuan/Yuan2-2B-hf'),
        ('IEITYuan/Yuan2.0-51B-hf', 'IEITYuan/Yuan2-51B-hf'),
        ('IEITYuan/Yuan2.0-102B-hf', 'IEITYuan/Yuan2-102B-hf'),
        ('IEITYuan/Yuan2-2B-Janus-hf', 'IEITYuan/Yuan2-2B-Janus-hf'),
        ('IEITYuan/Yuan2-M32-hf', 'IEITYuan/Yuan2-M32-hf'),
    ]

    def build_processor(self, model_dir: str, config, **kwargs):
        kwargs.setdefault('add_eos_token', False)
        kwargs.setdefault('add_bos_token', False)
        kwargs.setdefault('eos_token', '<eod>')
        kwargs.setdefault('legacy', True)
        return super().build_processor(model_dir, config, **kwargs)

    def process_tokenizer(self, tokenizer):
        tokenizer.add_tokens([
            '<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>',
            '<commit_before>', '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>',
            '<jupyter_code>', '<jupyter_output>', '<empty_output>'
        ],
                             special_tokens=True)
        return tokenizer


@register_model
class OrionLoader(ModelLoader):
    model_type = 'orion'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['OrionForCausalLM']
    template = 'orion'
    models = [
        ('OrionStarAI/Orion-14B-Chat', 'OrionStarAI/Orion-14B-Chat'),
        ('OrionStarAI/Orion-14B-Base', 'OrionStarAI/Orion-14B-Base'),
    ]


@register_model
class DbrxLoader(ModelLoader):
    model_type = 'dbrx'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['DbrxForCausalLM']
    template = 'dbrx'
    requires = ['transformers>=4.36']
    is_moe = True
    models = [
        ('AI-ModelScope/dbrx-base', 'databricks/dbrx-base'),
        ('AI-ModelScope/dbrx-instruct', 'databricks/dbrx-instruct'),
    ]


@register_model
class BlueLMLoader(ModelLoader):
    model_type = 'bluelm'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['BlueLMForCausalLM']
    template = 'bluelm'
    models = [
        ('vivo-ai/BlueLM-7B-Chat-32K', 'vivo-ai/BlueLM-7B-Chat-32K'),
        ('vivo-ai/BlueLM-7B-Chat', 'vivo-ai/BlueLM-7B-Chat'),
        ('vivo-ai/BlueLM-7B-Base-32K', 'vivo-ai/BlueLM-7B-Base-32K'),
        ('vivo-ai/BlueLM-7B-Base', 'vivo-ai/BlueLM-7B-Base'),
    ]


@register_model
class SeqGPTLoader(ModelLoader):
    """SeqGPT-560M (Bloom arch). Legacy pinned ``model_arch=None``; dev's empty default matches."""

    model_type = 'seggpt'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['BloomForCausalLM']
    template = 'default'
    models = [('damo/nlp_seqgpt-560m', 'DAMO-NLP/SeqGPT-560M')]


@register_model
class XverseLoader(ModelLoader):
    """XVERSE dense; reverse-lookup owner for ``XverseForCausalLM`` (the MoE checkpoint declares the
    same architecture, so it is a template-less variant distinguished by id)."""

    model_type = 'xverse'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['XverseForCausalLM']
    template = 'xverse'
    models = [
        ('xverse/XVERSE-7B-Chat', 'xverse/XVERSE-7B-Chat'),
        ('xverse/XVERSE-7B', 'xverse/XVERSE-7B'),
        ('xverse/XVERSE-13B', 'xverse/XVERSE-13B'),
        ('xverse/XVERSE-13B-Chat', 'xverse/XVERSE-13B-Chat'),
        ('xverse/XVERSE-65B', 'xverse/XVERSE-65B'),
        ('xverse/XVERSE-65B-2', 'xverse/XVERSE-65B-2'),
        ('xverse/XVERSE-65B-Chat', 'xverse/XVERSE-65B-Chat'),
        ('xverse/XVERSE-13B-256K', 'xverse/XVERSE-13B-256K'),
    ]


@register_model
class XverseMoeLoader(XverseLoader):
    model_type = 'xverse_moe'
    architectures = []
    is_moe = True
    models = [('xverse/XVERSE-MoE-A4.2B', 'xverse/XVERSE-MoE-A4.2B')]


@register_model
class C4aiLoader(ModelLoader):
    """Command-R (Cohere arch); reverse-lookup owner for ``CohereForCausalLM`` (aya shares it)."""

    model_type = 'c4ai'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['CohereForCausalLM']
    template = 'c4ai'
    requires = ['transformers>=4.39']
    models = [
        ('AI-ModelScope/c4ai-command-r-v01', 'CohereForAI/c4ai-command-r-v01'),
        ('AI-ModelScope/c4ai-command-r-plus', 'CohereForAI/c4ai-command-r-plus'),
    ]


@register_model
class AyaLoader(C4aiLoader):
    model_type = 'aya'
    architectures = []
    template = 'aya'
    requires = ['transformers>=4.44.0']
    models = [
        ('AI-ModelScope/aya-expanse-8b', 'CohereForAI/aya-expanse-8b'),
        ('AI-ModelScope/aya-expanse-32b', 'CohereForAI/aya-expanse-32b'),
    ]


@register_model
class LingLoader(ModelLoader):
    """Ling v1 (BailingMoe); remote-code."""

    model_type = 'ling'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['BailingMoeForCausalLM']
    template = 'ling'
    is_moe = True
    models = [
        ('inclusionAI/Ling-lite', 'inclusionAI/Ling-lite'),
        ('inclusionAI/Ling-plus', 'inclusionAI/Ling-plus'),
        ('inclusionAI/Ling-lite-base', 'inclusionAI/Ling-lite-base'),
        ('inclusionAI/Ling-plus-base', 'inclusionAI/Ling-plus-base'),
    ]


@register_model
class BailingMoeLoader(ModelLoader):
    """Ling/Ring 2.0 (BailingMoeV2); remote-code. Base group is the ``ling2`` template."""

    model_type = 'bailing_moe'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['BailingMoeV2ForCausalLM']
    template = 'ling2'
    is_moe = True
    models = [
        ('inclusionAI/Ling-mini-2.0', 'inclusionAI/Ling-mini-2.0'),
        ('inclusionAI/Ling-mini-base-2.0', 'inclusionAI/Ling-mini-base-2.0'),
        ('inclusionAI/Ling-1T', 'inclusionAI/Ling-1T'),
    ]


@register_model
class RingMoeLoader(BailingMoeLoader):
    model_type = 'ring_moe'
    architectures = []
    template = 'ring2'
    models = [('inclusionAI/Ring-mini-2.0', 'inclusionAI/Ring-mini-2.0')]


@register_model
class BailingHybridLoader(ModelLoader):
    """Ling/Ring 2.5-2.6 (BailingMoeV2_5); remote-code. Base group is the ``ling2`` template."""

    model_type = 'bailing_hybrid'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['BailingMoeV2_5ForCausalLM']
    template = 'ling2'
    is_moe = True
    models = [
        ('inclusionAI/Ling-2.5-1T', 'inclusionAI/Ling-2.5-1T'),
        ('inclusionAI/Ling-2.6-1T', 'inclusionAI/Ling-2.6-1T'),
        ('inclusionAI/Ling-2.6-flash', 'inclusionAI/Ling-2.6-flash'),
    ]


@register_model
class RingHybridLoader(BailingHybridLoader):
    model_type = 'ring_hybrid'
    architectures = []
    template = 'ring2_5'
    models = [
        ('inclusionAI/Ring-2.5-1T', 'inclusionAI/Ring-2.5-1T'),
        ('inclusionAI/Ring-2.6-1T', 'inclusionAI/Ring-2.6-1T'),
    ]


@register_model
class MiMoLoader(ModelLoader):
    """MiMo-7B; remote-code. Base group borrows the ``qwen`` template."""

    model_type = 'mimo'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['MiMoForCausalLM']
    template = 'qwen'
    requires = ['transformers>=4.37']
    models = [
        ('XiaomiMiMo/MiMo-7B-Base', 'XiaomiMiMo/MiMo-7B-Base'),
        ('XiaomiMiMo/MiMo-7B-SFT', 'XiaomiMiMo/MiMo-7B-SFT'),
        ('XiaomiMiMo/MiMo-7B-RL-Zero', 'XiaomiMiMo/MiMo-7B-RL-Zero'),
        ('XiaomiMiMo/MiMo-7B-RL', 'XiaomiMiMo/MiMo-7B-RL'),
    ]


@register_model
class MiMoRLLoader(MiMoLoader):
    model_type = 'mimo_rl'
    architectures = []
    template = 'mimo_rl'
    models = [('XiaomiMiMo/MiMo-7B-RL-0530', 'XiaomiMiMo/MiMo-7B-RL-0530')]


@register_model
class Dots1Loader(ModelLoader):
    model_type = 'dots1'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['Dots1ForCausalLM']
    template = 'dots1'
    requires = ['transformers>=4.53']
    is_moe = True
    models = [
        ('rednote-hilab/dots.llm1.base', 'rednote-hilab/dots.llm1.base'),
        ('rednote-hilab/dots.llm1.inst', 'rednote-hilab/dots.llm1.inst'),
    ]


@register_model
class HunyuanMoeLoader(ModelLoader):
    model_type = 'hunyuan'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['HunYuanMoEV1ForCausalLM']
    template = 'hunyuan_moe'
    is_moe = True
    models = [('Tencent-Hunyuan/Hunyuan-A13B-Instruct', 'tencent/Hunyuan-A13B-Instruct')]


@register_model
class HunyuanV1DenseLoader(ModelLoader):
    model_type = 'hunyuan_v1_dense'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['HunYuanDenseV1ForCausalLM']
    template = 'hunyuan'
    requires = ['transformers>=4.55.0.dev0']
    models = [
        ('Tencent-Hunyuan/Hunyuan-0.5B-Instruct', 'tencent/Hunyuan-0.5B-Instruct'),
        ('Tencent-Hunyuan/Hunyuan-1.8B-Instruct', 'tencent/Hunyuan-1.8B-Instruct'),
        ('Tencent-Hunyuan/Hunyuan-4B-Instruct', 'tencent/Hunyuan-4B-Instruct'),
        ('Tencent-Hunyuan/Hunyuan-7B-Instruct', 'tencent/Hunyuan-7B-Instruct'),
        ('Tencent-Hunyuan/Hunyuan-0.5B-Pretrain', 'tencent/Hunyuan-0.5B-Pretrain'),
        ('Tencent-Hunyuan/Hunyuan-1.8B-Pretrain', 'tencent/Hunyuan-1.8B-Pretrain'),
        ('Tencent-Hunyuan/Hunyuan-4B-Pretrain', 'tencent/Hunyuan-4B-Pretrain'),
        ('Tencent-Hunyuan/Hunyuan-7B-Pretrain', 'tencent/Hunyuan-7B-Pretrain'),
        # quantized releases (same architecture/template; the quant config rides in the checkpoint)
        ('Tencent-Hunyuan/Hunyuan-0.5B-Instruct-FP8', 'tencent/Hunyuan-0.5B-Instruct-FP8'),
        ('Tencent-Hunyuan/Hunyuan-1.8B-Instruct-FP8', 'tencent/Hunyuan-1.8B-Instruct-FP8'),
        ('Tencent-Hunyuan/Hunyuan-4B-Instruct-FP8', 'tencent/Hunyuan-4B-Instruct-FP8'),
        ('Tencent-Hunyuan/Hunyuan-7B-Instruct-FP8', 'tencent/Hunyuan-7B-Instruct-FP8'),
        ('Tencent-Hunyuan/Hunyuan-0.5B-Instruct-AWQ-Int4', 'tencent/Hunyuan-0.5B-Instruct-AWQ-Int4'),
        ('Tencent-Hunyuan/Hunyuan-1.8B-Instruct-AWQ-Int4', 'tencent/Hunyuan-1.8B-Instruct-AWQ-Int4'),
        ('Tencent-Hunyuan/Hunyuan-4B-Instruct-AWQ-Int4', 'tencent/Hunyuan-4B-Instruct-AWQ-Int4'),
        ('Tencent-Hunyuan/Hunyuan-7B-Instruct-AWQ-Int4', 'tencent/Hunyuan-7B-Instruct-AWQ-Int4'),
        ('Tencent-Hunyuan/Hunyuan-0.5B-Instruct-GPTQ-Int4', 'tencent/Hunyuan-0.5B-Instruct-GPTQ-Int4'),
        ('Tencent-Hunyuan/Hunyuan-1.8B-Instruct-GPTQ-Int4', 'tencent/Hunyuan-1.8B-Instruct-GPTQ-Int4'),
        ('Tencent-Hunyuan/Hunyuan-4B-Instruct-GPTQ-Int4', 'tencent/Hunyuan-4B-Instruct-GPTQ-Int4'),
        ('Tencent-Hunyuan/Hunyuan-7B-Instruct-GPTQ-Int4', 'tencent/Hunyuan-7B-Instruct-GPTQ-Int4'),
    ]


@register_model
class HyV3PreviewLoader(ModelLoader):
    """Hunyuan v3 preview; remote-code (``HYV3ForCausalLM`` not in transformers 5.5) and its own
    ``hy_v3_preview`` template. The reverse-lookup owner for the arch."""

    model_type = 'hy_v3_preview'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['HYV3ForCausalLM']
    template = 'hy_v3_preview'
    requires = ['transformers>=5.6.0']
    is_moe = True
    models = [
        ('Tencent-Hunyuan/Hy3-preview', 'tencent/Hy3-preview'),
        ('Tencent-Hunyuan/Hy3-preview-Base', 'tencent/Hy3-preview-Base'),
    ]


@register_model
class HyV3Loader(HyV3PreviewLoader):
    model_type = 'hy_v3'
    architectures = []
    template = 'hy_v3'
    models = [
        ('Tencent-Hunyuan/Hy3', 'tencent/Hy3'),
        ('Tencent-Hunyuan/Hy3-FP8', 'tencent/Hy3-FP8'),
    ]


@register_model
class GptOssLoader(ModelLoader):
    """gpt-oss: ``ignore_patterns`` skips the non-HF ``metal/`` and ``original/`` weight dirs."""

    model_type = 'gpt_oss'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['GptOssForCausalLM']
    template = 'gpt_oss'
    requires = ['transformers>=4.55']
    ignore_patterns = ['metal/', 'original/']
    is_moe = True
    models = [
        ('openai-mirror/gpt-oss-20b', 'openai/gpt-oss-20b'),
        ('openai-mirror/gpt-oss-120b', 'openai/gpt-oss-120b'),
    ]


@register_model
class LongCatLoader(ModelLoader):
    model_type = 'longchat'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['LongcatFlashForCausalLM']
    template = 'longchat'
    requires = ['transformers>=4.54,<4.56']
    is_moe = True
    models = [
        ('meituan-longcat/LongCat-Flash-Chat', 'meituan-longcat/LongCat-Flash-Chat'),
        ('meituan-longcat/LongCat-Flash-Chat-FP8', 'meituan-longcat/LongCat-Flash-Chat-FP8'),
    ]


@register_model
class YoutuLLMLoader(ModelLoader):
    model_type = 'youtu_llm'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['YoutuForCausalLM']
    template = 'youtu_llm'
    requires = ['transformers>=4.56']
    models = [
        ('Tencent-YouTu-Research/Youtu-LLM-2B', 'tencent/Youtu-LLM-2B'),
        ('Tencent-YouTu-Research/Youtu-LLM-2B-Base', 'tencent/Youtu-LLM-2B-Base'),
    ]


@register_model
class OlmoeLoader(ModelLoader):
    """OLMoE; reverse-lookup owner for ``OlmoeForCausalLM``. Base group is the 0125 ``olmoe``
    template; the 0924 checkpoints use their own."""

    model_type = 'olmoe'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['OlmoeForCausalLM']
    template = 'olmoe'
    is_moe = True
    models = [
        ('allenai/OLMoE-1B-7B-0125', 'allenai/OLMoE-1B-7B-0125'),
        ('allenai/OLMoE-1B-7B-0125-Instruct', 'allenai/OLMoE-1B-7B-0125-Instruct'),
    ]


@register_model
class Olmoe0924Loader(OlmoeLoader):
    model_type = 'olmoe_0924'
    architectures = []
    template = 'olmoe_0924'
    models = [
        ('allenai/OLMoE-1B-7B-0924', 'allenai/OLMoE-1B-7B-0924'),
        ('allenai/OLMoE-1B-7B-0924-Instruct', 'allenai/OLMoE-1B-7B-0924-Instruct'),
        ('allenai/OLMoE-1B-7B-0924-SFT', 'allenai/OLMoE-1B-7B-0924-SFT'),
    ]
