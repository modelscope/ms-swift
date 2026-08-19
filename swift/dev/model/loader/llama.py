# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the Llama text families.

Ported from ``swift/model/models/llama.py``. Legacy lumped ~15 distinct chat formats under the single
``llama`` model_type (one ``ModelMeta`` whose groups each carried their own template). dev is
one-template-per-loader, so each template becomes its own model_type: a base :class:`LlamaLoader`
(the actual Llama-2 / chinese-llama2 checkpoints) plus thin ``architectures=[]`` template-variant
subclasses that load identically. All share the base ``pretraining_tp`` config fix.

Routed to their own files (they are their own families, not Llama chat variants): ``deepseek-llm`` /
``deepseek-math`` / ``deepseek-coder`` and ``DeepSeek-R1-Distill-Llama`` -> deepseek.py; ``MiniCPM5``
-> minicpm.py. Multimodal (``llama3_2_vision`` / ``llama4`` / ``llama3_1_omni``) -> the MLLM pass.
Dropped: ``Llama-2-7b-AQLM-2Bit`` (niche 2-bit AQLM, extra deps; same call as the Mixtral AQLM drop).

Note: the legacy llama2 group carried ``ignore_patterns=[r'.+\\.bin$']``; not reproduced -- it was a
per-group download optimization (regex in a glob context, effectively inert) and applying it to the
merged base loader would risk the safetensors-less chinese-llama2 checkpoints.
"""
from __future__ import annotations

from .base import ModelArch, ModelLoader, register_model


@register_model
class LlamaLoader(ModelLoader):
    """Llama-2 and chinese-llama2; the reverse-lookup owner for ``LlamaForCausalLM``."""

    model_type = 'llama'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['LlamaForCausalLM']
    template = 'llama'
    models = [
        ('modelscope/Llama-2-7b-ms', 'meta-llama/Llama-2-7b-hf'),
        ('modelscope/Llama-2-13b-ms', 'meta-llama/Llama-2-13b-hf'),
        ('modelscope/Llama-2-70b-ms', 'meta-llama/Llama-2-70b-hf'),
        ('modelscope/Llama-2-7b-chat-ms', 'meta-llama/Llama-2-7b-chat-hf'),
        ('modelscope/Llama-2-13b-chat-ms', 'meta-llama/Llama-2-13b-chat-hf'),
        ('modelscope/Llama-2-70b-chat-ms', 'meta-llama/Llama-2-70b-chat-hf'),
        ('AI-ModelScope/chinese-llama-2-1.3b', 'hfl/chinese-llama-2-1.3b'),
        ('AI-ModelScope/chinese-llama-2-7b', 'hfl/chinese-llama-2-7b'),
        ('AI-ModelScope/chinese-llama-2-7b-16k', 'hfl/chinese-llama-2-7b-16k'),
        ('AI-ModelScope/chinese-llama-2-7b-64k', 'hfl/chinese-llama-2-7b-64k'),
        ('AI-ModelScope/chinese-llama-2-13b', 'hfl/chinese-llama-2-13b'),
        ('AI-ModelScope/chinese-llama-2-13b-16k', 'hfl/chinese-llama-2-13b-16k'),
        ('AI-ModelScope/chinese-alpaca-2-1.3b', 'hfl/chinese-alpaca-2-1.3b'),
        ('AI-ModelScope/chinese-alpaca-2-7b', 'hfl/chinese-alpaca-2-7b'),
        ('AI-ModelScope/chinese-alpaca-2-7b-16k', 'hfl/chinese-alpaca-2-7b-16k'),
        ('AI-ModelScope/chinese-alpaca-2-7b-64k', 'hfl/chinese-alpaca-2-7b-64k'),
        ('AI-ModelScope/chinese-alpaca-2-13b', 'hfl/chinese-alpaca-2-13b'),
        ('AI-ModelScope/chinese-alpaca-2-13b-16k', 'hfl/chinese-alpaca-2-13b-16k'),
    ]

    def process_config(self, config):
        # Legacy: some Llama configs ship pretraining_tp>1 (tensor-parallel training artifact) which
        # slows/complicates single-process load; force it off.
        if getattr(config, 'pretraining_tp', 1) > 1:
            config.pretraining_tp = 1
        return config


@register_model
class AtomLoader(LlamaLoader):
    model_type = 'atom'
    architectures = []
    template = 'atom'
    models = [
        ('FlagAlpha/Atom-7B', 'FlagAlpha/Atom-7B'),
        ('FlagAlpha/Atom-7B-Chat', 'FlagAlpha/Atom-7B-Chat'),
    ]


@register_model
class Mengzi3Loader(LlamaLoader):
    model_type = 'mengzi'
    architectures = []
    template = 'mengzi'
    models = [('langboat/Mengzi3-13B-Base', 'Langboat/Mengzi3-13B-Base')]


@register_model
class NuminaLoader(LlamaLoader):
    model_type = 'numina'
    architectures = []
    template = 'numina'
    tags = ['math']
    models = [('AI-ModelScope/NuminaMath-7B-TIR', 'AI-MO/NuminaMath-7B-TIR')]


@register_model
class Ziya2Loader(LlamaLoader):
    model_type = 'ziya'
    architectures = []
    template = 'ziya'
    models = [
        ('Fengshenbang/Ziya2-13B-Base', 'IDEA-CCNL/Ziya2-13B-Base'),
        ('Fengshenbang/Ziya2-13B-Chat', 'IDEA-CCNL/Ziya2-13B-Chat'),
    ]


@register_model
class MegrezLoader(LlamaLoader):
    model_type = 'megrez'
    architectures = []
    template = 'megrez'
    models = [('InfiniAI/Megrez-3b-Instruct', 'Infinigence/Megrez-3B-Instruct')]


@register_model
class MiniMindLoader(LlamaLoader):
    model_type = 'minimind'
    architectures = []
    template = 'minimind'
    requires = ['transformers>=4.57.1']
    # MiniMind2-Small has no ModelScope mirror; the bare hf id resolves by basename either way.
    models = [
        ('gongjy/MiniMind2', 'jingyaogong/MiniMind2'),
        'jingyaogong/MiniMind2-Small',
    ]


@register_model
class Llama3Loader(LlamaLoader):
    model_type = 'llama3'
    architectures = []
    template = 'llama3'
    models = [
        ('LLM-Research/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct'),
        ('LLM-Research/Meta-Llama-3-70B-Instruct', 'meta-llama/Meta-Llama-3-70B-Instruct'),
        ('LLM-Research/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-8B'),
        ('LLM-Research/Meta-Llama-3-70B', 'meta-llama/Meta-Llama-3-70B'),
        ('swift/Meta-Llama-3-8B-Instruct-GPTQ-Int4', 'study-hjt/Meta-Llama-3-8B-Instruct-GPTQ-Int4'),
        ('swift/Meta-Llama-3-8B-Instruct-GPTQ-Int8', 'study-hjt/Meta-Llama-3-8B-Instruct-GPTQ-Int8'),
        ('swift/Meta-Llama-3-8B-Instruct-AWQ', 'study-hjt/Meta-Llama-3-8B-Instruct-AWQ'),
        ('swift/Meta-Llama-3-70B-Instruct-GPTQ-Int4', 'study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int4'),
        ('swift/Meta-Llama-3-70B-Instruct-GPTQ-Int8', 'study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int8'),
        ('swift/Meta-Llama-3-70B-Instruct-AWQ', 'study-hjt/Meta-Llama-3-70B-Instruct-AWQ'),
        ('ChineseAlpacaGroup/llama-3-chinese-8b-instruct', 'hfl/llama-3-chinese-8b-instruct'),
        ('ChineseAlpacaGroup/llama-3-chinese-8b', 'hfl/llama-3-chinese-8b'),
    ]


@register_model
class Llama3_2Loader(LlamaLoader):
    # Llama-3.1 / 3.2 / 3.3 + Nemotron all share the `llama3_2` chat format; kept under one model_type
    # matching the legacy template name. requires>=4.43 (the floor legacy pinned on these groups).
    model_type = 'llama3_2'
    architectures = []
    template = 'llama3_2'
    requires = ['transformers>=4.43']
    models = [
        ('LLM-Research/Meta-Llama-3.1-8B-Instruct', 'meta-llama/Meta-Llama-3.1-8B-Instruct'),
        ('LLM-Research/Meta-Llama-3.1-70B-Instruct', 'meta-llama/Meta-Llama-3.1-70B-Instruct'),
        ('LLM-Research/Meta-Llama-3.1-405B-Instruct', 'meta-llama/Meta-Llama-3.1-405B-Instruct'),
        ('LLM-Research/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-8B'),
        ('LLM-Research/Meta-Llama-3.1-70B', 'meta-llama/Meta-Llama-3.1-70B'),
        ('LLM-Research/Meta-Llama-3.1-405B', 'meta-llama/Meta-Llama-3.1-405B'),
        ('LLM-Research/Meta-Llama-3.1-70B-Instruct-FP8', 'meta-llama/Meta-Llama-3.1-70B-Instruct-FP8'),
        ('LLM-Research/Meta-Llama-3.1-405B-Instruct-FP8', 'meta-llama/Meta-Llama-3.1-405B-Instruct-FP8'),
        ('LLM-Research/Meta-Llama-3.1-8B-Instruct-BNB-NF4', 'hugging-quants/Meta-Llama-3.1-8B-Instruct-BNB-NF4'),
        ('LLM-Research/Meta-Llama-3.1-70B-Instruct-bnb-4bit', 'unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit'),
        ('LLM-Research/Meta-Llama-3.1-405B-Instruct-BNB-NF4', 'hugging-quants/Meta-Llama-3.1-405B-Instruct-BNB-NF4'),
        ('LLM-Research/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4', 'hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4'),
        ('LLM-Research/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4', 'hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4'),
        ('LLM-Research/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4',
         'hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4'),
        ('LLM-Research/Meta-Llama-3.1-8B-Instruct-AWQ-INT4', 'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4'),
        ('LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4', 'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4'),
        ('LLM-Research/Meta-Llama-3.1-405B-Instruct-AWQ-INT4', 'hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4'),
        ('AI-ModelScope/Llama-3.1-Nemotron-70B-Instruct-HF', 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF'),
        ('LLM-Research/Llama-3.2-1B', 'meta-llama/Llama-3.2-1B'),
        ('LLM-Research/Llama-3.2-3B', 'meta-llama/Llama-3.2-3B'),
        ('LLM-Research/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-1B-Instruct'),
        ('LLM-Research/Llama-3.2-3B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct'),
        ('LLM-Research/Llama-3.3-70B-Instruct', 'meta-llama/Llama-3.3-70B-Instruct'),
        ('unsloth/Llama-3.3-70B-Instruct-bnb-4bit', 'unsloth/Llama-3.3-70B-Instruct-bnb-4bit'),
    ]


@register_model
class SkyworkO1Loader(LlamaLoader):
    model_type = 'skywork_o1'
    architectures = []
    template = 'skywork_o1'
    requires = ['transformers>=4.43']
    models = [('AI-ModelScope/Skywork-o1-Open-Llama-3.1-8B', 'Skywork/Skywork-o1-Open-Llama-3.1-8B')]


@register_model
class LongWriterLlamaLoader(LlamaLoader):
    model_type = 'longwriter_llama'
    architectures = []
    template = 'longwriter_llama'
    requires = ['transformers>=4.43']
    models = [('ZhipuAI/LongWriter-llama3.1-8b', 'zai-org/LongWriter-llama3.1-8b')]


@register_model
class ReflectionLoader(LlamaLoader):
    model_type = 'reflection'
    architectures = []
    template = 'reflection'
    requires = ['transformers>=4.43']
    models = [('LLM-Research/Reflection-Llama-3.1-70B', 'mattshumer/Reflection-Llama-3.1-70B')]


@register_model
class Llama3_2VisionLoader(ModelLoader):
    """Llama-3.2-Vision (Mllama). MLLM pilot: a fixed multimodal ``model_cls`` + a ``model_arch``
    partition. No keep-alive hook (only Qwen3-VL's mixed-data path needs one), no other seam."""

    model_type = 'llama3_2_vision'
    model_cls = 'transformers:MllamaForConditionalGeneration'
    architectures = ['MllamaForConditionalGeneration']
    template = 'llama3_2_vision'
    requires = ['transformers>=4.45']
    tags = ['vision']
    is_multimodal = True
    models = [
        ('LLM-Research/Llama-3.2-11B-Vision-Instruct', 'meta-llama/Llama-3.2-11B-Vision-Instruct'),
        ('LLM-Research/Llama-3.2-90B-Vision-Instruct', 'meta-llama/Llama-3.2-90B-Vision-Instruct'),
        ('LLM-Research/Llama-3.2-11B-Vision', 'meta-llama/Llama-3.2-11B-Vision'),
        ('LLM-Research/Llama-3.2-90B-Vision', 'meta-llama/Llama-3.2-90B-Vision'),
    ]

    @property
    def model_arch(self) -> ModelArch:
        # legacy `ModelArch.llama3_2_vision`, transformers>=4.52 (model.* prefix) branch.
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner='model.multi_modal_projector',
            vision_tower='model.vision_model',
        )


@register_model
class Llama4Loader(ModelLoader):
    """Llama-4 (Scout/Maverick). MLLM + MoE: verified against transformers 5.5, its parts sit bare at
    the top level (no ``model.`` prefix), and its sparse ``Llama4TextMoe`` needs a ZeRO-3 leaf mark."""

    model_type = 'llama4'
    model_cls = 'transformers:Llama4ForConditionalGeneration'
    architectures = ['Llama4ForConditionalGeneration']
    template = 'llama4'
    requires = ['transformers>=4.51']
    tags = ['vision']
    is_multimodal = True
    models = [
        ('LLM-Research/Llama-4-Scout-17B-16E', 'meta-llama/Llama-4-Scout-17B-16E'),
        ('LLM-Research/Llama-4-Maverick-17B-128E', 'meta-llama/Llama-4-Maverick-17B-128E'),
        ('LLM-Research/Llama-4-Scout-17B-16E-Instruct', 'meta-llama/Llama-4-Scout-17B-16E-Instruct'),
        ('LLM-Research/Llama-4-Maverick-17B-128E-Instruct-FP8', 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'),
        ('LLM-Research/Llama-4-Maverick-17B-128E-Instruct', 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'),
    ]

    @property
    def model_arch(self) -> ModelArch:
        # legacy `ModelArch.llama4` (not version-gated); parts are bare in transformers 5.5.
        return ModelArch(
            language_model='language_model',
            aligner='multi_modal_projector',
            vision_tower='vision_model',
            moe_block='Llama4TextMoe',
        )
