# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the Mistral families.

Ported from ``swift/model/models/mistral.py`` against the ``ModelLoader`` seams in ``.base``.

The plain-text families declare only ``model_cls = 'transformers:AutoModelForCausalLM'`` and lean on
the base happy path: ``AutoConfig``, tokenizer detection in ``build_processor`` (a text checkpoint
ships no ``preprocessor_config.json``, so it resolves to ``AutoTokenizer``), and an empty
``ModelArch``. A *template variant* -- same checkpoint loading, different chat format -- is a thin
subclass with ``architectures=[]`` so reverse-lookup never lands on it.
"""
from __future__ import annotations

from transformers import AutoProcessor, AutoTokenizer

from swift.dev.utils import safe_snapshot_download
from .base import ModelArch, ModelLoader, register_model


@register_model
class MistralLoader(ModelLoader):

    model_type = 'mistral'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['MistralForCausalLM']
    template = 'llama'
    requires = ['transformers>=4.34']
    models = [
        ('AI-ModelScope/Mistral-7B-Instruct-v0.1', 'mistralai/Mistral-7B-Instruct-v0.1'),
        ('AI-ModelScope/Mistral-7B-Instruct-v0.2', 'mistralai/Mistral-7B-Instruct-v0.2'),
        ('LLM-Research/Mistral-7B-Instruct-v0.3', 'mistralai/Mistral-7B-Instruct-v0.3'),
        ('AI-ModelScope/Mistral-7B-v0.1', 'mistralai/Mistral-7B-v0.1'),
        ('AI-ModelScope/Mistral-7B-v0.2-hf', 'alpindale/Mistral-7B-v0.2-hf'),
        ('swift/Codestral-22B-v0.1', 'mistralai/Codestral-22B-v0.1'),
    ]


@register_model
class MixtralLoader(ModelLoader):
    # The legacy `Mixtral-8x7b-AQLM-2Bit-1x16-hf` checkpoint is dropped: it is a niche 2-bit AQLM
    # variant that needed its own `aqlm` + `torch>=2.2` requires group, which no longer fits the
    # one-requires-per-family shape. See MODEL_MIGRATION.md.
    model_type = 'mixtral'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['MixtralForCausalLM']
    template = 'llama'
    requires = ['transformers>=4.36']
    models = [
        ('AI-ModelScope/Mixtral-8x7B-Instruct-v0.1', 'mistralai/Mixtral-8x7B-Instruct-v0.1'),
        ('AI-ModelScope/Mixtral-8x7B-v0.1', 'mistralai/Mixtral-8x7B-v0.1'),
        ('AI-ModelScope/Mixtral-8x22B-v0.1', 'mistral-community/Mixtral-8x22B-v0.1'),
    ]

    @property
    def model_arch(self) -> ModelArch:
        # A dense LLM otherwise, but its `MixtralSparseMoeBlock` must be a ZeRO-3 leaf (see
        # `apply_z3_leaf_modules`). Replaces legacy's hardcoded `hf_model_type -> class` z3 map entry.
        return ModelArch(moe_block='MixtralSparseMoeBlock')


@register_model
class MistralNemoLoader(ModelLoader):
    # Two legacy groups differed only by transformers version (Nemo >=4.43, Ministral-8B >=4.46);
    # collapsed to the higher bound so neither loads on an unsupported transformers.
    model_type = 'mistral_nemo'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['MistralForCausalLM']
    template = 'mistral_nemo'
    requires = ['transformers>=4.46']
    models = [
        ('AI-ModelScope/Mistral-Small-Instruct-2409', 'mistralai/Mistral-Small-Instruct-2409'),
        ('LLM-Research/Mistral-Large-Instruct-2407', 'mistralai/Mistral-Large-Instruct-2407'),
        ('AI-ModelScope/Mistral-Nemo-Base-2407', 'mistralai/Mistral-Nemo-Base-2407'),
        ('AI-ModelScope/Mistral-Nemo-Instruct-2407', 'mistralai/Mistral-Nemo-Instruct-2407'),
        ('AI-ModelScope/Ministral-8B-Instruct-2410', 'mistralai/Ministral-8B-Instruct-2410'),
    ]


@register_model
class Mistral2501Loader(ModelLoader):

    model_type = 'mistral_2501'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['MistralForCausalLM']
    template = 'mistral_2501'
    models = [
        'mistralai/Mistral-Small-24B-Base-2501',
        'mistralai/Mistral-Small-24B-Instruct-2501',
    ]


@register_model
class ZephyrLoader(ModelLoader):

    model_type = 'zephyr'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['MistralForCausalLM']
    template = 'zephyr'
    requires = ['transformers>=4.34']
    models = [('modelscope/zephyr-7b-beta', 'HuggingFaceH4/zephyr-7b-beta')]


@register_model
class WizardLM2MoeLoader(ModelLoader):

    model_type = 'wizardlm2_moe'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['MixtralForCausalLM']
    template = 'wizardlm2_moe'
    requires = ['transformers>=4.36']
    models = [('AI-ModelScope/WizardLM-2-8x22B', 'alpindale/WizardLM-2-8x22B')]


@register_model
class WizardLM2Loader(ModelLoader):

    model_type = 'wizardlm2'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['MistralForCausalLM']
    template = 'wizardlm2'
    requires = ['transformers>=4.34']
    models = [('AI-ModelScope/WizardLM-2-7B-AWQ', 'MaziyarPanahi/WizardLM-2-7B-AWQ')]


@register_model
class DevstralLoader(ModelLoader):
    """Devstral ships no tokenizer of its own; borrow the Mistral-Small-3.1 one (as sglang does)."""

    model_type = 'devstral'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['MistralForCausalLM']
    template = 'devstral'
    requires = ['transformers>=4.43', 'mistral-common>=1.5.5']
    models = ['mistralai/Devstral-Small-2505']

    def build_processor(self, model_dir, config, **kwargs):
        # src: sglang did the same (https://github.com/sgl-project/sglang/pull/6547)
        tokenizer_dir = safe_snapshot_download('mistralai/Mistral-Small-3.1-24B-Instruct-2503', download_model=False)
        return AutoTokenizer.from_pretrained(tokenizer_dir)


@register_model
class Mistral3Loader(ModelLoader):
    """Mistral-Small-3.1 vision: the root of the Mistral3 multimodal chain (llava_hf partition)."""

    model_type = 'mistral3'
    model_cls = 'transformers:Mistral3ForConditionalGeneration'
    architectures = ['Mistral3ForConditionalGeneration']
    template = 'mistral_2503'
    requires = ['transformers>=4.49']
    tags = ['vision']
    is_multimodal = True
    # Mistral ships its real weights as `consolidated*`, which the download default skips; fetch
    # everything. Inherited by the Ministral-3 / Mistral-Small-3.2 subclasses below.
    ignore_patterns = []
    models = [
        'mistralai/Mistral-Small-3.1-24B-Base-2503',
        'mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    ]

    @property
    def model_arch(self) -> ModelArch:
        # legacy `ModelArch.llava_hf` on transformers>=4.52.
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner='model.multi_modal_projector',
            vision_tower='model.vision_tower',
        )


@register_model
class Ministral3Loader(Mistral3Loader):
    # Split from the legacy `mistral3` group: Ministral-3 needs its own `mistral-common>=1.8.6` +
    # transformers>=5 requires, which cannot share a single family-level requires.
    model_type = 'ministral3'
    template = 'mistral_2512'
    requires = ['transformers>=5.0.0.dev0', 'mistral-common>=1.8.6']
    models = [
        'mistralai/Ministral-3-3B-Base-2512',
        'mistralai/Ministral-3-3B-Instruct-2512',
        'mistralai/Ministral-3-3B-Instruct-2512-BF16',
        'mistralai/Ministral-3-8B-Base-2512',
        'mistralai/Ministral-3-8B-Instruct-2512',
        'mistralai/Ministral-3-8B-Instruct-2512-BF16',
        'mistralai/Ministral-3-14B-Base-2512',
        'mistralai/Ministral-3-14B-Instruct-2512',
        'mistralai/Ministral-3-14B-Instruct-2512-BF16',
    ]


@register_model
class Ministral3ThinkingLoader(Ministral3Loader):
    """Reasoning checkpoints: Ministral-3 loading, a thinking chat template (template variant)."""

    model_type = 'ministral3_thinking'
    template = 'mistral_2512_thinking'
    architectures = []  # reachable by id / explicit --model_type only; base owns the reverse-lookup
    models = [
        'mistralai/Ministral-3-3B-Reasoning-2512',
        'mistralai/Ministral-3-8B-Reasoning-2512',
        'mistralai/Ministral-3-14B-Reasoning-2512',
    ]


@register_model
class Mistral3_2506Loader(Mistral3Loader):
    """Mistral-Small-3.2: 3.1 loading, its own processor borrowed from the 3.1 repo."""

    model_type = 'mistral3_2506'
    template = 'mistral_2506'
    requires = ['transformers>=4.49']
    models = ['mistralai/Mistral-Small-3.2-24B-Instruct-2506']

    def build_processor(self, model_dir, config, **kwargs):
        tokenizer_dir = safe_snapshot_download('mistralai/Mistral-Small-3.1-24B-Instruct-2503', download_model=False)
        return AutoProcessor.from_pretrained(tokenizer_dir)
