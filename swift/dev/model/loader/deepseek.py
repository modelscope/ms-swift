# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the DeepSeek text families (from the text blocks of ``swift/model/models/deepseek.py``).

Legacy's ``DeepseekLoader.get_model`` only ran ``patch_output_to_input_device`` on every MLP -- an HF
device-map placement patch, obsolete per PATCH_INVENTORY (twinkle owns placement) -- so it is dropped
and these become plain loaders. Following the llama/qwen text pattern: one base per architecture (the
reverse-lookup owner) plus ``architectures=[]`` template-variant subclasses. Text models keep the
default empty ``ModelArch``; the MoE families are flagged ``is_moe`` (no ``moe_block`` yet, same as the
qwen3_moe / phi3_moe call: z3 is unwired and the leaf class name is not guessed).

``deepseek`` (v1 MoE) is remote-code on transformers 5.5 (no in-tree ``DeepseekForCausalLM``), loaded
via the g7 ``trust_remote_code`` flag + ``AutoModelForCausalLM``.

``deepseek_v32`` / ``deepseek_v4`` are Megatron-primary and their architectures are not in
transformers 5.5, so on the HF path they load through remote code (g7 ``trust_remote_code``).
``deepseek_v32`` additionally keeps legacy's V3 fallback: when the checkpoint ships no
``deepseek_v32`` code, ``DeepseekV3ForCausalLM`` loads it instead (weight-compatible; the extra
V3.2 attention config is ignored) -- enough for Megatron conversion / vLLM-SGLang inference while
we wait for the HF arch to land.

Not migrated here (see MODEL_MIGRATION.md):
  * the ``moonlight`` group of legacy ``deepseek_v3`` -- pinned ``transformers<4.49`` (dead on 5.5).
  * the ``deepseek_r1`` *distill* groups living under legacy qwen2/qwen3/llama (Qwen/Llama arch,
    DeepSeek-branded) -- they reverse-lookup to the qwen2/qwen3/llama base; a dedicated distill
    model_type is a later call.
"""
from __future__ import annotations

from .base import ModelArch, ModelLoader, register_model


@register_model
class DeepseekLoader(ModelLoader):
    """DeepSeek-MoE v1 (2024): remote-code ``DeepseekForCausalLM`` via the g7 trust_remote_code flag."""

    model_type = 'deepseek'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['DeepseekForCausalLM']
    template = 'deepseek'
    is_moe = True
    models = [
        ('deepseek-ai/deepseek-moe-16b-chat', 'deepseek-ai/deepseek-moe-16b-chat'),
        ('deepseek-ai/deepseek-moe-16b-base', 'deepseek-ai/deepseek-moe-16b-base'),
    ]


@register_model
class DeepseekLLMLoader(ModelLoader):
    """DeepSeek-LLM (2023-11): the first-generation *dense* DeepSeek models. Plain
    ``LlamaForCausalLM`` checkpoints that speak the ``deepseek`` template, so ``architectures`` is
    empty -- the class name belongs to the ``llama`` family and these must not win reverse-lookup.

    Legacy filed them under the ``llama`` model_type as three template groups differing only in
    ``tags``; dev carries ``tags`` per model_type, so the math / coder lines are separate loaders
    below rather than a merged tag union.
    """

    model_type = 'deepseek_llm'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = []
    template = 'deepseek'
    models = [
        'deepseek-ai/deepseek-llm-7b-base',
        'deepseek-ai/deepseek-llm-7b-chat',
        'deepseek-ai/deepseek-llm-67b-base',
        'deepseek-ai/deepseek-llm-67b-chat',
    ]


@register_model
class DeepseekMathLoader(DeepseekLLMLoader):
    model_type = 'deepseek_math'
    tags = ['math']
    models = [
        'deepseek-ai/deepseek-math-7b-base',
        'deepseek-ai/deepseek-math-7b-instruct',
        'deepseek-ai/deepseek-math-7b-rl',
    ]


@register_model
class DeepseekCoderLoader(DeepseekLLMLoader):
    model_type = 'deepseek_coder'
    tags = ['coding']
    models = [
        'deepseek-ai/deepseek-coder-1.3b-base',
        'deepseek-ai/deepseek-coder-1.3b-instruct',
        'deepseek-ai/deepseek-coder-6.7b-base',
        'deepseek-ai/deepseek-coder-6.7b-instruct',
        'deepseek-ai/deepseek-coder-33b-base',
        'deepseek-ai/deepseek-coder-33b-instruct',
    ]


@register_model
class DeepseekV2Loader(ModelLoader):
    """DeepSeek-V2 / Coder-V2 (MoE); reverse-lookup owner for ``DeepseekV2ForCausalLM``."""

    model_type = 'deepseek_v2'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['DeepseekV2ForCausalLM']
    template = 'deepseek'
    requires = ['transformers>=4.39.3']
    is_moe = True
    models = [
        ('deepseek-ai/DeepSeek-Coder-V2-Instruct', 'deepseek-ai/DeepSeek-Coder-V2-Instruct'),
        ('deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'),
        ('deepseek-ai/DeepSeek-Coder-V2-Base', 'deepseek-ai/DeepSeek-Coder-V2-Base'),
        ('deepseek-ai/DeepSeek-Coder-V2-Lite-Base', 'deepseek-ai/DeepSeek-Coder-V2-Lite-Base'),
        ('deepseek-ai/DeepSeek-V2-Lite', 'deepseek-ai/DeepSeek-V2-Lite'),
        ('deepseek-ai/DeepSeek-V2-Lite-Chat', 'deepseek-ai/DeepSeek-V2-Lite-Chat'),
        ('deepseek-ai/DeepSeek-V2', 'deepseek-ai/DeepSeek-V2'),
        ('deepseek-ai/DeepSeek-V2-Chat', 'deepseek-ai/DeepSeek-V2-Chat'),
    ]


@register_model
class DeepseekV2_5Loader(DeepseekV2Loader):
    model_type = 'deepseek_v2_5'
    architectures = []
    template = 'deepseek_v2_5'
    models = [
        ('deepseek-ai/DeepSeek-V2.5', 'deepseek-ai/DeepSeek-V2.5'),
        ('deepseek-ai/DeepSeek-V2.5-1210', 'deepseek-ai/DeepSeek-V2.5-1210'),
    ]


@register_model
class DeepseekV3Loader(ModelLoader):
    """DeepSeek-V3 / R1 / V3.1 (MoE); reverse-lookup owner for ``DeepseekV3ForCausalLM``. Its base
    group carries the ``deepseek_v2_5`` template; R1 / V3.1 / Kimi-K2 are template variants."""

    model_type = 'deepseek_v3'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['DeepseekV3ForCausalLM']
    template = 'deepseek_v2_5'
    requires = ['transformers>=4.39.3']
    is_moe = True
    models = [
        ('deepseek-ai/DeepSeek-V3-Base', 'deepseek-ai/DeepSeek-V3-Base'),
        ('deepseek-ai/DeepSeek-V3', 'deepseek-ai/DeepSeek-V3'),
        ('deepseek-ai/DeepSeek-V3-0324', 'deepseek-ai/DeepSeek-V3-0324'),
        ('cognitivecomputations/DeepSeek-V3-awq', 'cognitivecomputations/DeepSeek-V3-AWQ'),
        ('cognitivecomputations/DeepSeek-V3-0324-AWQ', 'cognitivecomputations/DeepSeek-V3-0324-AWQ'),
        ('deepseek-ai/DeepSeek-Prover-V2-7B', 'deepseek-ai/DeepSeek-Prover-V2-7B'),
        ('deepseek-ai/DeepSeek-Prover-V2-671B', 'deepseek-ai/DeepSeek-Prover-V2-671B'),
        ('unsloth/DeepSeek-V3-bf16', 'unsloth/DeepSeek-V3-bf16'),
        ('unsloth/DeepSeek-V3-0324-BF16', 'unsloth/DeepSeek-V3-0324-BF16'),
        ('unsloth/DeepSeek-Prover-V2-671B-BF16', 'unsloth/DeepSeek-Prover-V2-671B-BF16'),
    ]


@register_model
class DeepseekR1Loader(DeepseekV3Loader):
    model_type = 'deepseek_r1'
    architectures = []
    template = 'deepseek_r1'
    models = [
        ('deepseek-ai/DeepSeek-R1', 'deepseek-ai/DeepSeek-R1'),
        ('deepseek-ai/DeepSeek-R1-Zero', 'deepseek-ai/DeepSeek-R1-Zero'),
        ('deepseek-ai/DeepSeek-R1-0528', 'deepseek-ai/DeepSeek-R1-0528'),
        ('cognitivecomputations/DeepSeek-R1-awq', 'cognitivecomputations/DeepSeek-R1-AWQ'),
        ('cognitivecomputations/DeepSeek-R1-0528-AWQ', 'cognitivecomputations/DeepSeek-R1-0528-AWQ'),
        ('unsloth/DeepSeek-R1-BF16', 'unsloth/DeepSeek-R1-BF16'),
        ('unsloth/DeepSeek-R1-Zero-BF16', 'unsloth/DeepSeek-R1-Zero-BF16'),
        ('unsloth/DeepSeek-R1-0528-BF16', 'unsloth/DeepSeek-R1-0528-BF16'),
    ]


@register_model
class DeepseekV3_1Loader(DeepseekV3Loader):
    model_type = 'deepseek_v3_1'
    architectures = []
    template = 'deepseek_v3_1'
    models = [
        ('deepseek-ai/DeepSeek-V3.1-Base', 'deepseek-ai/DeepSeek-V3.1-Base'),
        ('deepseek-ai/DeepSeek-V3.1', 'deepseek-ai/DeepSeek-V3.1'),
        ('deepseek-ai/DeepSeek-V3.1-Terminus', 'deepseek-ai/DeepSeek-V3.1-Terminus'),
    ]


@register_model
class KimiK2Loader(DeepseekV3Loader):
    model_type = 'kimi_k2'
    architectures = []
    template = 'kimi_k2'
    models = [
        ('moonshotai/Kimi-K2-Base', 'moonshotai/Kimi-K2-Base'),
        ('moonshotai/Kimi-K2-Instruct', 'moonshotai/Kimi-K2-Instruct'),
        ('moonshotai/Kimi-K2-Instruct-0905', 'moonshotai/Kimi-K2-Instruct-0905'),
        ('moonshotai/Kimi-K2-Thinking', 'moonshotai/Kimi-K2-Thinking'),
    ]


# ---------------------------- R1 distill checkpoints ----------------------------
# The `deepseek_r1` template spans three architectures: the original R1 is DeepseekV3 (above), while
# the distills are plain Llama / Qwen3 checkpoints -- legacy filed them as `deepseek_r1`-template
# groups under the `llama` / `qwen3` model_types. Since a dev model_type must be unique, the V3 one
# keeps the bare `deepseek_r1` name and the distills take family-qualified names (the same convention
# used for qwen3_moe_thinking / qwen3_next_thinking). Both declare ``architectures=[]`` so
# reverse-lookup for LlamaForCausalLM / Qwen3ForCausalLM still resolves to `llama` / `qwen3`.


@register_model
class DeepseekR1DistillLlamaLoader(ModelLoader):
    """R1 distilled into Llama (2025-01): a plain in-tree ``LlamaForCausalLM``."""

    model_type = 'deepseek_r1_distill_llama'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = []  # template variant of the llama architecture; owner stays `llama`
    template = 'deepseek_r1'
    models = [
        ('deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'),
        ('deepseek-ai/DeepSeek-R1-Distill-Llama-70B', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'),
    ]


@register_model
class DeepseekR1DistillQwen3Loader(ModelLoader):
    """R1-0528 distilled into Qwen3 (2025-05): a plain in-tree ``Qwen3ForCausalLM``."""

    model_type = 'deepseek_r1_distill_qwen3'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = []  # template variant of the qwen3 architecture; owner stays `qwen3`
    template = 'deepseek_r1'
    requires = ['transformers>=4.51']
    models = [('deepseek-ai/DeepSeek-R1-0528-Qwen3-8B', 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')]


@register_model
class DeepseekR1DistillQwen2Loader(ModelLoader):
    """R1 distilled into Qwen2.5 (2025-01), plus QwenLong-L1 built on the same base: plain in-tree
    ``Qwen2ForCausalLM``. Legacy filed both as ``deepseek_r1``-template groups under ``qwen2``."""

    model_type = 'deepseek_r1_distill_qwen2'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = []  # template variant of the qwen2 architecture; owner stays `qwen2`
    template = 'deepseek_r1'
    requires = ['transformers>=4.37']
    models = [
        ('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'),
        ('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'),
        ('deepseek-ai/DeepSeek-R1-Distill-Qwen-14B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'),
        ('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'),
        ('iic/QwenLong-L1-32B', 'Tongyi-Zhiwen/QwenLong-L1-32B'),
    ]


# ---------------------------- V3.2 / V4 (Megatron-primary) ----------------------------


@register_model
class DeepseekV32Loader(DeepseekV3Loader):
    """DeepSeek-V3.2 (incl. V3.2-Exp / -Speciale and DeepSeek-Math-V2). The ``DeepseekV32*`` classes
    are not in transformers 5.5; the checkpoint ships its own modeling code, so it loads through the
    g7 ``trust_remote_code`` path (``AutoModelForCausalLM`` resolves the checkpoint's own class).

    ``build_config`` keeps legacy's fallback: V3.2 only adds a sparse-attention (indexer) block on top
    of V3's config, so when neither an in-tree ``deepseek_v32`` module nor checkpoint code is present,
    a plain ``DeepseekV3Config`` still parses the file -- enough for Megatron conversion / vLLM-SGLang
    inference. Shares V3's ``deepseek_v3_1`` template.
    """

    model_type = 'deepseek_v32'
    architectures = ['DeepseekV32ForCausalLM']
    template = 'deepseek_v3_1'
    trust_remote_code = True

    def build_config(self, model_dir: str, **kwargs):
        try:
            from transformers.models.deepseek_v32 import DeepseekV32Config
        except ImportError:
            from transformers.models.deepseek_v3 import DeepseekV3Config as DeepseekV32Config
        kwargs.setdefault('trust_remote_code', True)
        return DeepseekV32Config.from_pretrained(model_dir, **kwargs)

    models = [
        ('deepseek-ai/DeepSeek-V3.2', 'deepseek-ai/DeepSeek-V3.2'),
        ('deepseek-ai/DeepSeek-V3.2-Speciale', 'deepseek-ai/DeepSeek-V3.2-Speciale'),
        ('deepseek-ai/DeepSeek-V3.2-Exp', 'deepseek-ai/DeepSeek-V3.2-Exp'),
        ('deepseek-ai/DeepSeek-V3.2-Exp-Base', 'deepseek-ai/DeepSeek-V3.2-Exp-Base'),
        ('deepseek-ai/DeepSeek-Math-V2', 'deepseek-ai/DeepSeek-Math-V2'),
    ]


@register_model
class DeepseekV4Loader(ModelLoader):
    """DeepSeek-V4 (Flash / Pro). ``DeepseekV4ForCausalLM`` is not in transformers 5.5 and V4 is
    Megatron-primary (the real training path lives under ``swift/megatron`` with a custom MLA RoPE
    shape contract); on the HF path the checkpoint's own modeling code loads through the g7
    ``trust_remote_code`` flag. The reverse-lookup owner for ``DeepseekV4ForCausalLM``. Legacy
    declared no loader body (plain base), so nothing else is ported.
    """

    model_type = 'deepseek_v4'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['DeepseekV4ForCausalLM']
    template = 'deepseek_v4'
    trust_remote_code = True
    is_moe = True
    models = [
        ('deepseek-ai/DeepSeek-V4-Flash', 'deepseek-ai/DeepSeek-V4-Flash'),
        ('deepseek-ai/DeepSeek-V4-Flash-Base', 'deepseek-ai/DeepSeek-V4-Flash-Base'),
        ('deepseek-ai/DeepSeek-V4-Pro', 'deepseek-ai/DeepSeek-V4-Pro'),
        ('deepseek-ai/DeepSeek-V4-Pro-Base', 'deepseek-ai/DeepSeek-V4-Pro-Base'),
    ]


@register_model
class DeepseekOCR2Loader(ModelLoader):
    """DeepSeek-OCR-2: remote-code ``DeepseekOCR2ForCausalLM`` -- a SAM tower plus a Qwen2 tower feeding
    a projector into the decoder (that second tower is the only structural difference from DeepSeek-OCR,
    which uses ``vision_model``; legacy expressed it as a ``visual_name`` class attribute).

    Legacy's ``get_model`` body is entirely obsolete on dev -- ``patch_output_clone`` on the token
    embedding plus ``patch_output_to_input_device`` on the SAM tower, the visual tower and the projector,
    all of it HF ``device_map`` cross-device plumbing (twinkle owns placement; see PATCH_INVENTORY.md).
    With those dropped only the declaration remains, so the loader is deliberately thin.

    Legacy also wrapped processor construction in ``try: AutoProcessor except: AutoTokenizer``, to serve
    inference backends that want a tokenizer without triggering remote-code config loading. dev's base
    ``build_processor`` already picks between the two by inspecting the checkpoint's files, which covers
    the same case without swallowing exceptions.

    **Not usable on transformers 5.5 as it stands.** The ``transformers==4.46.3`` pin is kept: it spans
    nine minor versions from the dev environment, and the checkpoint's own ``modeling_*.py`` was written
    against that era's internals -- unlike ``qwen3_asr``, where the single incompatibility was
    identified and shimmed, nothing here has been narrowed down (``easydict`` is also missing). The
    declaration lands now so the id resolves and the version check reports the real conflict; relaxing
    the pin needs an actual load attempt against the checkpoint.
    """

    model_type = 'deepseek_ocr2'
    model_cls = 'transformers:AutoModel'
    architectures = ['DeepseekOCR2ForCausalLM']
    template = 'deepseek_ocr2'
    trust_remote_code = True
    requires = ['transformers==4.46.3', 'easydict']
    tags = ['vision']
    is_multimodal = True
    models = [('deepseek-ai/DeepSeek-OCR-2', 'deepseek-ai/DeepSeek-OCR-2')]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['model.layers'],
            aligner=['model.projector'],
            vision_tower=['model.sam_model', 'model.qwen2_model'],
        )
