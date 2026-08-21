# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the Google Gemma families (text + vision), from ``swift/model/models/gemma.py``.

Migrated: the plain causal-LM ``gemma`` / ``gemma2`` / ``gemma3_text`` and the transformers-native
multimodal ``paligemma`` / ``gemma3_vision`` / ``gemma3n``. Gemma-3 trains best with eager attention
(legacy set it in ``get_config``); ``_EagerAttnDefault`` defaults it while still honouring a user
``--attn_impl`` (dev only puts ``attn_implementation`` in the load kwargs when explicitly requested).

Also migrated: ``gemma_emb`` (EmbeddingGemma), ``gemma4`` / ``gemma4_unified`` and
``diffusion_gemma`` -- see the classes at the bottom. Legacy's ``_patch_gemma4_forward`` (a 200-line
fork of ``forward`` that injects dummy image/audio features so the vision path stays alive under
DeepSpeed ZeRO-3) is NOT ported: dev keeps the vision path alive without forking forward, via the
data-side dummy + ``apply_vision_keep_alive`` aligner hooks (same mechanism as the Qwen-VL loaders).
"""
from __future__ import annotations

from ..keep_alive import apply_vision_keep_alive
from .base import ModelArch, ModelLoader, register_model


class _EagerAttnDefault:
    """Default the attention kernel to ``eager`` (Gemma-3's recommended impl) unless the user picked
    one. dev only injects ``attn_implementation`` when ``--attn_impl`` was set, so ``setdefault``
    never overrides a user choice."""

    def build_model(self, model_dir, config, processor, **kwargs):
        kwargs.setdefault('attn_implementation', 'eager')
        return super().build_model(model_dir, config, processor, **kwargs)


@register_model
class GemmaLoader(ModelLoader):
    model_type = 'gemma'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['GemmaForCausalLM']
    template = 'gemma'
    requires = ['transformers>=4.38']
    models = [
        ('AI-ModelScope/gemma-2b-it', 'google/gemma-2b-it'),
        ('AI-ModelScope/gemma-2b', 'google/gemma-2b'),
        ('AI-ModelScope/gemma-7b', 'google/gemma-7b'),
        ('AI-ModelScope/gemma-7b-it', 'google/gemma-7b-it'),
    ]


@register_model
class Gemma2Loader(ModelLoader):
    model_type = 'gemma2'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['Gemma2ForCausalLM']
    template = 'gemma'
    requires = ['transformers>=4.42']
    models = [
        ('LLM-Research/gemma-2-2b-it', 'google/gemma-2-2b-it'),
        ('LLM-Research/gemma-2-2b', 'google/gemma-2-2b'),
        ('LLM-Research/gemma-2-9b', 'google/gemma-2-9b'),
        ('LLM-Research/gemma-2-9b-it', 'google/gemma-2-9b-it'),
        ('LLM-Research/gemma-2-27b', 'google/gemma-2-27b'),
        ('LLM-Research/gemma-2-27b-it', 'google/gemma-2-27b-it'),
    ]


@register_model
class Gemma3TextLoader(_EagerAttnDefault, ModelLoader):
    model_type = 'gemma3_text'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['Gemma3ForCausalLM']
    template = 'gemma3_text'
    requires = ['transformers>=4.49']
    models = [
        ('LLM-Research/gemma-3-1b-pt', 'google/gemma-3-1b-pt'),
        ('LLM-Research/gemma-3-1b-it', 'google/gemma-3-1b-it'),
        ('google/gemma-3-270m', 'google/gemma-3-270m'),
        ('google/gemma-3-270m-it', 'google/gemma-3-270m-it'),
        ('google/medgemma-27b-text-it', 'google/medgemma-27b-text-it'),
    ]


@register_model
class PaligemmaLoader(ModelLoader):
    model_type = 'paligemma'
    model_cls = 'transformers:PaliGemmaForConditionalGeneration'
    architectures = ['PaliGemmaForConditionalGeneration']
    template = 'paligemma'
    requires = ['transformers>=4.41']
    tags = ['vision']
    is_multimodal = True
    models = [
        ('AI-ModelScope/paligemma-3b-pt-224', 'google/paligemma-3b-pt-224'),
        ('AI-ModelScope/paligemma-3b-pt-448', 'google/paligemma-3b-pt-448'),
        ('AI-ModelScope/paligemma-3b-pt-896', 'google/paligemma-3b-pt-896'),
        ('AI-ModelScope/paligemma-3b-mix-224', 'google/paligemma-3b-mix-224'),
        ('AI-ModelScope/paligemma-3b-mix-448', 'google/paligemma-3b-mix-448'),
        ('AI-ModelScope/paligemma2-3b-pt-224', 'google/paligemma2-3b-pt-224'),
        ('AI-ModelScope/paligemma2-3b-pt-448', 'google/paligemma2-3b-pt-448'),
        ('AI-ModelScope/paligemma2-3b-pt-896', 'google/paligemma2-3b-pt-896'),
        ('AI-ModelScope/paligemma2-10b-pt-224', 'google/paligemma2-10b-pt-224'),
        ('AI-ModelScope/paligemma2-10b-pt-448', 'google/paligemma2-10b-pt-448'),
        ('AI-ModelScope/paligemma2-10b-pt-896', 'google/paligemma2-10b-pt-896'),
        ('AI-ModelScope/paligemma2-28b-pt-224', 'google/paligemma2-28b-pt-224'),
        ('AI-ModelScope/paligemma2-28b-pt-448', 'google/paligemma2-28b-pt-448'),
        ('AI-ModelScope/paligemma2-28b-pt-896', 'google/paligemma2-28b-pt-896'),
        ('AI-ModelScope/paligemma2-3b-ft-docci-448', 'google/paligemma2-3b-ft-docci-448'),
        ('AI-ModelScope/paligemma2-10b-ft-docci-448', 'google/paligemma2-10b-ft-docci-448'),
    ]

    @property
    def model_arch(self) -> ModelArch:
        # legacy `ModelArch.llava_hf`, transformers>=4.52 (model.* prefix) branch.
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner='model.multi_modal_projector',
            vision_tower='model.vision_tower',
        )


@register_model
class Gemma3VisionLoader(_EagerAttnDefault, ModelLoader):
    model_type = 'gemma3_vision'
    model_cls = 'transformers:Gemma3ForConditionalGeneration'
    architectures = ['Gemma3ForConditionalGeneration']
    template = 'gemma3_vision'
    requires = ['transformers>=4.49']
    tags = ['vision']
    is_multimodal = True
    models = [
        ('LLM-Research/gemma-3-4b-pt', 'google/gemma-3-4b-pt'),
        ('LLM-Research/gemma-3-4b-it', 'google/gemma-3-4b-it'),
        ('LLM-Research/gemma-3-12b-pt', 'google/gemma-3-12b-pt'),
        ('LLM-Research/gemma-3-12b-it', 'google/gemma-3-12b-it'),
        ('LLM-Research/gemma-3-27b-pt', 'google/gemma-3-27b-pt'),
        ('LLM-Research/gemma-3-27b-it', 'google/gemma-3-27b-it'),
        ('google/medgemma-4b-pt', 'google/medgemma-4b-pt'),
        ('google/medgemma-4b-it', 'google/medgemma-4b-it'),
        ('google/medgemma-27b-it', 'google/medgemma-27b-it'),
    ]

    @property
    def model_arch(self) -> ModelArch:
        # legacy `ModelArch.llava_hf`, transformers>=4.52 (model.* prefix) branch.
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner='model.multi_modal_projector',
            vision_tower='model.vision_tower',
        )


@register_model
class Gemma3nLoader(ModelLoader):
    """Gemma-3n (vision + audio). Legacy also ran ``patch_output_to_input_device`` on the vision/audio
    embedders for HF ``device_map`` multi-GPU; dev uses twinkle strategies (no HF device_map, see
    PATCH_INVENTORY.md), so that patch is obsolete and dropped."""

    model_type = 'gemma3n'
    model_cls = 'transformers:Gemma3nForConditionalGeneration'
    architectures = ['Gemma3nForConditionalGeneration']
    template = 'gemma3n'
    requires = ['transformers>=4.53.1']
    is_multimodal = True
    models = [
        ('google/gemma-3n-E2B', 'google/gemma-3n-E2B'),
        ('google/gemma-3n-E4B', 'google/gemma-3n-E4B'),
        ('google/gemma-3n-E2B-it', 'google/gemma-3n-E2B-it'),
        ('google/gemma-3n-E4B-it', 'google/gemma-3n-E4B-it'),
    ]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner=['model.embed_vision', 'model.embed_audio'],
            vision_tower=['model.vision_tower', 'model.audio_tower'],
        )


@register_model
class GemmaEmbLoader(ModelLoader):
    """EmbeddingGemma: a ``Gemma3TextModel`` backbone served as a sentence embedder. Legacy loaded it
    through ``SentenceTransformersLoader`` (``SentenceTransformer(model_dir)``); on dev that pipeline
    construction lives at the model layer (``swift/dev/model/sentence_transformer_model.py``), so the
    loader only carries registration facts and pins the intrinsic embedding task. ``Gemma3TextModel``
    is its own architecture (distinct from ``gemma3_text``'s ``Gemma3ForCausalLM``), so declaring it
    does not collide with any other family on reverse-lookup.
    """

    model_type = 'gemma_emb'
    model_cls = 'transformers:Gemma3TextModel'
    architectures = ['Gemma3TextModel']
    template = 'dummy'
    task_type = 'embedding'
    models = [('google/embeddinggemma-300m', 'google/embeddinggemma-300m')]


class _Gemma4KeepAlive:
    """Shared vision/audio keep-alive for the gemma4 family: on a text-only ZeRO-3 micro-batch the
    stock forward would skip the vision/audio towers, darkening their params and deadlocking the
    cross-rank all-gather. The data side feeds a dummy image and these aligner hooks zero its
    contribution -- the non-forking replacement for legacy ``_patch_gemma4_forward``."""

    def process_model(self, model):
        model = super().process_model(model)
        model._vision_keep_alive = apply_vision_keep_alive(model, self.model_arch.aligner)
        return model


@register_model
class Gemma4Loader(_Gemma4KeepAlive, ModelLoader):
    """Gemma-4 (vision + audio, MoE). Reverse-lookup owner for ``Gemma4ForConditionalGeneration``; its
    base group is the ``gemma4_nothinking`` (E2B/E4B) template. MoE sparse block is ``Gemma4TextExperts``
    (the z3 leaf legacy set by hand). Shares the ``gemma3n`` module layout."""

    model_type = 'gemma4'
    model_cls = 'transformers:Gemma4ForConditionalGeneration'
    architectures = ['Gemma4ForConditionalGeneration']
    template = 'gemma4_nothinking'
    tags = ['vision', 'audio']
    is_multimodal = True
    is_moe = True

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner=['model.embed_vision', 'model.embed_audio'],
            vision_tower=['model.vision_tower', 'model.audio_tower'],
            moe_block=['Gemma4TextExperts'],
        )

    models = [
        ('google/gemma-4-E2B', 'google/gemma-4-E2B'),
        ('google/gemma-4-E2B-it', 'google/gemma-4-E2B-it'),
        ('google/gemma-4-E4B', 'google/gemma-4-E4B'),
        ('google/gemma-4-E4B-it', 'google/gemma-4-E4B-it'),
    ]


@register_model
class Gemma4ThinkingLoader(Gemma4Loader):
    """The larger gemma4 checkpoints, which speak the (thinking) ``gemma4`` template."""

    model_type = 'gemma4_thinking'
    architectures = []  # template variant of the gemma4 architecture; owner stays `gemma4`
    template = 'gemma4'
    models = [
        ('google/gemma-4-31B', 'google/gemma-4-31B'),
        ('google/gemma-4-31B-it', 'google/gemma-4-31B-it'),
        ('google/gemma-4-26B-A4B', 'google/gemma-4-26B-A4B'),
        ('google/gemma-4-26B-A4B-it', 'google/gemma-4-26B-A4B-it'),
    ]


@register_model
class Gemma4UnifiedLoader(_Gemma4KeepAlive, ModelLoader):
    """Gemma-4 Unified: a text+vision variant whose module layout drops the separate vision_tower part
    (``gemma4_unified`` arch: LLM + embed_vision/embed_audio aligners only). ``Gemma4Unified*`` is not
    in transformers 5.5 (needs >=5.10.1); the ``model_cls`` string is resolved lazily so the absent
    class costs nothing until a new-enough transformers is installed."""

    model_type = 'gemma4_unified'
    model_cls = 'transformers:Gemma4UnifiedForConditionalGeneration'
    architectures = ['Gemma4UnifiedForConditionalGeneration']
    template = 'gemma4'
    requires = ['transformers>=5.10.1']
    tags = ['vision', 'audio']
    is_multimodal = True

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner=['model.embed_vision', 'model.embed_audio'],
        )

    models = [
        ('google/gemma-4-12B', 'google/gemma-4-12B'),
        ('google/gemma-4-12B-it', 'google/gemma-4-12B-it'),
    ]


@register_model
class DiffusionGemmaLoader(ModelLoader):
    """DiffusionGemma: block-diffusion generation (encoder LLM + decoder), not autoregressive.
    ``DiffusionGemmaForBlockDiffusion`` is not in transformers 5.5 (needs >=5.11); ``model_cls`` is
    resolved lazily. ``process_model`` keeps legacy's two generation-time settings: block diffusion
    does not use the AR ``prepare_inputs_for_generation`` path (cleared) and relies on a KV cache
    (``use_cache=True``). No vision keep-alive: its forward does not branch on media presence.
    """

    model_type = 'diffusion_gemma'
    model_cls = 'transformers:DiffusionGemmaForBlockDiffusion'
    architectures = ['DiffusionGemmaForBlockDiffusion']
    template = 'diffusion_gemma'
    requires = ['transformers>=5.11']
    tags = ['vision']
    is_multimodal = True

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['model.encoder.language_model', 'model.decoder', 'lm_head'],
            aligner=['model.encoder.embed_vision'],
            vision_tower=['model.encoder.vision_tower'],
        )

    models = [('google/diffusiongemma-26B-A4B-it', 'google/diffusiongemma-26B-A4B-it')]

    def process_model(self, model):
        model.prepare_inputs_for_generation = None
        model.config.use_cache = True
        return model
