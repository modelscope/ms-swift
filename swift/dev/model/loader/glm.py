# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the GLM / GLM-4 / GLM-4V families, from ``swift/model/models/glm.py``.

Only the transformers-native GLM-4-era models are migrated here. Legacy packed several chat formats
into one ``model_type`` via per-``ModelGroup`` ``template`` overrides; dev has one ``template`` per
loader, so each format becomes its own thin subclass with ``architectures=[]`` (a *template variant*
that must never win reverse-lookup -- only the base family owns the HF class name). The invented
variant ids follow the template name (matching the llama convention): ``glm4_z1_rumination``,
``glm4_7``, ``glm5_1``, ``glm5_2``, ``glm4_5v``.

Every legacy GLM-4V loader carried ``patch_get_input_embeddings(model.visual, 'patch_embed')``;
PATCH_INVENTORY.md marks that obsolete, so it is dropped and the vision loaders are plain.

Not migrated here (bucket C, see MODEL_MIGRATION.md): ``chatglm2``/``chatglm3`` + ``codegeex4``
(same remote-code ``ChatGLMModel`` as ``chatglm4`` below, but pinned ``transformers<4.42`` -- dead on
5.5), and ``cogvlm``/``cogagent_*``/``cogvlm2`` (remote-code, ``transformers<4.42``, borrowed vicuna
tokenizer).
"""
from __future__ import annotations

from .base import ModelArch, ModelLoader, register_model


# ------------------------------- text, dense -------------------------------
@register_model
class Glm4Loader(ModelLoader):
    model_type = 'glm4'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['Glm4ForCausalLM']
    template = 'glm4'
    requires = ['transformers>=4.51']
    models = [
        'ZhipuAI/GLM-4-9B-0414',
        'ZhipuAI/GLM-4-32B-0414',
        'ZhipuAI/GLM-4-32B-Base-0414',
        'ZhipuAI/GLM-Z1-9B-0414',
        'ZhipuAI/GLM-Z1-32B-0414',
    ]


@register_model
class Glm4Z1RuminationLoader(Glm4Loader):
    model_type = 'glm4_z1_rumination'
    template = 'glm4_z1_rumination'
    architectures = []  # template variant of glm4; keep reverse-lookup on the base
    models = ['ZhipuAI/GLM-Z1-Rumination-32B-0414']


@register_model
class GlmEdgeLoader(ModelLoader):
    model_type = 'glm_edge'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['GlmForCausalLM']
    template = 'chatglm4'
    requires = ['transformers>=4.46']
    models = ['ZhipuAI/glm-edge-1.5b-chat', 'ZhipuAI/glm-edge-4b-chat']


# ------------------------------- text, MoE -------------------------------
@register_model
class Glm4MoeLoader(ModelLoader):
    model_type = 'glm4_moe'
    model_cls = 'transformers:Glm4MoeForCausalLM'
    architectures = ['Glm4MoeForCausalLM']
    template = 'glm4_5'
    requires = ['transformers>=4.54']
    models = [
        'ZhipuAI/GLM-4.5-Air-Base',
        'ZhipuAI/GLM-4.5-Air',
        'ZhipuAI/GLM-4.5-Air-FP8',
        'ZhipuAI/GLM-4.5-Base',
        'ZhipuAI/GLM-4.5',
        'ZhipuAI/GLM-4.5-FP8',
        'ZhipuAI/GLM-4.6',
        'ZhipuAI/GLM-4.6-FP8',
    ]

    @property
    def model_arch(self) -> ModelArch:
        # dense-shaped LLM whose `Glm4MoeMoE` block must be a ZeRO-3 leaf.
        return ModelArch(moe_block='Glm4MoeMoE')


@register_model
class Glm4_7Loader(Glm4MoeLoader):
    model_type = 'glm4_7'
    template = 'glm4_7'
    architectures = []  # template variant of glm4_moe (same Glm4MoeForCausalLM)
    models = ['ZhipuAI/GLM-4.7', 'ZhipuAI/GLM-4.7-FP8']


@register_model
class Glm4MoeLiteLoader(ModelLoader):
    model_type = 'glm4_moe_lite'
    model_cls = 'transformers:Glm4MoeLiteForCausalLM'
    architectures = ['Glm4MoeLiteForCausalLM']
    template = 'glm4_7'
    requires = ['transformers>=5.0.0.dev']
    models = ['ZhipuAI/GLM-4.7-Flash']

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(moe_block='Glm4MoeLiteMoE')


@register_model
class GlmMoeDsaLoader(ModelLoader):
    model_type = 'glm_moe_dsa'
    model_cls = 'transformers:GlmMoeDsaForCausalLM'
    architectures = ['GlmMoeDsaForCausalLM']
    template = 'glm4_7'
    requires = ['transformers>=5.2.0']
    models = ['ZhipuAI/GLM-5']

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(moe_block='GlmMoeDsaMoE')


@register_model
class Glm5_1Loader(GlmMoeDsaLoader):
    model_type = 'glm5_1'
    template = 'glm5_1'
    architectures = []  # template variant of glm_moe_dsa
    models = ['ZhipuAI/GLM-5.1', 'ZhipuAI/GLM-5.1-FP8']


@register_model
class Glm5_2Loader(GlmMoeDsaLoader):
    model_type = 'glm5_2'
    template = 'glm5_2'
    architectures = []  # template variant of glm_moe_dsa
    models = ['ZhipuAI/GLM-5.2', 'ZhipuAI/GLM-5.2-FP8']


# ------------------------------- vision -------------------------------
@register_model
class Glm4vLoader(ModelLoader):
    model_type = 'glm4v'
    model_cls = 'transformers:Glm4vForConditionalGeneration'
    architectures = ['Glm4vForConditionalGeneration']
    template = 'glm4v'
    requires = ['transformers>=4.53']
    is_multimodal = True
    models = ['ZhipuAI/GLM-4.1V-9B-Base', 'ZhipuAI/GLM-4.1V-9B-Thinking', 'ZhipuAI/AutoGLM-Phone-9B']

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner='model.visual.merger',
            vision_tower='model.visual',
        )


@register_model
class Glm4_5vLoader(Glm4vLoader):
    model_type = 'glm4_5v'
    template = 'glm4_5v'
    architectures = []  # template variant of glm4v (dense Glm4vForConditionalGeneration)
    # highest floor across the two merged groups (Glyph needs >=4.57, GLM-4.6V-Flash >=5.0.0.dev)
    requires = ['transformers>=5.0.0.dev']
    models = ['ZhipuAI/Glyph', 'ZhipuAI/GLM-4.6V-Flash']


@register_model
class Glm4vMoeLoader(Glm4vLoader):
    model_type = 'glm4v_moe'
    model_cls = 'transformers:Glm4vMoeForConditionalGeneration'
    architectures = ['Glm4vMoeForConditionalGeneration']
    template = 'glm4_5v'
    # highest floor: base group needs >=4.56, GLM-4.6V group >=5.0.0.dev
    requires = ['transformers>=5.0.0.dev']
    models = ['ZhipuAI/GLM-4.5V', 'ZhipuAI/GLM-4.5V-FP8', 'ZhipuAI/GLM-4.6V', 'ZhipuAI/GLM-4.6V-FP8']

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner='model.visual.merger',
            vision_tower='model.visual',
            moe_block='Glm4vMoeTextMoE',
        )


@register_model
class GlmOcrLoader(Glm4vLoader):
    model_type = 'glm_ocr'
    model_cls = 'transformers:AutoModelForImageTextToText'
    architectures = ['GlmOcrForConditionalGeneration']
    template = 'glm_ocr'
    requires = ['transformers>=5.0.1dev0']
    models = ['ZhipuAI/GLM-OCR']  # model_arch inherited from Glm4vLoader (glm4v partitions, no MoE)


@register_model
class GlmEdgeVLoader(ModelLoader):
    model_type = 'glm_edge_v'
    model_cls = 'transformers:AutoModelForCausalLM'
    processor_cls = 'transformers:AutoImageProcessor'  # legacy forced AutoImageProcessor as the processor
    architectures = ['GlmForCausalLM']
    template = 'glm_edge_v'
    requires = ['transformers>=4.46']
    tags = ['vision']
    is_multimodal = True
    # 'glm-edge-4b-chat' is also listed under `glm_edge` (a legacy duplication); id-match resolves it
    # to whichever registers first (glm_edge, the text family) -- ported faithfully.
    models = ['ZhipuAI/glm-edge-v-2b', 'ZhipuAI/glm-edge-4b-chat']

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(language_model='model.layers', vision_tower='model.vision')


@register_model
class CogVLM2VideoLoader(ModelLoader):
    """CogVLM2-Video: remote-code ``CogVLMVideoForCausalLM`` (absent from transformers 5.5), loaded via
    ``AutoModelForCausalLM`` + the ``trust_remote_code`` flag.

    Legacy's ``CogVLM2Loader`` body is *entirely* obsolete on dev: it ran
    ``patch_output_to_input_device`` over every vision MLP / post-attention layernorm and hand-moved the
    ``boi``/``eoi`` parameters onto the projector's device -- all of it working around HF ``device_map``
    sharding, which dev does not use (twinkle owns placement; see PATCH_INVENTORY.md). With those
    dropped this becomes a plain declarative loader.

    Only ``cogvlm2_video`` migrates: its ``transformers>=4.42`` is a floor, whereas the image-only
    ``cogvlm``/``cogvlm2``/``cogagent_*`` siblings pin ``transformers<4.42`` (dead on 5.5).
    """

    model_type = 'cogvlm2_video'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['CogVLMVideoForCausalLM']
    template = 'cogvlm2_video'
    requires = ['decord', 'pytorchvideo', 'transformers>=4.42']
    tags = ['video']
    is_multimodal = True
    models = [('ZhipuAI/cogvlm2-video-llama3-chat', 'zai-org/cogvlm2-video-llama3-chat')]

    @property
    def model_arch(self) -> ModelArch:
        # legacy `ModelArch.cogvlm`: no aligner partition declared.
        return ModelArch(language_model='model.layers', vision_tower='model.vision')


# ------------------------------- ChatGLM (remote-code) -------------------------------


class _ChatGLMLoader(ModelLoader):
    """Shared base for the remote-code ``ChatGLMModel`` checkpoints (glm-4-9b era).

    The one legacy behaviour worth keeping is the ``_pad`` signature fix: these checkpoints ship an
    old tokenizer whose ``_pad`` predates transformers' ``padding_side`` argument, so a caller passing
    ``padding_side=None`` (which transformers itself now does) raises ``TypeError``. The wrapper drops
    that key only when the underlying ``_pad`` cannot accept it. It patches the tokenizer *class*
    because that is where ``_pad`` lives, and is idempotent via the ``_origin_pad`` marker.

    Dropped from legacy:
      * the global ``CrossEntropyLoss.forward`` override that moved ``target`` onto the inputs' device
        -- a cross-device workaround for HF ``device_map`` sharding, which dev does not use (twinkle
        owns placement, so labels and logits already share a device). Patching a *torch* class
        globally for one model family is exactly the kind of blast radius PATCH_INVENTORY rules out.
      * ``quantization_config.llm_int8_skip_modules = ['output_layer']`` -- QLoRA is not wired through
        dev's loader path (build_model never receives a quantization config).
      * the dynamic-class tokenizer surgery (fetch the class from ``auto_map``, set ``_auto_class``,
        ``remove_property``): the g7 ``trust_remote_code`` flag already routes ``AutoTokenizer``
        through the checkpoint's ``auto_map``. Only re-add pieces of it if a specific checkpoint is
        observed to fail.
    """

    model_cls = 'transformers:AutoModelForCausalLM'
    processor_cls = 'transformers:AutoTokenizer'
    trust_remote_code = True
    architectures = ['ChatGLMModel', 'ChatGLMForConditionalGeneration']

    @property
    def model_arch(self) -> ModelArch:
        # legacy `ModelArch.chatglm`: the head is `transformer.output_layer`, not `lm_head`.
        return ModelArch(language_model=['transformer.encoder'], lm_head='transformer.output_layer')

    def process_tokenizer(self, tokenizer):
        import inspect
        tokenizer_cls = type(tokenizer)
        if hasattr(tokenizer_cls, '_origin_pad'):
            return tokenizer
        tokenizer_cls._origin_pad = tokenizer_cls._pad
        parameters = inspect.signature(tokenizer_cls._origin_pad).parameters

        def _pad(self, *args, **kwargs):
            if 'padding_side' in kwargs and kwargs['padding_side'] is None and 'padding_side' not in parameters:
                kwargs.pop('padding_side')
            return tokenizer_cls._origin_pad(self, *args, **kwargs)

        tokenizer_cls._pad = _pad
        return tokenizer


@register_model
class ChatGLM4Loader(_ChatGLMLoader):
    """glm-4-9b (the pre-GLM-4-0414 remote-code line). Legacy's extra step: some of these checkpoints
    list the chat control tokens in ``tokenizer.special_tokens`` but never register them, so
    ``<|user|>`` tokenizes into several pieces -- detected and repaired by adding them.

    Legacy split this family into two groups purely for bookkeeping (glm-4-9b* and LongWriter-glm4-9b);
    both speak the ``chatglm4`` template, so they are one loader here.
    """

    model_type = 'chatglm4'
    template = 'chatglm4'
    requires = ['transformers>=4.42']
    models = [
        ('ZhipuAI/glm-4-9b-chat', 'zai-org/glm-4-9b-chat'),
        ('ZhipuAI/glm-4-9b', 'zai-org/glm-4-9b'),
        ('ZhipuAI/glm-4-9b-chat-1m', 'zai-org/glm-4-9b-chat-1m'),
        ('ZhipuAI/LongWriter-glm4-9b', 'zai-org/LongWriter-glm4-9b'),
    ]

    def process_tokenizer(self, tokenizer):
        tokenizer = super().process_tokenizer(tokenizer)
        if len(tokenizer.encode('<|user|>', add_special_tokens=False)) > 1:
            for key in tokenizer.special_tokens.keys():
                tokenizer.add_tokens(key)
        return tokenizer


@register_model
class ChatGLM4VLoader(_ChatGLMLoader):
    """glm-4v-9b / cogagent-9b: the vision half of the same remote-code ``ChatGLMModel`` class, hence a
    declared ``architectures`` (a *task/modality* difference, not a template variant -- reverse-lookup
    returns chatglm4 and chatglm4v together and the caller disambiguates by id).

    ``build_processor`` keeps the ``image_size=1120`` default legacy wrote into ``init_kwargs``; the
    checkpoint's own preprocessor config omits it and the template relies on it.

    Legacy's ``get_model`` body was purely HF ``device_map`` work (``patch_output_to_input_device`` on
    every vision MLP / layernorm once >=4 GPUs per process, plus hand-moving ``boi``/``eoi`` onto the
    projector's device); obsolete on dev and dropped, same call as ``CogVLM2VideoLoader`` above.

    Only the ``cogagent-9b-20241220`` group's ``transformers>=4.42`` is a plain floor; the
    ``glm-4v-9b`` group additionally pins ``<4.45``, which is dead on 5.5 -- the id stays registered
    (id matching still resolves it) but it will fail the version check, exactly as legacy would.
    """

    model_type = 'chatglm4v'
    template = 'chatglm4v'
    requires = ['transformers>=4.42']
    tags = ['vision']
    is_multimodal = True
    models = [
        ('ZhipuAI/glm-4v-9b', 'zai-org/glm-4v-9b'),
        ('ZhipuAI/cogagent-9b-20241220', 'zai-org/cogagent-9b-20241220'),
    ]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['transformer.encoder'],
            vision_tower=['transformer.vision'],
            lm_head='transformer.output_layer',
        )

    def build_processor(self, model_dir: str, config, **kwargs):
        processor = super().build_processor(model_dir, config, **kwargs)
        processor.init_kwargs['image_size'] = 1120
        return processor
