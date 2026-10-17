# Copyright (c) ModelScope Contributors. All rights reserved.
# GLM-4-era families: per-Group templates become template variants; GLM-4V patches dropped.
from __future__ import annotations

from .base import ModelArch, ModelLoader, register_model


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
    """CogVLM2-Video: remote-code ``CogVLMVideoForCausalLM`` via ``AutoModelForCausalLM`` +
    ``trust_remote_code``. Legacy's loader body was entirely HF ``device_map`` sharding work
    (patch_output_to_input_device + hand-moved boi/eoi params), dropped per PATCH_INVENTORY. Only
    ``cogvlm2_video`` migrates: its ``transformers>=4.42`` is a floor, whereas the image-only
    siblings pin ``<4.42`` (dead on 5.5)."""

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


# ChatGLM (remote-code): the glm-4-9b era checkpoints.


class _ChatGLMLoader(ModelLoader):
    """Shared base for the remote-code ``ChatGLMModel`` checkpoints (glm-4-9b era).

    Kept: the ``_pad`` signature fix (old tokenizers whose ``_pad`` predates ``padding_side`` raise
    on ``padding_side=None``; the wrapper drops that key when ``_pad`` rejects it, patches the
    tokenizer *class*, idempotent via ``_origin_pad``). Dropped: the CrossEntropyLoss device
    override, llm_int8_skip_modules (QLoRA unwired), dynamic-class tokenizer surgery.
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
    """glm-4-9b (the pre-GLM-4-0414 remote-code line). Some of these checkpoints list the chat
    control tokens in ``tokenizer.special_tokens`` but never register them, so ``<|user|>``
    tokenizes into several pieces -- detected and repaired by adding them. Legacy split this family
    into two bookkeeping groups; both speak the ``chatglm4`` template, so they are one loader."""

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
    """glm-4v-9b / cogagent-9b: the vision half of the same remote-code ``ChatGLMModel`` class, hence
    a declared ``architectures`` (a modality difference, not a template variant; the caller
    disambiguates chatglm4 vs chatglm4v by id). ``build_processor`` keeps the ``image_size=1120``
    default legacy wrote into ``init_kwargs`` (the checkpoint omits it; the template relies on it).
    Legacy's ``get_model`` was pure HF ``device_map`` work, dropped."""

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
