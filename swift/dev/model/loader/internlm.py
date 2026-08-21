# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the InternLM / InternVL families, from ``swift/model/models/internlm.py``.

Migrated: the remote-code text trio ``internlm`` / ``internlm2`` / ``internlm3`` (``InternLM*ForCausalLM``
are not in transformers 5.5, so they load via ``AutoModelForCausalLM`` + ``trust_remote_code=True`` --
the base ``trust_remote_code`` flag), and the transformers-native ``internvl`` (the ``-hf`` checkpoints,
``InternVLForConditionalGeneration`` via ``AutoModelForImageTextToText``). Legacy wrapped internvl in a
``patched_enable_input_require_grads`` hook needed only under reentrant gradient checkpointing;
PATCH_INVENTORY.md marks that obsolete when ``use_reentrant=False`` (dev's path), so it is dropped.
The text trio are pure LLMs, so they keep the empty ``ModelArch()`` default (legacy's
``ModelArch.llama`` / ``internlm2`` only mattered for multimodal partitioning).

Not migrated here (see MODEL_MIGRATION.md):
  * ``interns1`` -- ``InternS1ForConditionalGeneration`` absent from transformers 5.5 and pinned
    ``<4.56``.
  * ``xcomposer2`` / ``xcomposer2_4khd`` / ``xcomposer2_5`` -- remote-code + dynamic CLIP vision tower
    (bucket C); ``xcomposer2_5_ol_audio`` -- the Qwen2-Audio half of xcomposer, part of that family.
  * the two phi3-based ``internvl_chat`` groups (``Mini-InternVL-Chat-4B-V1-5`` /
    ``InternVL2-4B``) -- pinned ``transformers<4.42``, dead on 5.5.
"""
from __future__ import annotations

from .base import ModelArch, ModelLoader, register_model


@register_model
class InternLMLoader(ModelLoader):
    model_type = 'internlm'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True  # InternLMForCausalLM ships in the checkpoint's modeling_*.py
    architectures = ['InternLMForCausalLM']
    template = 'internlm'
    models = [
        ('Shanghai_AI_Laboratory/internlm-chat-7b', 'internlm/internlm-chat-7b'),
        ('Shanghai_AI_Laboratory/internlm-7b', 'internlm/internlm-7b'),
        'Shanghai_AI_Laboratory/internlm-chat-7b-8k',
        ('Shanghai_AI_Laboratory/internlm-20b', 'internlm/internlm-20b'),
        ('Shanghai_AI_Laboratory/internlm-chat-20b', 'internlm/internlm-chat-20b'),
    ]


@register_model
class InternLM2Loader(ModelLoader):
    model_type = 'internlm2'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['InternLM2ForCausalLM']
    template = 'internlm2'
    requires = ['transformers>=4.38']
    models = [
        ('Shanghai_AI_Laboratory/internlm2-chat-1_8b', 'internlm/internlm2-chat-1_8b'),
        ('Shanghai_AI_Laboratory/internlm2-1_8b', 'internlm/internlm2-1_8b'),
        ('Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft', 'internlm/internlm2-chat-1_8b-sft'),
        ('Shanghai_AI_Laboratory/internlm2-base-7b', 'internlm/internlm2-base-7b'),
        ('Shanghai_AI_Laboratory/internlm2-7b', 'internlm/internlm2-7b'),
        ('Shanghai_AI_Laboratory/internlm2-chat-7b', 'internlm/internlm2-chat-7b'),
        ('Shanghai_AI_Laboratory/internlm2-chat-7b-sft', 'internlm/internlm2-chat-7b-sft'),
        ('Shanghai_AI_Laboratory/internlm2-base-20b', 'internlm/internlm2-base-20b'),
        ('Shanghai_AI_Laboratory/internlm2-20b', 'internlm/internlm2-20b'),
        ('Shanghai_AI_Laboratory/internlm2-chat-20b', 'internlm/internlm2-chat-20b'),
        ('Shanghai_AI_Laboratory/internlm2-chat-20b-sft', 'internlm/internlm2-chat-20b-sft'),
        # math
        ('Shanghai_AI_Laboratory/internlm2-math-7b', 'internlm/internlm2-math-7b'),
        ('Shanghai_AI_Laboratory/internlm2-math-base-7b', 'internlm/internlm2-math-base-7b'),
        ('Shanghai_AI_Laboratory/internlm2-math-base-20b', 'internlm/internlm2-math-base-20b'),
        ('Shanghai_AI_Laboratory/internlm2-math-20b', 'internlm/internlm2-math-20b'),
        # 2.5
        ('Shanghai_AI_Laboratory/internlm2_5-1_8b-chat', 'internlm/internlm2_5-1_8b-chat'),
        ('Shanghai_AI_Laboratory/internlm2_5-1_8b', 'internlm/internlm2_5-1_8b'),
        ('Shanghai_AI_Laboratory/internlm2_5-7b', 'internlm/internlm2_5-7b'),
        ('Shanghai_AI_Laboratory/internlm2_5-7b-chat', 'internlm/internlm2_5-7b-chat'),
        ('Shanghai_AI_Laboratory/internlm2_5-7b-chat-1m', 'internlm/internlm2_5-7b-chat-1m'),
        ('Shanghai_AI_Laboratory/internlm2_5-20b', 'internlm/internlm2_5-20b'),
        ('Shanghai_AI_Laboratory/internlm2_5-20b-chat', 'internlm/internlm2_5-20b-chat'),
    ]


@register_model
class InternLM3Loader(ModelLoader):
    model_type = 'internlm3'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['InternLM3ForCausalLM']
    template = 'internlm2'
    requires = ['transformers>=4.48']
    models = [('Shanghai_AI_Laboratory/internlm3-8b-instruct', 'internlm/internlm3-8b-instruct')]


@register_model
class InternVLLoader(ModelLoader):
    model_type = 'internvl'
    model_cls = 'transformers:AutoModelForImageTextToText'
    architectures = ['InternVLForConditionalGeneration']
    template = 'internvl_hf'
    # Highest floor across the merged groups (the GPT-OSS-hf checkpoint needs >=4.55.0); dev's
    # per-model_type `requires` takes the strictest, per the per-group-requires convention.
    requires = ['transformers>=4.55.0', 'timm']
    tags = ['vision', 'video']
    is_multimodal = True
    models = [
        'OpenGVLab/InternVL3-1B-hf',
        'OpenGVLab/InternVL3-2B-hf',
        'OpenGVLab/InternVL3-8B-hf',
        'OpenGVLab/InternVL3-9B-hf',
        'OpenGVLab/InternVL3-14B-hf',
        'OpenGVLab/InternVL3-38B-hf',
        'OpenGVLab/InternVL3-78B-hf',
        'OpenGVLab/InternVL3_5-1B-HF',
        'OpenGVLab/InternVL3_5-2B-HF',
        'OpenGVLab/InternVL3_5-4B-HF',
        'OpenGVLab/InternVL3_5-8B-HF',
        'OpenGVLab/InternVL3_5-14B-HF',
        'OpenGVLab/InternVL3_5-38B-HF',
        'OpenGVLab/InternVL3_5-30B-A3B-HF',
        'OpenGVLab/InternVL3_5-241B-A28B-HF',
        'OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF',
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
class InternLM2RewardLoader(ModelLoader):
    """InternLM2 reward model. Remote-code ``InternLM2ForRewardModel`` (absent from transformers 5.5),
    which legacy's ``RewardModelLoader`` loads via ``AutoModel`` (the checkpoint's ``auto_map`` points
    ``AutoModel`` at the reward class) -- so ``model_cls='transformers:AutoModel'`` + the g7
    ``trust_remote_code`` flag. ``is_reward`` records that this is a reward head (num_labels=1 seq_cls
    at build time); like other dev metadata it is declared now and consumed when the reward path is
    wired."""

    model_type = 'internlm2_reward'
    model_cls = 'transformers:AutoModel'
    trust_remote_code = True
    architectures = ['InternLM2ForRewardModel']
    template = 'internlm2_reward'
    is_reward = True
    requires = ['transformers>=4.38']
    models = [
        ('Shanghai_AI_Laboratory/internlm2-1_8b-reward', 'internlm/internlm2-1_8b-reward'),
        ('Shanghai_AI_Laboratory/internlm2-7b-reward', 'internlm/internlm2-7b-reward'),
        ('Shanghai_AI_Laboratory/internlm2-20b-reward', 'internlm/internlm2-20b-reward'),
    ]


# ---------------------------- InternVL-Chat (remote-code, non -hf) ----------------------------
# Legacy filed all of InternVL1.5 / 2 / 2.5 / 3 / 3.5 under the single `internvl_chat` model_type with
# eight template groups. A dev model_type carries exactly one template, so the base keeps the legacy
# name (the `internvl` template) and the other formats are family-qualified subclasses with
# `architectures=[]`, leaving `InternVLChatModel` reverse-lookup on the base alone. Distinct from the
# `internvl` loader above, which owns the transformers-native ``-hf`` checkpoints.


@register_model
class InternVLChatLoader(ModelLoader):
    """InternVL-Chat: remote-code ``InternVLChatModel`` -- an outer wrapper whose real LLM hangs off
    ``model.language_model``, so ``process_model`` proxies the top-level calls there
    (legacy ``use_submodel_func(model, 'language_model')``). ``processor_cls`` is pinned to
    ``AutoTokenizer`` because legacy forced it: these checkpoints ship a ``preprocessor_config.json``
    that would otherwise send the base's file detection to ``AutoProcessor``, while the template does
    its own image preprocessing and needs the plain tokenizer.

    Dropped from legacy, per PATCH_INVENTORY: ``patch_output_clone`` on the input embeddings (a
    reentrant-checkpointing workaround) and the bnb ``force_no_igemmlt`` branch (QLoRA is not wired
    through dev's loader path).
    """

    model_type = 'internvl_chat'
    model_cls = 'transformers:AutoModel'
    processor_cls = 'transformers:AutoTokenizer'
    trust_remote_code = True
    architectures = ['InternVLChatModel']
    template = 'internvl'
    requires = ['transformers>=4.35', 'timm']
    tags = ['vision']
    is_multimodal = True
    models = [
        'OpenGVLab/Mini-InternVL-Chat-2B-V1-5',
        ('AI-ModelScope/InternVL-Chat-V1-5', 'OpenGVLab/InternVL-Chat-V1-5'),
        ('AI-ModelScope/InternVL-Chat-V1-5-int8', 'OpenGVLab/InternVL-Chat-V1-5-int8'),
    ]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(language_model=['language_model'], aligner=['mlp1'], vision_tower=['vision_model'])

    def process_model(self, model):
        self.delegate_to_submodel(model, 'language_model')
        return model


@register_model
class InternVLChat2Loader(InternVLChatLoader):
    model_type = 'internvl_chat_v2'
    architectures = []  # template variant; `InternVLChatModel` stays owned by `internvl_chat`
    template = 'internvl2'
    requires = ['transformers>=4.36', 'timm']
    tags = ['vision', 'video']
    models = [
        'OpenGVLab/InternVL2-1B',
        'OpenGVLab/InternVL2-2B',
        'OpenGVLab/InternVL2-8B',
        'OpenGVLab/InternVL2-26B',
        'OpenGVLab/InternVL2-40B',
        'OpenGVLab/InternVL2-Llama3-76B',
        'OpenGVLab/InternVL2-2B-AWQ',
        'OpenGVLab/InternVL2-8B-AWQ',
        'OpenGVLab/InternVL2-26B-AWQ',
        'OpenGVLab/InternVL2-40B-AWQ',
        'OpenGVLab/InternVL2-Llama3-76B-AWQ',
        'OpenGVLab/InternVL2-8B-MPO',
        'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-1B-Pretrain',
        'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-2B-Pretrain',
        'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-4B-Pretrain',
        'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-8B-Pretrain',
        'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-26B-Pretrain',
        'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-40B-Pretrain',
        'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-Llama3-76B-Pretrain',
    ]


@register_model
class InternVLChat2_5Loader(InternVLChatLoader):
    """InternVL2.5 *and* InternVL3: legacy filed them as two separate groups that share the
    ``internvl2_5`` template and differ only in floor (>=4.36 vs >=4.37.2), so they merge here under
    the stricter floor -- the same per-group-requires convention used elsewhere."""

    model_type = 'internvl_chat_v2_5'
    architectures = []
    template = 'internvl2_5'
    requires = ['transformers>=4.37.2', 'timm']
    tags = ['vision', 'video']
    models = [
        'OpenGVLab/InternVL2_5-1B',
        'OpenGVLab/InternVL2_5-2B',
        'OpenGVLab/InternVL2_5-4B',
        'OpenGVLab/InternVL2_5-8B',
        'OpenGVLab/InternVL2_5-26B',
        'OpenGVLab/InternVL2_5-38B',
        'OpenGVLab/InternVL2_5-78B',
        'OpenGVLab/InternVL2_5-4B-AWQ',
        'OpenGVLab/InternVL2_5-8B-AWQ',
        'OpenGVLab/InternVL2_5-26B-AWQ',
        'OpenGVLab/InternVL2_5-38B-AWQ',
        'OpenGVLab/InternVL2_5-78B-AWQ',
        'OpenGVLab/InternVL2_5-1B-MPO',
        'OpenGVLab/InternVL2_5-2B-MPO',
        'OpenGVLab/InternVL2_5-4B-MPO',
        'OpenGVLab/InternVL2_5-8B-MPO',
        'OpenGVLab/InternVL2_5-26B-MPO',
        'OpenGVLab/InternVL2_5-38B-MPO',
        'OpenGVLab/InternVL2_5-78B-MPO',
        # InternVL3 (same template, stricter floor)
        'OpenGVLab/InternVL3-1B-Pretrained',
        'OpenGVLab/InternVL3-2B-Pretrained',
        'OpenGVLab/InternVL3-8B-Pretrained',
        'OpenGVLab/InternVL3-9B-Pretrained',
        'OpenGVLab/InternVL3-14B-Pretrained',
        'OpenGVLab/InternVL3-38B-Pretrained',
        'OpenGVLab/InternVL3-78B-Pretrained',
        'OpenGVLab/InternVL3-1B-Instruct',
        'OpenGVLab/InternVL3-2B-Instruct',
        'OpenGVLab/InternVL3-8B-Instruct',
        'OpenGVLab/InternVL3-9B-Instruct',
        'OpenGVLab/InternVL3-14B-Instruct',
        'OpenGVLab/InternVL3-38B-Instruct',
        'OpenGVLab/InternVL3-78B-Instruct',
        'OpenGVLab/InternVL3-1B',
        'OpenGVLab/InternVL3-2B',
        'OpenGVLab/InternVL3-8B',
        'OpenGVLab/InternVL3-9B',
        'OpenGVLab/InternVL3-14B',
        'OpenGVLab/InternVL3-38B',
        'OpenGVLab/InternVL3-78B',
        'OpenGVLab/InternVL3-1B-AWQ',
        'OpenGVLab/InternVL3-2B-AWQ',
        'OpenGVLab/InternVL3-8B-AWQ',
        'OpenGVLab/InternVL3-9B-AWQ',
        'OpenGVLab/InternVL3-14B-AWQ',
        'OpenGVLab/InternVL3-38B-AWQ',
        'OpenGVLab/InternVL3-78B-AWQ',
        ('SenseNova/SenseNova-SI-InternVL3-2B', 'sensenova/SenseNova-SI-InternVL3-2B'),
        ('SenseNova/SenseNova-SI-InternVL3-8B', 'sensenova/SenseNova-SI-InternVL3-8B'),
        ('SenseNova/SenseNova-SI-1.1-InternVL3-2B', 'sensenova/SenseNova-SI-1.1-InternVL3-2B'),
        ('SenseNova/SenseNova-SI-1.1-InternVL3-8B', 'sensenova/SenseNova-SI-1.1-InternVL3-8B'),
    ]


@register_model
class InternVLChat3_5Loader(InternVLChatLoader):
    model_type = 'internvl_chat_v3_5'
    architectures = []
    template = 'internvl3_5'
    requires = ['transformers>=4.37.2', 'timm']
    tags = ['vision', 'video']
    models = [
        'OpenGVLab/InternVL3_5-1B-Pretrained',
        'OpenGVLab/InternVL3_5-2B-Pretrained',
        'OpenGVLab/InternVL3_5-4B-Pretrained',
        'OpenGVLab/InternVL3_5-8B-Pretrained',
        'OpenGVLab/InternVL3_5-14B-Pretrained',
        'OpenGVLab/InternVL3_5-38B-Pretrained',
        'OpenGVLab/InternVL3_5-30B-A3B-Pretrained',
        'OpenGVLab/InternVL3_5-241B-A28B-Pretrained',
        'OpenGVLab/InternVL3_5-1B-Instruct',
        'OpenGVLab/InternVL3_5-2B-Instruct',
        'OpenGVLab/InternVL3_5-4B-Instruct',
        'OpenGVLab/InternVL3_5-8B-Instruct',
        'OpenGVLab/InternVL3_5-14B-Instruct',
        'OpenGVLab/InternVL3_5-38B-Instruct',
        'OpenGVLab/InternVL3_5-30B-A3B-Instruct',
        'OpenGVLab/InternVL3_5-241B-A28B-Instruct',
        'OpenGVLab/InternVL3_5-1B-MPO',
        'OpenGVLab/InternVL3_5-2B-MPO',
        'OpenGVLab/InternVL3_5-4B-MPO',
        'OpenGVLab/InternVL3_5-8B-MPO',
        'OpenGVLab/InternVL3_5-14B-MPO',
        'OpenGVLab/InternVL3_5-38B-MPO',
        'OpenGVLab/InternVL3_5-30B-A3B-MPO',
        'OpenGVLab/InternVL3_5-241B-A28B-MPO',
        'OpenGVLab/InternVL3_5-1B',
        'OpenGVLab/InternVL3_5-2B',
        'OpenGVLab/InternVL3_5-4B',
        'OpenGVLab/InternVL3_5-8B',
        'OpenGVLab/InternVL3_5-14B',
        'OpenGVLab/InternVL3_5-38B',
        'OpenGVLab/InternVL3_5-30B-A3B',
        'OpenGVLab/InternVL3_5-241B-A28B',
    ]


@register_model
class InternVLChat3_5GptLoader(InternVLChatLoader):
    """The GPT-OSS-backed InternVL3.5 preview, which speaks its own chat format."""

    model_type = 'internvl_chat_v3_5_gpt'
    architectures = []
    template = 'internvl3_5_gpt'
    requires = ['transformers>=4.37.2', 'timm']
    tags = ['vision', 'video']
    models = ['OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview']
