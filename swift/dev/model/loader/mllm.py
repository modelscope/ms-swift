# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for assorted transformers-native multimodal families (from ``swift/model/models/mllm.py``).

Only the checkpoints that load through a stock transformers Auto/``*ForConditionalGeneration`` class
with nothing but a ``model_arch`` partition are migrated here: ``idefics3`` and ``pixtral``. The
``qwen2_gme`` embedding variant is a pure Qwen2-VL template variant and already lives in ``qwen.py``.

Not migrated here (see MODEL_MIGRATION.md):
  * bucket B: ``keye_vl`` / ``keye_vl_1_5`` -- the processor must patch ``keye_vl_utils.vision_process``
    (a Qwen-VL-style env seam); ``molmoe`` -- forces float32, monkeypatches ``config.to_dict`` and
    clones the embedding output.
  * bucket C: ``molmo`` / ``molmo2`` / ``megrez_omni`` / ``dots_ocr`` (dynamic-module class +
    ``_no_split_modules`` + output-clone/submodel patches), ``jina_reranker_m0`` (``AutoModel`` with a
    wholesale ``forward`` rewrite into a reranker head), ``sail_vl2`` (``use_submodel_func``).
"""
from __future__ import annotations

from .base import ModelArch, ModelLoader, register_model


@register_model
class Idefics3Loader(ModelLoader):
    model_type = 'idefics3'
    model_cls = 'transformers:AutoModelForVision2Seq'
    architectures = ['Idefics3ForConditionalGeneration']
    template = 'idefics3'
    requires = ['transformers>=4.45']
    tags = ['vision']
    is_multimodal = True
    models = [('AI-ModelScope/Idefics3-8B-Llama3', 'HuggingFaceM4/Idefics3-8B-Llama3')]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model='model.text_model',
            aligner='model.connector',
            vision_tower='model.vision_model',
        )


@register_model
class PixtralLoader(ModelLoader):
    model_type = 'pixtral'
    model_cls = 'transformers:LlavaForConditionalGeneration'
    architectures = ['LlavaForConditionalGeneration']
    template = 'pixtral'
    requires = ['transformers>=4.45']
    tags = ['vision']
    is_multimodal = True
    models = [('AI-ModelScope/pixtral-12b', 'mistral-community/pixtral-12b')]

    @property
    def model_arch(self) -> ModelArch:
        # legacy `ModelArch.llava_hf`, transformers>=4.52 (model.* prefix) branch.
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner='model.multi_modal_projector',
            vision_tower='model.vision_tower',
        )
