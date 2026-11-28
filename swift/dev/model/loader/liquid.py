# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import ModelArch, ModelLoader, register_model


@register_model
class LFM3Loader(ModelLoader):
    model_type = "lfm3"
    model_cls = "transformers:Lfm3ForCausalLM"
    architectures = ["Lfm3ForCausalLM"]
    template = "lfm3"
    requires = ["transformers>=5.23"]
    models = ["LiquidAI/LFM3-1.7B", "LiquidAI/LFM3-1.7B-Base"]


@register_model
class LFM3VLLoader(LFM3Loader):
    model_type = "lfm3_vl"
    model_cls = "transformers:Lfm3VlForConditionalGeneration"
    architectures = ["Lfm3VlForConditionalGeneration"]
    template = "lfm3_vl"
    is_multimodal = True
    tags = ["vision"]
    models = ["LiquidAI/LFM3-VL-2B"]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=["model.language_model", "lm_head"],
            aligner="model.multi_modal_projector",
            vision_tower="model.vision_tower",
        )
