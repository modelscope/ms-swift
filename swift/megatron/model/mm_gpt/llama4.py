# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import ModelType, Template
from ..constant import MegatronModelType
from ..gpt_bridge import MultimodalGPTBridge
from ..register import MegatronModelMeta, register_megatron_model
from .utils import HuggingFaceModule


class Llama4Vit(HuggingFaceModule):
    module_mapping = {'multi_modal_projector': 'multi_modal_projector', 'vision_model': 'vision_model'}
    _vision_tower = ['vision_model']
    _aligner = ['multi_modal_projector']

    def __init__(self, config):
        from transformers.models.llama4 import Llama4TextModel
        super().__init__(config, Llama4TextModel)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return Template._get_inputs_embeds_hf(inputs_embeds, kwargs, self.visual, self.processor, self.model_config)


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.llama4, [
            ModelType.llama4,
        ], bridge_cls=MultimodalGPTBridge, visual_cls=Llama4Vit))
