# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import ModelType, Template
from ..constant import MegatronModelType
from ..gpt_bridge import MultimodalGPTBridge
from ..register import MegatronModelMeta, register_megatron_model
from .utils import HuggingFaceModule


class Glm4_5vVit(HuggingFaceModule):
    module_mapping = {'model.visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger']

    def __init__(self, config):
        from transformers.models.glm4v_moe import Glm4vMoeTextModel
        super().__init__(config, Glm4vMoeTextModel)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return Template._get_inputs_embeds_hf(inputs_embeds, kwargs, self.visual, self.processor, self.model_config)


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.glm4_5v, [
            ModelType.glm4_5v,
        ], bridge_cls=MultimodalGPTBridge, visual_cls=Glm4_5vVit))
