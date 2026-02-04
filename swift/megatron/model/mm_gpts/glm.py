# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.model import ModelType
from swift.template import Template
from ..constant import MegatronModelType
from ..gpt_bridge import MultimodalGPTBridge
from ..gpts.glm4 import Glm4Bridge, Glm4Loader
from ..register import MegatronModelLoader, MegatronModelMeta, register_megatron_model
from .utils import HuggingFaceModule


class Glm4vVit(HuggingFaceModule):
    module_mapping = {'model.visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger']

    def __init__(self, config):
        from transformers.models.glm4v_moe import Glm4vMoeTextModel
        super().__init__(config, Glm4vMoeTextModel)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return Template._get_inputs_embeds_hf(inputs_embeds, kwargs, self.visual, self.processor, self.model_config)


class Glm4vMoeLoader(MegatronModelLoader):
    bridge_cls = MultimodalGPTBridge
    visual_cls = Glm4vVit


register_megatron_model(MegatronModelMeta(
    MegatronModelType.glm4v_moe,
    [
        ModelType.glm4v_moe,
    ],
    Glm4vMoeLoader,
))


class Glm4vBridge(Glm4Bridge, MultimodalGPTBridge):
    pass


class Glm4vLoader(Glm4Loader):
    bridge_cls = Glm4vBridge
    visual_cls = Glm4vVit


register_megatron_model(MegatronModelMeta(
    MegatronModelType.glm4v,
    [
        ModelType.glm4v,
    ],
    Glm4vLoader,
))
