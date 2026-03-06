# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.model import ModelType
from swift.template import Template
from ..constant import MegatronModelType
from ..gpts.qwen3_next import Qwen3NextBridge, Qwen3NextLoader
from ..register import MegatronModelMeta, register_megatron_model
from .utils import HuggingFaceModule


class Qwen3_5Vit(HuggingFaceModule):
    module_mapping = {'model.visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger']

    def __init__(self, config):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel
        super().__init__(config, [Qwen3_5TextModel, Qwen3_5MoeTextModel])

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return Template._get_inputs_embeds_hf(inputs_embeds, kwargs, self.visual, self.processor, self.hf_config)


class Qwen3_5Bridge(Qwen3NextBridge):
    hf_layers_prefix = 'model.language_model.layers'
    hf_embed_key = 'model.language_model.embed_tokens.weight'
    hf_final_layernorm_key = 'model.language_model.norm.weight'


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.qwen3_5,
        [
            ModelType.qwen3_5,
            ModelType.qwen3_5_moe,
        ],
        bridge_cls=Qwen3_5Bridge,
        visual_cls=Qwen3_5Vit,
        loader=Qwen3NextLoader,
    ))
