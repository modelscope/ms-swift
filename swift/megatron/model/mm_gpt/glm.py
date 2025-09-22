# Copyright (c) Alibaba, Inc. and its affiliates.
from megatron.training import get_args

from swift.llm import ModelType, Template
from ..constant import MegatronModelType
from ..gpt.hf2mcore import set_layer_state as set_layer_state_hf2mcore
from ..gpt.mcore2hf import set_layer_state as set_layer_state_mcore2hf
from ..register import register_megatron_model
from .utils import HuggingFaceModule, MMGPTMegatronModelMeta


def convert_hf2mcore_glm4_5v(hf_model, mg_model):
    language_model = hf_model.model.language_model
    mg_language_model = mg_model.language_model
    args = get_args()
    mg_language_model.embedding.word_embeddings.weight.data.copy_(language_model.embed_tokens.weight)
    if args.untie_embeddings_and_output_weights:
        mg_language_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)
    mg_language_model.decoder.final_layernorm.weight.data.copy_(language_model.norm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_hf2mcore(args, mg_language_model, language_model, layer_idx)
    mg_model.visual.visual.load_state_dict(hf_model.model.visual.state_dict())


def convert_mcore2hf_glm4_5v(hf_model, mg_model):
    language_model = hf_model.model.language_model
    mg_language_model = mg_model.language_model
    args = get_args()
    language_model.embed_tokens.weight.data.copy_(mg_language_model.embedding.word_embeddings.weight)
    if args.untie_embeddings_and_output_weights:
        lm_head_weight = hf_model.score.weight if args.task_type == 'seq_cls' else hf_model.lm_head.weight
        lm_head_weight.data.copy_(mg_language_model.output_layer.weight)
    language_model.norm.weight.data.copy_(mg_language_model.decoder.final_layernorm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_mcore2hf(args, mg_language_model, language_model, layer_idx)
    hf_model.model.visual.load_state_dict(mg_model.visual.visual.state_dict())


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
    MMGPTMegatronModelMeta(
        MegatronModelType.glm4_5v, [
            ModelType.glm4_5v,
        ],
        convert_hf2mcore=convert_hf2mcore_glm4_5v,
        convert_mcore2hf=convert_mcore2hf_glm4_5v,
        visual_cls=Glm4_5vVit))
