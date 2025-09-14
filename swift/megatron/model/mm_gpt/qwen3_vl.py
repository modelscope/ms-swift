# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from megatron.training import get_args

from swift.llm import ModelType
from ..constant import MegatronModelType
from ..gpt.hf2mcore import convert_hf2mcore
from ..gpt.hf2mcore import set_layer_state as set_layer_state_hf2mcore
from ..gpt.mcore2hf import convert_mcore2hf
from ..gpt.mcore2hf import set_layer_state as set_layer_state_mcore2hf
from ..register import register_megatron_model
from .utils import HuggingFaceModule, MMGPTMegatronModelMeta


def convert_hf2mcore_qwen3_vl(hf_model, mg_model):
    language_model = hf_model.model
    if hasattr(language_model, 'language_model'):
        language_model = language_model.language_model
    visual = hf_model.visual if hasattr(hf_model, 'visual') else hf_model.model.visual
    mg_language_model = mg_model.language_model
    args = get_args()
    mg_language_model.embedding.word_embeddings.weight.data.copy_(language_model.embed_tokens.weight)
    if args.untie_embeddings_and_output_weights:
        mg_language_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)
    mg_language_model.decoder.final_layernorm.weight.data.copy_(language_model.norm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_hf2mcore(args, mg_language_model, language_model, layer_idx)
    mg_model.visual.visual.load_state_dict(visual.state_dict())


def convert_mcore2hf_qwen3_vl(hf_model, mg_model):
    language_model = hf_model.model
    if hasattr(language_model, 'language_model'):
        language_model = language_model.language_model
    visual = hf_model.visual if hasattr(hf_model, 'visual') else hf_model.model.visual
    mg_language_model = mg_model.language_model
    args = get_args()
    language_model.embed_tokens.weight.data.copy_(mg_language_model.embedding.word_embeddings.weight)
    if args.untie_embeddings_and_output_weights:
        hf_model.lm_head.weight.data.copy_(mg_language_model.output_layer.weight)
    language_model.norm.weight.data.copy_(mg_language_model.decoder.final_layernorm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_mcore2hf(args, mg_language_model, language_model, layer_idx)
    visual.load_state_dict(mg_model.visual.visual.state_dict())


class Qwen3VL_Vit(HuggingFaceModule):
    module_mapping = {'visual': 'visual'}
    vision_tower = ['visual']
    aligner = ['visual.merger', 'visual.deepstack_merger_list']

    def __init__(self, config):
        from transformers.models.qwen3_vl import Qwen3VLTextModel
        from transformers.models.qwen3_vl_moe import Qwen3VLMoeTextModel
        super().__init__(config, [Qwen3VLTextModel, Qwen3VLMoeTextModel])

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        model = self._hf_model[0]
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        if pixel_values is None:
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), dtype=self.vision_model.dtype, device=inputs_embeds.device)
            vit_embeds = model.extract_feature(dummy_pixel_values)
            inputs_embeds = inputs_embeds + vit_embeds.mean() * 0.
        else:
            vit_embeds = model.extract_feature(pixel_values)
            selected = (input_ids == self.processor.encode('<IMG_CONTEXT>', add_special_tokens=False)[0])
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[selected] = vit_embeds.reshape(-1, vit_embeds.shape[-1]).to(dtype=inputs_embeds.dtype)
        return inputs_embeds


register_megatron_model(
    MMGPTMegatronModelMeta(
        MegatronModelType.qwen3_vl, [
            ModelType.qwen3_vl,
            ModelType.qwen3_moe_vl,
        ],
        convert_hf2mcore=convert_hf2mcore_qwen3_vl,
        convert_mcore2hf=convert_mcore2hf_qwen3_vl,
        visual_cls=Qwen3VL_Vit))
