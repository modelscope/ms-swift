import torch
from megatron.training import get_args

from swift.llm import ModelType, to_device
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
        hf_model.lm_head.weight.data.copy_(mg_language_model.output_layer.weight)
    language_model.norm.weight.data.copy_(mg_language_model.decoder.final_layernorm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_mcore2hf(args, mg_language_model, language_model, layer_idx)
    hf_model.model.visual.load_state_dict(mg_model.visual.visual.state_dict())


class Glm4_5v_Vit(HuggingFaceModule):
    module_mapping = {'model.visual': 'visual'}
    vision_tower = ['visual']
    aligner = ['visual.merger']

    def __init__(self, config):
        from transformers.models.glm4v_moe import Glm4vMoeTextModel
        super().__init__(config, Glm4vMoeTextModel)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        image_grid_thw = kwargs.get('image_grid_thw')
        pixel_values_videos = kwargs.get('pixel_values_videos')
        video_grid_thw = kwargs.get('video_grid_thw')
        model = self._hf_model[0].model
        if pixel_values is not None:
            image_embeds = model.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = model.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = model.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = model.get_placeholder_mask(input_ids, inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        return inputs_embeds


register_megatron_model(
    MMGPTMegatronModelMeta(
        MegatronModelType.glm4_5v, [
            ModelType.glm4_5v,
        ],
        convert_hf2mcore=convert_hf2mcore_glm4_5v,
        convert_mcore2hf=convert_mcore2hf_glm4_5v,
        visual_cls=Glm4_5v_Vit))
