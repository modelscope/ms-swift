# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from megatron.training import get_args
from PIL import Image

from swift.llm import ModelType, to_device
from ..constant import MegatronModelType
from ..gpt.hf2mcore import set_layer_state as set_layer_state_hf2mcore
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
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        pixel_values_videos = kwargs.get('pixel_values_videos')
        image_grid_thw = kwargs.get('image_grid_thw')
        video_grid_thw = kwargs.get('video_grid_thw')
        visual = self.visual
        processor = self.processor
        dtype = visual.dtype
        config = self.model_config
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            images = [Image.new('RGB', (32, 32), (0, 0, 0))]
            media_inputs = processor.image_processor(images=images, return_tensors='pt')
            media_inputs = to_device(media_inputs, input_ids.device)
            pixel_values = media_inputs['pixel_values'].type(dtype)
            image_embeds = visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
            inputs_embeds = inputs_embeds + image_embeds.mean().to(device=inputs_embeds.device) * 0.
            deepstack_visual_embeds = None
        else:
            if pixel_values is None:
                pixel_values_mixed = pixel_values_videos
                grid_thw = video_grid_thw
            elif pixel_values_videos is None:
                pixel_values_mixed = pixel_values
                grid_thw = image_grid_thw
            else:
                pixel_values_mixed = torch.concat([pixel_values, pixel_values_videos], dim=0)
                grid_thw = torch.concat([image_grid_thw, video_grid_thw], dim=0)
            pixel_values_mixed = pixel_values_mixed.type(dtype)
            mixed_embeds, deepstack_visual_embeds = visual(pixel_values_mixed, grid_thw=grid_thw)
            if pixel_values is None:
                image_embeds = None
                video_embeds = mixed_embeds
            elif pixel_values_videos is None:
                image_embeds = mixed_embeds
                video_embeds = None
            else:
                merge_length = processor.image_processor.merge_size**2
                image_tokens = (image_grid_thw.prod(dim=-1) // merge_length).sum()
                image_embeds = mixed_embeds[:image_tokens]
                video_embeds = mixed_embeds[image_tokens:]

            if image_embeds is not None:
                image_mask = (input_ids == config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask = image_mask.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if video_embeds is not None:
                video_mask = (input_ids == config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                video_mask = video_mask.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        return {'inputs_embeds': inputs_embeds, 'deepstack_visual_embeds': deepstack_visual_embeds}


register_megatron_model(
    MMGPTMegatronModelMeta(
        MegatronModelType.qwen3_vl, [
            ModelType.qwen3_vl,
            ModelType.qwen3_moe_vl,
        ],
        convert_hf2mcore=convert_hf2mcore_qwen3_vl,
        convert_mcore2hf=convert_mcore2hf_qwen3_vl,
        visual_cls=Qwen3VL_Vit))
