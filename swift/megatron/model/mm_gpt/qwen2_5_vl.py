import torch
from megatron.core.models.huggingface import HuggingFaceModule
from megatron.training import get_args, get_tokenizer
from swift.llm import ModelType, get_model_tokenizer, to_device
from ..constant import MegatronModelType
from ..gpt.hf2mcore import set_layer_state as set_layer_state_hf2mcore
from ..gpt.mcore2hf import set_layer_state as set_layer_state_mcore2hf
from ..register import MegatronModelMeta, register_megatron_model
from .utils import MMGPTMegatronModelMeta, patch_device_map_meta


def convert_hf2mcore_qwen2_5_vl(hf_model, mg_model):
    language_model = hf_model.model.language_model
    mg_language_model = mg_model.language_model
    args = get_args()
    mg_language_model.embedding.word_embeddings.weight.data.copy_(language_model.embed_tokens.weight)
    if args.untie_embeddings_and_output_weights:
        mg_language_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)
    mg_language_model.decoder.final_layernorm.weight.data.copy_(language_model.norm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_hf2mcore(args, mg_language_model, language_model, layer_idx)
    mg_model.visual.model.load_state_dict(hf_model.model.visual.state_dict())


def convert_mcore2hf_qwen2_5_vl(hf_model, mg_model):
    language_model = hf_model.model.language_model
    mg_language_model = mg_model.language_model
    args = get_args()
    language_model.embed_tokens.weight.data.copy_(mg_language_model.embedding.word_embeddings.weight)
    if args.untie_embeddings_and_output_weights:
        hf_model.lm_head.weight.data.copy_(mg_language_model.output_layer.weight)
    language_model.norm.weight.data.copy_(mg_language_model.decoder.final_layernorm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_mcore2hf(args, mg_language_model, language_model, layer_idx)
    hf_model.model.visual.load_state_dict(mg_model.visual.model.state_dict())


class Qwen2_5VL_Vit(HuggingFaceModule):
    vision_tower = ['model']
    aligner = ['model.merger']

    def __init__(self, config):
        from transformers.models.qwen2_5_vl import Qwen2_5_VLTextModel
        super().__init__(config)
        args = get_args()
        model_dir = args.model_info.model_dir
        kwargs = {'attn_impl': 'flash_attn'} if args.attention_backend.name == 'flash' else {}
        with patch_device_map_meta(Qwen2_5_VLTextModel):
            model, _ = get_model_tokenizer(model_dir, args.torch_dtype, return_dummy_model=True, **kwargs)
        self.model = model.visual.to('cuda')
        self.model_config = model.config
        self.processor = get_tokenizer()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        pixel_values_videos = kwargs.get('pixel_values_videos')
        image_grid_thw = kwargs.get('image_grid_thw')
        video_grid_thw = kwargs.get('video_grid_thw')
        dtype = self.model.dtype
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            from PIL import Image
            images = [Image.new('RGB', (32, 32), (0, 0, 0))]
            media_inputs = self.processor.image_processor(images=images, return_tensors='pt')
            device = input_ids.device
            media_inputs = to_device(media_inputs, device)
            pixel_values = media_inputs['pixel_values'].type(dtype)
            image_embeds = self.model(pixel_values, grid_thw=media_inputs['image_grid_thw'])
            inputs_embeds += image_embeds.mean() * 0.
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
            mixed_embeds = self.model(pixel_values_mixed, grid_thw=grid_thw)
            if pixel_values is None:
                image_embeds = None
                video_embeds = mixed_embeds
            elif pixel_values_videos is None:
                image_embeds = mixed_embeds
                video_embeds = None
            else:
                merge_length = self.processor.image_processor.merge_size**2
                image_tokens = (image_grid_thw.prod(dim=-1) // merge_length).sum()
                image_embeds = mixed_embeds[:image_tokens]
                video_embeds = mixed_embeds[image_tokens:]

            if image_embeds is not None:
                image_mask = (input_ids == self.model_config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if video_embeds is not None:
                video_mask = (input_ids == self.model_config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        return inputs_embeds


register_megatron_model(
    MMGPTMegatronModelMeta(
        MegatronModelType.qwen2_5_vl, [
            ModelType.qwen2_5_vl,
        ],
        convert_hf2mcore=convert_hf2mcore_qwen2_5_vl,
        convert_mcore2hf=convert_mcore2hf_qwen2_5_vl,
        visual_cls=Qwen2_5VL_Vit))
