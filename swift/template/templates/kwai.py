# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

import numpy as np
import torch

from swift.utils import is_deepspeed_enabled, to_device
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Word, findall
from .utils import ChatmlTemplateMeta


@dataclass
class KeyeTemplateMeta(ChatmlTemplateMeta):
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])


class KeyeVLTemplate(Template):
    image_token_id = 151655
    video_token_id = 151656
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from keye_vl_utils import fetch_image, fetch_video
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            if getattr(self, 'mode', None) == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            video = inputs.videos[index]
            video, video_kwargs = fetch_video({'video': video})
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            for k, v in video_kwargs.items():
                inputs.mm_processor_kwargs.setdefault(k, []).append(v)
            return ['<|vision_start|><|video_pad|><|vision_end|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        for media_type in ['images', 'videos']:
            mm_data = getattr(inputs, media_type)
            if mm_data:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(images=mm_data, return_tensors='pt')
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    split_token = self._tokenize('\n')[0]
                    media_inputs = processor(
                        text=['\n'.join(['<|video_pad|>'] * len(mm_data))],
                        videos=mm_data,
                        return_tensors='pt',
                        **inputs.mm_processor_kwargs)
                    splited_tokens = self._split_list(media_inputs['input_ids'][0].tolist(), split_token)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    if media_type == 'images':
                        token_len = (media_grid_thw[i].prod() // merge_length)
                        return [media_token] * token_len
                    else:
                        return splited_tokens[i]

                input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                    _get_new_tokens)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    @staticmethod
    def _grid_thw_to_list(grid_thw):
        grid_hws = []
        for thw in grid_thw:
            if isinstance(thw, torch.Tensor):
                thw_tuple = tuple(thw.detach().cpu().numpy().tolist())
            else:
                thw_tuple = tuple(thw)
            grid_hws.append(thw_tuple)
        return grid_hws

    @staticmethod
    def _build_siglip_inputs(grid_hws, device):
        position_ids = []
        sample_indices = []
        cu_seqlens = [0]
        for idx, thw_tuple in enumerate(grid_hws):
            numel = int(np.prod(thw_tuple))
            media_position_ids = torch.arange(numel, device=device) % int(np.prod(thw_tuple[1:]))
            position_ids.append(media_position_ids)
            sample_indices.append(torch.full((numel, ), idx, dtype=torch.int64, device=device))
            cu_seqlens.append(cu_seqlens[-1] + numel)
        return (
            torch.concat(position_ids, dim=0),
            torch.concat(sample_indices, dim=0),
            torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
        )

    def _encode_visual_embeds(self, model, pixel_values, grid_thw, dtype):
        pixel_values = pixel_values.type(dtype).unsqueeze(0)
        if grid_thw is not None:
            grid_hws = self._grid_thw_to_list(grid_thw)
            siglip_position_ids, sample_indices, cu_seqlens = self._build_siglip_inputs(
                grid_hws, pixel_values.device)
            vision_outputs = model.visual(
                pixel_values=pixel_values,
                image_grid_thw=grid_hws,
                position_ids=siglip_position_ids,
                vision_return_embed_list=True,
                interpolate_pos_encoding=True,
                sample_indices=sample_indices,
                cu_seqlens=cu_seqlens,
                return_pooler_output=False,
                use_rope=True,
                window_size=-1,
            )
            media_embeds = vision_outputs.last_hidden_state
            media_embeds = model.mlp_AR(media_embeds, grid_thw)
            media_embeds = torch.cat(media_embeds, dim=0)
        else:
            num_patches = pixel_values.shape[1]
            position_ids = torch.arange(num_patches, device=pixel_values.device)
            vision_outputs = model.visual(pixel_values=pixel_values, position_ids=position_ids)
            media_embeds = vision_outputs.last_hidden_state.reshape(-1, vision_outputs.last_hidden_state.shape[-1])
        return media_embeds

    @staticmethod
    def _scatter_visual_embeds(inputs_embeds, input_ids, token_id, visual_embeds):
        visual_mask = (input_ids == token_id).unsqueeze(-1).expand_as(inputs_embeds)
        visual_embeds = visual_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(visual_mask, visual_embeds)

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        pixel_values = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        fast_pixel_values_videos = inputs.get('fast_pixel_values_videos')
        image_grid_thw = inputs.get('image_grid_thw')
        video_grid_thw = inputs.get('video_grid_thw')
        fast_video_grid_thw = inputs.get('fast_video_grid_thw')

        base_model = self.get_base_model(model)
        if hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)

        # Get dtype from visual model, adapting for KeyeVL model structure
        if hasattr(model.visual, 'get_dtype'):
            dtype = model.visual.get_dtype()
        else:
            dtype = model.visual.dtype

        if pixel_values is None and pixel_values_videos is None:  # plain-text
            if is_deepspeed_enabled():
                from PIL import Image
                images = [Image.new('RGB', (32, 32), (0, 0, 0))]
                media_inputs = self.processor.image_processor(images=images, return_tensors='pt')
                device = input_ids.device
                media_inputs = to_device(media_inputs, device)
                pixel_values = media_inputs['pixel_values'].type(dtype)
                pixel_values = pixel_values.unsqueeze(0)

                num_patches = pixel_values.shape[1]
                position_ids = torch.arange(num_patches, device=device)

                merge_size = getattr(self.processor.image_processor, 'merge_size', 2)
                grid_size = int(np.sqrt(num_patches))

                if grid_size % merge_size != 0:
                    grid_size = ((grid_size + merge_size - 1) // merge_size) * merge_size

                dummy_grid_hw = [(1, grid_size, grid_size)]
                sample_indices = torch.zeros(num_patches, dtype=torch.int64, device=device)
                cu_seqlens = torch.tensor([0, num_patches], dtype=torch.int32, device=device)

                vision_outputs = model.visual(
                    pixel_values=pixel_values,
                    image_grid_thw=dummy_grid_hw,
                    position_ids=position_ids,
                    vision_return_embed_list=True,
                    interpolate_pos_encoding=True,
                    sample_indices=sample_indices,
                    cu_seqlens=cu_seqlens,
                    return_pooler_output=False,
                    use_rope=True,
                    window_size=-1,
                )
                image_embeds = vision_outputs.last_hidden_state
                image_embeds = model.mlp_AR(image_embeds, dummy_grid_hw)
                image_embeds = torch.cat(image_embeds, dim=0)
                inputs_embeds += image_embeds.mean() * 0.
        else:
            if pixel_values is not None:
                image_embeds = self._encode_visual_embeds(model, pixel_values, image_grid_thw, dtype)
                inputs_embeds = self._scatter_visual_embeds(inputs_embeds, input_ids, model.config.image_token_id,
                                                            image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self._encode_visual_embeds(model, pixel_values_videos, video_grid_thw, dtype)
                inputs_embeds = self._scatter_visual_embeds(inputs_embeds, input_ids, model.config.video_token_id,
                                                            video_embeds)

            if fast_pixel_values_videos is not None:
                fast_video_embeds = self._encode_visual_embeds(model, fast_pixel_values_videos, fast_video_grid_thw,
                                                               dtype)
                inputs_embeds = self._scatter_visual_embeds(inputs_embeds, input_ids,
                                                            model.config.fast_video_token_id, fast_video_embeds)

        position_ids, _ = model.get_rope_index_slowfast(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            fast_video_grid_thw=fast_video_grid_thw,
            attention_mask=attention_mask,
        )
        return {'inputs_embeds': inputs_embeds, 'position_ids': position_ids}

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        fast_pixel_values_videos = [b['fast_pixel_values_videos'] for b in batch if b.get('fast_pixel_values_videos') is not None]
        if len(fast_pixel_values_videos) > 0:
            res['fast_pixel_values_videos'] = torch.concat(fast_pixel_values_videos)

        fast_video_grid_thw = self.concat_tensor(batch, 'fast_video_grid_thw', 0)
        if fast_video_grid_thw is not None:
            res['fast_video_grid_thw'] = fast_video_grid_thw

        second_per_grid_ts = self.gather_list(batch, 'second_per_grid_ts')
        if second_per_grid_ts:
            res['second_per_grid_ts'] = second_per_grid_ts
        return res


# Register the Keye VL template
register_template(KeyeTemplateMeta(MLLMTemplateType.keye_vl, template_cls=KeyeVLTemplate))


class KeyeVL1_5Template(KeyeVLTemplate):

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return super(KeyeVLTemplate, self)._post_encode(model, inputs)


register_template(
    KeyeTemplateMeta(
        MLLMTemplateType.keye_vl_1_5, template_cls=KeyeVL1_5Template, default_system='You are a helpful assistant.'))
