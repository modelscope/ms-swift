# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from torch import nn

from swift.utils import get_env_args
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, findall, gather_list
from ..vision_utils import load_video_minicpmv_mplug_owl3, replace_video2image
from .qwen import QwenTemplateMeta
from .utils import DEFAULT_SYSTEM


class mPlugOwl3Template(Template):
    system = None

    def _get_image_token_list(self, cut_shape):
        processor = self.processor
        text = processor.image_processor.cut_prompt_template(img_token='<|image|>', h=cut_shape[0], w=cut_shape[1])
        text_list = text.split('<|image|>')
        if text_list[-1] == '':
            text_list.pop()
        res_text_list = []
        for text in text_list:
            res_text_list += [text, '<|image|>']
        token_list = self._encode_context_list(res_text_list)[0]
        return token_list

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type in {'image', 'video'}
        max_num_frames = get_env_args('max_num_frames', int, 16)
        load_video = partial(load_video_minicpmv_mplug_owl3, max_num_frames=max_num_frames)
        if media_type == 'image':
            return [[-100], '\n']
        elif media_type == 'video':
            return replace_video2image(load_video, inputs, lambda i: [[-100]]) + ['\n']

    def _encode(self, inputs: StdTemplateInputs, *, model: Optional[nn.Module] = None) -> Dict[str, Any]:
        inputs = super()._encode(inputs)
        if len(inputs) == 0:
            return inputs
        images = inputs.images
        videos = inputs.videos
        cut_enable = not videos
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = findall(input_ids, -100)
        processor = self.processor
        inputs = {'_data': {}}
        if images:
            image_inputs = processor.image_processor(images, cut_enable=cut_enable, return_tensors='pt')
            added_tokens_len = 0
            cut_shapes = image_inputs['cut_shape'] or [None] * len(idx_list)
            image_token_list = self.processor.encode('<|image|>', add_special_tokens=False)
            for idx, cut_shape in zip(idx_list, cut_shapes):
                if cut_shape:
                    token_list = self._get_image_token_list(cut_shape)
                else:
                    token_list = image_token_list
                input_ids = input_ids[:idx + added_tokens_len] + token_list + input_ids[added_tokens_len + idx + 1:]
                if labels:
                    labels = labels[:idx + added_tokens_len] + [-100] * len(token_list) + labels[added_tokens_len + idx
                                                                                                 + 1:]
                added_tokens_len += len(token_list) - 1
            image_token_idx = torch.tensor(findall(input_ids, image_token_list))[None]
            _range = torch.arange(len(input_ids))[:, None]
            matrix = (_range > image_token_idx).sum(dim=1)
            media_offset = torch.stack([torch.zeros(matrix.shape[0], dtype=torch.long), matrix], dim=-1)[None]
            inputs['_data'].update({
                'pixel_values': image_inputs['pixel_values'],
                'media_offset': media_offset,
            })
        inputs['_data']['input_ids'] = input_ids
        inputs['labels'] = labels
        return inputs

    def post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if 'pixel_values' in inputs:
            pixel_values = inputs.pop('pixel_values')
            inputs['image_embeds'] = model.forward_image(pixel_values)
        return inputs

    def data_collator(self,
                      batch: List[Dict[str, Any]],
                      *,
                      padding_side: Optional[str] = None,
                      padding_to: Optional[int] = None,
                      model: Optional[nn.Module] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to=padding_to, padding_side=padding_side)
        image_embeds = [b['image_embeds'] for b in batch if 'image_embeds' in b]
        if image_embeds:
            res['image_embeds'] = torch.concat(image_embeds)
        media_offset = []
        cusum_offset = 0

        for bi, b in enumerate(batch):
            if 'media_offset' in b:
                max_sequence_length = res['input_ids'].shape[1]
                curr_media_offset = b['media_offset']
                if curr_media_offset.shape[1] < max_sequence_length:
                    padding = curr_media_offset[:, -1:, :].expand(curr_media_offset.shape[0],
                                                                  max_sequence_length - curr_media_offset.shape[1],
                                                                  curr_media_offset.shape[2])
                    curr_media_offset = torch.concat([curr_media_offset, padding], dim=1)
                media_offset.append(curr_media_offset + cusum_offset)
                cusum_offset += image_embeds[bi].shape[0]

        # media_offset = [b['media_offset'] for b in batch if 'media_offset' in b]

        if media_offset:
            res['media_offset'] = torch.concat(media_offset)
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.mplug_owl3, template_cls=mPlugOwl3Template, default_system=None))
