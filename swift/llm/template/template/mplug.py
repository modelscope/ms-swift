# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
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


class mPlugOwl3Template(Template):

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

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if len(encoded) == 0:
            return encoded
        images = inputs.images
        videos = inputs.videos
        cut_enable = not videos
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, -100)
        processor = self.processor
        encoded = {'_data': {}}
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
            encoded['_data'].update({
                'pixel_values': image_inputs['pixel_values'],
                'media_offset': media_offset,
            })
        encoded['_data']['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if 'pixel_values' in inputs:
            pixel_values = inputs.pop('pixel_values')
            inputs['image_embeds'] = model.forward_image(pixel_values)
        return inputs

    def _data_collator(self,
                       batch: List[Dict[str, Any]],
                       *,
                       padding_side: Optional[str] = None,
                       padding_to: Optional[int] = None,
                       model: Optional[nn.Module] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to, padding_side=padding_side)
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


class mPlugOwl3vTemplate(mPlugOwl3Template):

    def _encode(self, template_inputs: StdTemplateInputs) -> Dict[str, Any]:
        inputs, _ = super(mPlugOwl3Template, self)._encode(template_inputs)
        if len(inputs) == 0:
            return inputs
        images = template_inputs.images
        videos = template_inputs.videos
        cut_enable = not videos
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = findall(input_ids, -100)
        processor = self.tokenizer.processor
        inputs = {'_data': {}}
        if images:
            image_inputs = processor.image_processor(images, cut_enable=cut_enable, return_tensors='pt')
            added_tokens_len = 0
            cut_shapes = image_inputs['cut_shape'] or [None] * 2 * len(idx_list)
            image_token_list = self.tokenizer.encode('<|image|>', add_special_tokens=False)
            for idx, cut_shape in zip(idx_list, cut_shapes[::2]):
                if cut_shape:
                    token_list = self._get_image_token_list(cut_shape)
                else:
                    token_list = image_token_list
                input_ids = input_ids[:idx + added_tokens_len] + token_list + input_ids[added_tokens_len + idx + 1:]
                if labels:
                    labels = labels[:idx + added_tokens_len] + [-100] * len(token_list) + labels[added_tokens_len + idx
                                                                                                 + 1:]
                added_tokens_len += len(token_list) - 1
            image_token_idx = torch.tensor(findall(input_ids, image_token_list))

            inputs['_data'].update({
                'pixel_values': image_inputs['pixel_values'],
                'media_offset': image_token_idx,
            })
        inputs['_data']['input_ids'] = input_ids
        inputs['labels'] = labels
        return inputs

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        if 'pixel_values' in data:
            pixel_values = data.pop('pixel_values')
            data['image_embeds'] = model.forward_image(pixel_values)
        return data

    def _data_collator(self,
                       batch: List[Dict[str, Any]],
                       *,
                       padding_side: Optional[str] = None,
                       padding_to: Optional[int] = None,
                       model: Optional[nn.Module] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to, padding_side=padding_side)
        image_embeds = [b['image_embeds'] for b in batch if 'image_embeds' in b]
        if image_embeds:
            res['image_embeds'] = torch.concat(image_embeds)
        media_offset = []

        for bi, b in enumerate(batch):
            media_offset.append(b.get('media_offset', torch.tensor([]).long()))

        if media_offset:
            res['media_offset'] = media_offset
        return res


@dataclass
class mPlugOwl3TemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = None


register_template(mPlugOwl3TemplateMeta(MLLMTemplateType.mplug_owl3, template_cls=mPlugOwl3Template))

register_template(mPlugOwl3TemplateMeta(MLLMTemplateType.mplug_owl3v, template_cls=mPlugOwl3vTemplate))
