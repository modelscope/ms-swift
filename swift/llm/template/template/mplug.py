# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional

import torch
from torch import nn

from swift.utils import get_env_args
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall
from ..vision_utils import load_video_minicpmv_mplug_owl3
from .qwen import QwenTemplateMeta


class mPlugOwl2Template(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [[-200]]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from mplug_owl2.mm_utils import process_images
        processor = self.processor
        images = inputs.images
        for i, image in enumerate(images):
            # ref: https://modelscope.cn/models/iic/mPLUG-Owl2.1
            max_edge = max(image.size)
            image = image.resize((max_edge, max_edge))
            images[i] = image
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        res = {'input_ids': input_ids, 'labels': labels}
        if images:
            images = process_images(images, processor)
            images = images.to(self.config.torch_dtype)
            res['images'] = images
        return res

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.mplug_owl2,
        template_cls=mPlugOwl2Template,
        prefix=['{{SYSTEM}}'],
        prompt=['USER: {{QUERY}}ASSISTANT:'],
        chat_sep=['</s>'],
        suffix=[['eos_token_id']],
        stop_words=['<|endoftext|>', '</s>']))


class mPlugOwl3Template(Template):
    version = None

    def _get_image_token_list(self, cut_shape):
        text = self.processor.image_processor.cut_prompt_template(img_token='<|image|>', h=cut_shape[0], w=cut_shape[1])
        text_list = text.split('<|image|>')
        res_text_list = []
        for text in text_list[:-1]:
            res_text_list += [text, '<|image|>']
        res_text_list += text_list[-1]
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
            return self.replace_video2image(load_video, inputs, lambda i: [[-100]]) + ['\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        videos = inputs.videos
        cut_enable = not videos
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, -100)
        processor = self.processor
        encoded = {}
        if images:
            image_inputs = processor.image_processor(images, cut_enable=cut_enable, return_tensors='pt')
            added_tokens_len = 0
            cut_shapes = image_inputs['cut_shape'] or [None] * 2 * len(idx_list)
            image_token_list = self.processor.encode('<|image|>', add_special_tokens=False)
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
            if self.version == '241101':
                media_offset = image_token_idx
            else:
                _range = torch.arange(len(input_ids))[:, None]
                matrix = (_range > image_token_idx[None]).sum(dim=1)
                media_offset = torch.stack([torch.zeros(matrix.shape[0], dtype=torch.long), matrix], dim=-1)[None]
            encoded.update({
                'pixel_values': image_inputs['pixel_values'],
                'media_offset': media_offset,
            })
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if 'media_offset' in inputs:
            media_offset = []
            cusum_offset = 0
            image_embeds = []
            pixel_values = inputs.pop('pixel_values')
            max_sequence_length = inputs['input_ids'].shape[1]
            for i, curr_media_offset in enumerate(inputs['media_offset']):
                if curr_media_offset is None:
                    continue
                if curr_media_offset.shape[1] < max_sequence_length:
                    padding = curr_media_offset[:, -1:, :].expand(curr_media_offset.shape[0],
                                                                  max_sequence_length - curr_media_offset.shape[1],
                                                                  curr_media_offset.shape[2])
                    curr_media_offset = torch.concat([curr_media_offset, padding], dim=1)
                media_offset.append(curr_media_offset + cusum_offset)
                image_embeds.append(model.forward_image(pixel_values[i]))
                cusum_offset += image_embeds[-1].shape[0]
            inputs['media_offset'] = torch.concat(media_offset)
            inputs['image_embeds'] = torch.concat(image_embeds)
        return inputs

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = self.fetch_inputs(batch, ['media_offset', 'pixel_values'])
        for b in batch:
            b.pop('pixel_values', None)
        res.update(super()._data_collator(batch, padding_to=padding_to))
        return res


class mPlugOwl3_241101Template(mPlugOwl3Template):
    version = '241101'

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if 'pixel_values' in inputs:
            pixel_values = inputs.pop('pixel_values')
            inputs['image_embeds'] = torch.concat([model.forward_image(pv) for pv in pixel_values])
        else:
            inputs['media_offset'] = [None] * inputs['input_ids'].shape[0]
        return inputs


@dataclass
class mPlugOwl3TemplateMeta(QwenTemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'])
    default_system: Optional[str] = None
    system_prefix: Optional[Prompt] = None


register_template(mPlugOwl3TemplateMeta(MLLMTemplateType.mplug_owl3, template_cls=mPlugOwl3Template))

register_template(mPlugOwl3TemplateMeta(MLLMTemplateType.mplug_owl3_241101, template_cls=mPlugOwl3_241101Template))


class DocOwl2Template(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return [f'<img {index + 1}>', [-200]]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if inputs.images:
            image_tensor, patch_positions, _ = self.processor._process_image(inputs.images)
            image_tensor = image_tensor.to(self.config.torch_dtype)
            encoded.update({'images': image_tensor, 'patch_positions': patch_positions})
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        keys = ['images', 'patch_positions']
        res = self.fetch_inputs(batch, keys)
        for key in keys:
            val = res.get(key)
            if val:
                res[key] = torch.concat([v for v in val if v is not None])
        res.update(super()._data_collator(batch, padding_to=padding_to))
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.doc_owl2,
        prefix=['<s>'],
        prompt=[' USER: {{QUERY}} ASSISTANT:'],
        chat_sep=['</s>'],
        suffix=['</s>'],
        template_cls=DocOwl2Template,
    ))
