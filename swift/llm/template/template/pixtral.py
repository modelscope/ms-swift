# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional

import torch

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import findall


class PixtralTemplate(Template):
    image_placeholder = ['[IMG]']
    placeholder_tokens = ['[IMG]']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        idx_list = findall(input_ids, 10)
        if idx_list:
            image_inputs = processor.image_processor(images, patch_size=processor.patch_size, return_tensors='pt')
            encoded['pixel_values'] = image_inputs['pixel_values'].to(dtype=self.model_info.torch_dtype)
            encoded['image_sizes'] = image_sizes = image_inputs['image_sizes']

            def _get_new_tokens(i):
                height, width = image_sizes[i]
                num_height_tokens = height // processor.patch_size
                num_width_tokens = width // processor.patch_size
                replace_tokens = [processor.image_token * num_width_tokens + processor.image_break_token] * (
                    num_height_tokens - 1)
                replace_tokens += [processor.image_token * num_width_tokens + processor.image_end_token]
                # Flatten list
                replace_str = ''.join(replace_tokens)
                img_tokens: List[int] = self.processor.encode(replace_str, add_special_tokens=False)
                return img_tokens

            encoded['input_ids'], encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, _get_new_tokens)

        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        pixel_values = self.gather_list(batch, 'pixel_values')
        image_sizes = self.gather_list(batch, 'image_sizes')
        res = super()._data_collator(batch, padding_to=padding_to)
        if pixel_values:
            pixel_values = torch.stack(pixel_values)
            res['pixel_values'] = pixel_values
        if image_sizes:
            image_sizes = torch.stack(image_sizes)
            res['image_sizes'] = image_sizes
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.pixtral,
        prefix=['<s>{{SYSTEM}}'],
        prompt=['[INST]{{QUERY}}[/INST]'],
        chat_sep=['</s>'],
        suffix=['</s>'],
        template_cls=PixtralTemplate,
    ))
