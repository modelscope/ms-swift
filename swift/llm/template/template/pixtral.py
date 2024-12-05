# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Literal, Optional, Tuple

from torch import nn

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, findall
from .utils import DEFAULT_SYSTEM


class PixtralTemplate(Template):
    image_placeholder = ['[IMG]']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if len(encoded) == 0:
            return encoded
        processor = self.processor
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, 10)
        if idx_list:
            image_inputs = processor.image_processor(images, patch_size=processor.patch_size, return_tensors='pt')
            encoded['pixel_values'] = image_inputs['pixel_values'][0]
            image_sizes = image_inputs['image_sizes'][0]
            added_tokens_len = 0
            for idx, image_size in zip(idx_list, image_sizes):
                height, width = image_size
                num_height_tokens = height // processor.patch_size
                num_width_tokens = width // processor.patch_size
                replace_tokens = [processor.image_token * num_width_tokens + processor.image_break_token] * (
                    num_height_tokens - 1)
                replace_tokens += [processor.image_token * num_width_tokens + processor.image_end_token]
                # Flatten list
                replace_str = ''.join(replace_tokens)
                img_tokens: List[int] = self.processor.encode(replace_str, add_special_tokens=False)
                input_ids = input_ids[:idx + added_tokens_len] + img_tokens + input_ids[idx + added_tokens_len + 1:]
                if labels is not None:
                    labels = labels[:idx + added_tokens_len] + [-100] * len(img_tokens) + labels[idx + added_tokens_len
                                                                                                 + 1:]
                added_tokens_len += len(img_tokens) - 1
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels

        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        pixel_values = self.gather_list(batch, 'pixel_values')
        res = super()._data_collator(batch, padding_to=padding_to)
        if pixel_values:
            res['pixel_values'] = pixel_values
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.pixtral,
        prefix=['<s>{{SYSTEM}}'],
        prompt=['[INST]{{QUERY}}[/INST]'],
        chat_sep=['</s>'],
        suffix=['</s>'],
        template_cls=PixtralTemplate,
        placeholder_tokens=['[IMG]'],
    ))
