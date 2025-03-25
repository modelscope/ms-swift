# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Literal, Optional

import torch

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, findall
from .llm import mistral_2501_system


class Mistral2503Template(Template):
    placeholder_tokens = ['[IMG]']
    image_token = 10

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return ['[IMG]']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, self.image_token)
        if idx_list:
            image_inputs = processor.image_processor(images, patch_size=processor.patch_size, return_tensors='pt')
            encoded['pixel_values'] = image_inputs['pixel_values'].to(self.model_info.torch_dtype)
            encoded['image_sizes'] = image_sizes = image_inputs['image_sizes']

            def _get_new_tokens(i):
                height, width = image_sizes[i]
                num_height_tokens = height // (processor.patch_size * processor.spatial_merge_size)
                num_width_tokens = width // (processor.patch_size * processor.spatial_merge_size)
                replace_tokens = [[processor.image_token] * num_width_tokens + [processor.image_break_token]
                                  ] * num_height_tokens
                # Flatten list
                replace_tokens = [item for sublist in replace_tokens for item in sublist]
                replace_tokens[-1] = processor.image_end_token
                replace_str = ''.join(replace_tokens)
                return processor.encode(replace_str, add_special_tokens=False)

            encoded['input_ids'], encoded['labels'] = self._extend_tokens(input_ids, labels, idx_list, _get_new_tokens)

        return encoded


register_template(
    TemplateMeta(
        MLLMTemplateType.mistral_2503,
        prefix=['<s>'],
        prompt=['[INST]{{QUERY}}[/INST]'],
        chat_sep=['</s>'],
        suffix=['</s>'],
        system_prefix=['<s>[SYSTEM_PROMPT]{{SYSTEM}}[/SYSTEM_PROMPT]'],
        default_system=mistral_2501_system,
        template_cls=Mistral2503Template))
