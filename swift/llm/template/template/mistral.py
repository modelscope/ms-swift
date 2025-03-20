# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional

import torch

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from .llm import mistral_2501_system
from ..utils import findall

class Mistral2503Template(Template):
    placeholder_tokens = ['<IMG>']
    image_token = 10
    boi_token_id = [1060, 125250, 1062]
    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        processor = self.processor
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, self.boi_token_id)
        if idx_list:
            image_inputs = processor.image_processor(images, patch_size=processor.patch_size, return_tensors='pt')
            encoded['pixel_values'] = image_inputs['pixel_values']
            encoded['image_sizes'] = image_sizes = image_inputs['image_sizes']
            added_tokens_len = 0
            for idx, image_size in zip(idx_list, image_sizes):
                height, width = image_size
                num_height_tokens = height // (processor.patch_size * processor.spatial_merge_size)
                num_width_tokens = width // (processor.patch_size * processor.spatial_merge_size)
                replace_tokens = [processor.image_token * num_width_tokens + processor.image_break_token] * num_height_tokens
                # Flatten list
                replace_tokens = [item for sublist in replace_tokens for item in sublist]
                replace_tokens[-1] = processor.image_end_token
                replace_str = "".join(replace_tokens)
                img_tokens: List[int] = processor.tokenizer(replace_str, add_special_tokens=False)
                input_ids = input_ids[:idx + added_tokens_len] + img_tokens + input_ids[idx + added_tokens_len + 1:]
                if labels is not None:
                    labels = labels[:idx + added_tokens_len] + [-100] * len(img_tokens) + labels[idx + added_tokens_len
                                                                                                 + 1:]
                added_tokens_len += len(img_tokens) - 1
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels

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
