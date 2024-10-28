# Copyright (c) Alibaba, Inc. and its affiliates.

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..base import Template
from ..constant import TemplateType
from ..register import TemplateMeta, register_template
from ..utils import Context, Prompt, findall

# ref: https://github.com/facebookresearch/llama/blob/main/llama/generation.py
LLAMA_DEFAULT_SYSTEM = (
    'You are a helpful, respectful and honest assistant. '
    'Always answer as helpfully as possible, while being safe. '
    'Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n\n'
    'If a question does not make any sense, or is not factually coherent, '
    'explain why instead of answering something not correct. '
    "If you don't know the answer to a question, please don't share false information.")

register_template(
    TemplateMeta(
        TemplateType.llama, ['<s>[INST] '], ['{{QUERY}} [/INST]'], ['</s><s>[INST] '], ['</s>'],
        default_system=LLAMA_DEFAULT_SYSTEM,
        system_prefix=['<s>[INST] <<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n']))


@dataclass
class Llama3TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<|begin_of_text|>'])
    prompt: Prompt = field(default_factory=lambda: [
        '<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    ])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|eot_id|>'])
    suffix: Prompt = field(default_factory=lambda: ['<|eot_id|>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{SYSTEM}}<|eot_id|>'])
    tool_prompt: Optional[Prompt] = field(default_factory=lambda: [
        '<|start_header_id|>tool<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    ])

    default_tools_prompt: str = 'toolbench'


register_template(Llama3TemplateMeta(TemplateType.llama3))


def _get_llama3_2_prefix() -> Prompt:
    now = datetime.now()
    date_string = now.strftime('%d %b %Y')
    date_prompt = f'Cutting Knowledge Date: December 2023\nToday Date: {date_string}'
    return [f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{date_prompt}\n\n' '{{SYSTEM}}<|eot_id|>']


@dataclass
class Llama3_2TemplateMeta(Llama3TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: _get_llama3_2_prefix())
    system_prefix: Optional[Prompt] = None


register_template(Llama3_2TemplateMeta(TemplateType.llama3_2))


class Llama3_2VisionTemplate(Template):

    def replace_tag(self, media_type, index, example) -> List[Context]:
        assert media_type == 'image'
        return ['<|image|>']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        from transformers.models.mllama.processing_mllama import (get_cross_attention_token_mask,
                                                                  convert_sparse_cross_attention_mask_to_dense)
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example['images']
        if images:
            input_ids = inputs['input_ids']
            processor = self.tokenizer.processor
            image_features = processor.image_processor(images, return_tensors='pt')
            num_tiles = image_features.pop('num_tiles')
            inputs.update(image_features)

            cross_attention_token_mask = [get_cross_attention_token_mask(input_ids, processor.image_token_id)]
            cross_attention_mask = convert_sparse_cross_attention_mask_to_dense(
                cross_attention_token_mask,
                num_tiles=num_tiles,
                max_num_tiles=processor.image_processor.max_image_tiles,
                length=len(input_ids),
            )
            inputs['cross_attention_mask'] = torch.tensor(cross_attention_mask)

        return inputs, {}

    def data_collator(self,
                      batch: List[Dict[str, Any]],
                      *,
                      padding_side: Optional[str] = None,
                      padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_side=padding_side, padding_to=padding_to)
        for key in ['aspect_ratio_ids', 'aspect_ratio_mask']:
            value = [b[key] for b in batch if b.get(key) is not None]
            if value:
                res[key] = torch.concat(value)

        cross_attention_mask = [
            b['cross_attention_mask'][0] for b in batch if b.get('cross_attention_mask') is not None
        ]
        if cross_attention_mask:
            res['cross_attention_mask'] = self.pad_sequence(cross_attention_mask, 0, self.padding_side)
        return res


register_template(Llama3TemplateMeta(TemplateType.llama3_2_vision, template_cls=Llama3_2VisionTemplate))
