# Copyright (c) Alibaba, Inc. and its affiliates.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall


@dataclass
class MoonlightTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda:
                           ['<|im_user|>user<|im_middle|>{{QUERY}}<|im_end|><|im_assistant|>assistant<|im_middle|>'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<|im_system|>system<|im_middle|>{{SYSTEM}}<|im_end|>'])
    default_system: str = 'You are a helpful assistant'


register_template(MoonlightTemplateMeta(LLMTemplateType.moonlight))


class KimiVLTemplate(Template):
    placeholder_tokens = ['<|media_pad|>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<|media_start|>image<|media_content|><|media_pad|><|media_end|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        media_token = self._tokenize('<|media_pad|>')[0]
        idx_list = findall(input_ids, media_token)
        if inputs.images:
            image_processor = self.processor.image_processor
            image_inputs = image_processor(inputs.images, return_tensors='pt')
            image_grid_hws = image_inputs['image_grid_hws']
            merge_length = image_processor.merge_kernel_size[0] * image_processor.merge_kernel_size[1]

            def _get_new_tokens(i):
                token_len = (image_grid_hws[i].prod() // merge_length)
                return [media_token] * token_len

            input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, _get_new_tokens)
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
            encoded.update(image_inputs)
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        image_grid_hws = self.concat_tensor(batch, 'image_grid_hws', 0)
        if image_grid_hws is not None:
            res['image_grid_hws'] = image_grid_hws
        return res


register_template(MoonlightTemplateMeta(MLLMTemplateType.kimi_vl, template_cls=KimiVLTemplate))
