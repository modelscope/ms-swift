# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from swift.utils import upper_bound
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context

register_template(
    TemplateMeta(
        LLMTemplateType.gemma,
        prefix=['<bos>'],
        prompt=['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'],
        chat_sep=['<end_of_turn>\n'],
        suffix=['<end_of_turn>'],
        system_prefix=['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n']))


class PaliGemmaTemplate(Template):

    def _check_inputs(self, inputs: StdTemplateInputs):
        images = inputs.images or []
        assert len(images) <= 1

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        if self.mode == 'vllm':
            self.prompt = ['{{QUERY}}']
            return []
        else:
            self.prompt = ['{{QUERY}}\n']
            return ['<image>' * self.processor.image_seq_length + '<bos>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if len(encoded) == 0:
            return encoded
        raw_image = inputs.images
        processor = self.processor
        if encoded['labels'] is not None:
            n = upper_bound(0, len(encoded['labels']), lambda idx: encoded['labels'][idx] == -100)
            n2 = len(encoded['labels']) - n
            encoded['token_type_ids'] = [0] * n + [1] * n2
        else:
            encoded['token_type_ids'] = [0] * len(encoded['input_ids'])
        if raw_image:
            model_inputs = processor(text=encoded.to_history()['query'], images=raw_image[0], return_tensors='pt')
            encoded['pixel_values'] = model_inputs['pixel_values']
        return encoded

    def _data_collator(self,
                       batch: List[Dict[str, Any]],
                       *,
                       padding_side: Optional[str] = None,
                       padding_to: Optional[int] = None,
                       model: Optional[nn.Module] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_side=padding_side, padding_to=padding_to)
        token_type_ids = [torch.tensor(b['token_type_ids']) for b in batch]
        token_type_ids = self._pad_sequence(token_type_ids, 0, padding_side=padding_side)
        res['token_type_ids'] = token_type_ids
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.paligemma,
        prefix=[],
        prompt=['{{QUERY}}\n'],
        chat_sep=None,
        suffix=['<eos>'],
        template_cls=PaliGemmaTemplate,
    ))
