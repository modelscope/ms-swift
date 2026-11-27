# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Word, findall
from ..vision_utils import load_image
from .utils import ChatmlTemplateMeta


@dataclass
class LFM3TemplateMeta(ChatmlTemplateMeta):
    prefix: List = field(default_factory=lambda: ['<|startoftext|>'])
    system_prefix: List = field(default_factory=lambda: ['<|startoftext|><|im_start|>system\n{{SYSTEM}}<|im_end|>\n'])
    auto_add_bos: bool = False
    is_thinking: bool = True
    thinking_prefix: str = '<think>'
    default_enable_thinking: bool = True
    # The official jinja always writes an empty think block when thinking is off.
    non_thinking_prefix: str = '<think></think>'
    history_thinking_prefix: str = '<think></think>'
    agent_template: str = 'lfm3'
    stop_words: List[Word] = field(default_factory=lambda: ['<|im_end|>'])


class LFM3Template(Template):
    # LFM3 has 13 short-conv layers whose receptive field crosses sample boundaries,
    # so packing/padding-free would leak context between samples.
    support_padding_free = False

    def _encode(self, inputs):
        messages = inputs.messages
        if messages and messages[-1]['role'] == 'tool':
            raise ValueError(
                'LFM3 renders tool results as a `tool` turn, which the query/response encoder cannot place at '
                'the end of a conversation. Append the assistant turn under evaluation, or a user turn after '
                'the tool result.')
        return super()._encode(inputs)

    def _non_thinking_prefix_start(self, messages) -> int:
        # Every assistant turn carries the empty think block once history thinking is stripped.
        return -1


class LFM3VLTemplate(LFM3Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            inputs.images[index] = load_image(inputs.images[index])
            return ['<|vision_start|><|image|><|vision_end|>']
        raise NotImplementedError('LFM3-VL only supports image inputs in swift')

    def _encode(self, inputs: StdTemplateInputs):
        encoded = super()._encode(inputs)
        if not inputs.images:
            return encoded
        media_inputs = self.processor.image_processor(images=inputs.images, return_tensors='pt')
        image_grid_thw = media_inputs['image_grid_thw']
        image_token_id = self.processor.image_token_id
        merge_length = self.processor.image_processor.merge_size**2
        idx_list = findall(encoded['input_ids'], image_token_id)
        encoded['input_ids'], encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
            encoded['input_ids'], encoded['labels'], encoded.get('loss_scale'), idx_list,
            lambda i: [image_token_id] * int(image_grid_thw[i].prod() // merge_length))
        encoded.update(media_inputs)
        return encoded


register_template(LFM3TemplateMeta(LLMTemplateType.lfm3, template_cls=LFM3Template))
register_template(LFM3TemplateMeta(MLLMTemplateType.lfm3_vl, template_cls=LFM3VLTemplate))
