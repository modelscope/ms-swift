# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch

from swift.utils import upper_bound
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall


@dataclass
class GemmaTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<bos>'])
    prompt: Prompt = field(
        default_factory=lambda: ['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<end_of_turn>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<end_of_turn>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'])


register_template(GemmaTemplateMeta(LLMTemplateType.gemma))


class PaliGemmaTemplate(Template):
    placeholder_tokens = ['<image>']

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
        raw_image = inputs.images
        processor = self.processor
        if encoded['labels'] is not None:
            n = upper_bound(0, len(encoded['labels']), lambda idx: encoded['labels'][idx] == -100)
            n2 = len(encoded['labels']) - n
            encoded['token_type_ids'] = [0] * n + [1] * n2
        else:
            encoded['token_type_ids'] = [0] * len(encoded['input_ids'])
        if raw_image:
            model_inputs = processor(text='<image>' * len(raw_image), images=raw_image, return_tensors='pt')
            encoded['pixel_values'] = model_inputs['pixel_values'].to(self.model_info.torch_dtype)
        return encoded


register_template(
    TemplateMeta(
        MLLMTemplateType.paligemma,
        prefix=[],
        prompt=['{{QUERY}}\n'],
        chat_sep=None,
        suffix=['<eos>'],
        template_cls=PaliGemmaTemplate,
    ))


@dataclass
class Gemma3TextTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<bos>'])
    prompt: Prompt = field(
        default_factory=lambda: ['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<end_of_turn>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<end_of_turn>'])


class Gemma3Template(Template):

    def _swift_encode(self, inputs: StdTemplateInputs):
        if inputs.system is not None:
            system = inputs.system
            inputs.system = None
            inputs.messages[0]['content'] = system + '\n\n' + inputs.messages[0]['content']
        for message in inputs.messages:
            if message['role'] == 'assistant' and isinstance(message['content'], str):
                message['content'] = message['content'].strip('\n')
        return super()._swift_encode(inputs)


register_template(Gemma3TextTemplateMeta(LLMTemplateType.gemma3_text, template_cls=Gemma3Template))


class Gemma3VisionTemplate(Gemma3Template):
    boi_token_id = 255999
    placeholder_tokens = ['<start_of_image>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return ['<start_of_image>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from transformers.models.gemma3.processing_gemma3 import Gemma3ProcessorKwargs

        encoded = super()._encode(inputs)
        if inputs.images:
            input_ids = encoded['input_ids']
            labels = encoded['labels']
            idx_list = findall(input_ids, self.boi_token_id)
            img_tokens = self.tokenizer.encode(self.processor.full_image_sequence)
            input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, lambda _: img_tokens)

            # TODO: customize
            processor_kwargs = Gemma3ProcessorKwargs._defaults['images_kwargs']
            image_inputs = self.processor.image_processor(inputs.images, **processor_kwargs)
            image_inputs['pixel_values'] = torch.as_tensor(image_inputs['pixel_values'])
            image_inputs.pop('num_crops')

            array_ids = np.array(input_ids)
            mm_token_type_ids = np.zeros_like(input_ids)
            mm_token_type_ids[array_ids == self.processor.image_token_id] = 1
            encoded['token_type_ids'] = mm_token_type_ids.tolist()
            encoded['input_ids'] = input_ids
            encoded['pixel_values'] = image_inputs['pixel_values']
            encoded['labels'] = labels
        return encoded


register_template(GemmaTemplateMeta(MLLMTemplateType.gemma3_vision, template_cls=Gemma3VisionTemplate))
