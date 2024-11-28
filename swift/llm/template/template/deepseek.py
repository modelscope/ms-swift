# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Prompt, findall


@dataclass
class DeepseekTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: [['bos_token_id']])
    prompt: Prompt = field(default_factory=lambda: ['User: {{QUERY}}\n\nAssistant:'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: [['eos_token_id']])
    suffix: Prompt = field(default_factory=lambda: [['eos_token_id']])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: [['bos_token_id'], '{{SYSTEM}}\n\n'])
    auto_add_bos: bool = True


register_template(DeepseekTemplateMeta(LLMTemplateType.deepseek, ))

register_template(
    TemplateMeta(
        LLMTemplateType.deepseek_coder,
        prefix=['{{SYSTEM}}'],
        prompt=['### Instruction:\n{{QUERY}}\n### Response:\n'],
        chat_sep=['\n<|EOT|>\n'],
        suffix=['\n<|EOT|>'],
        default_system=('You are an AI programming assistant, utilizing the Deepseek Coder model, '
                        'developed by Deepseek Company, and you only answer questions related to computer science. '
                        'For politically sensitive questions, security and privacy issues, '
                        'and other non-computer science questions, you will refuse to answer\n')))


class DeepseekVLTemplate(Template):
    image_placeholder = ['<image_placeholder>']
    skip_prompt = False

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        is_janus = getattr(self, 'is_janus', False)

        encoded = super()._encode(inputs)
        if len(encoded) == 0:
            return encoded
        images = inputs.images
        processor = self.processor
        input_ids, labels = encoded['input_ids'], encoded['labels']
        idx_list = findall(input_ids, processor.image_id)  # '<image_placeholder>'
        new_input_ids, new_labels = [], []
        lo = 0
        for hi in idx_list:
            new_input_ids += input_ids[lo:hi]
            if labels is not None:
                new_labels += labels[lo:hi]
            image_tokens = [processor.image_id] * processor.num_image_tokens
            if is_janus:
                image_tokens = [processor.image_start_id] + image_tokens + [processor.image_end_id]
            new_input_ids += image_tokens
            new_labels += [-100] * len(image_tokens)
            lo = hi + 1
        new_input_ids += input_ids[lo:]
        if labels is not None:
            new_labels += labels[lo:]
        else:
            new_labels = None
        if is_janus:
            from janus.models.processing_vlm import VLChatProcessorOutput
        else:
            from deepseek_vl.models.processing_vlm import VLChatProcessorOutput

        images_outputs = processor.image_processor(images, return_tensors='pt')
        output = VLChatProcessorOutput(
            sft_format=None,
            input_ids=torch.tensor(new_input_ids),
            pixel_values=images_outputs.pixel_values,
            num_image_tokens=torch.tensor([processor.num_image_tokens] * len(idx_list)))
        batched_output = dict(processor.batchify([output]))
        batched_output['pixel_values'] = batched_output['pixel_values'].to(dtype=model.dtype)
        encoded = {'input_ids': new_input_ids, 'labels': new_labels, '_data': batched_output}
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs_embeds = model.prepare_inputs_embeds(**inputs)[0]
        return {'inputs_embeds': inputs_embeds}


@dataclass
class DeepseekVLTemplateMeta(DeepseekTemplateMeta):
    default_system: Optional[str] = ('You are a helpful language and vision assistant. '
                                     'You are able to understand the visual content that the user provides, '
                                     'and assist the user with a variety of tasks using natural language.')


register_template(DeepseekVLTemplateMeta(
    MLLMTemplateType.deepseek_vl,
    template_cls=DeepseekVLTemplate,
))


class DeepseekJanus(DeepseekVLTemplate):
    is_janus = True
    image_placeholder = ['<image_placeholder>\n']


register_template(DeepseekVLTemplateMeta(MLLMTemplateType.deepseek_janus, template_cls=DeepseekJanus))

register_template(
    TemplateMeta(
        LLMTemplateType.deepseek2_5,
        prefix=['<｜begin▁of▁sentence｜>'],
        prompt=['<｜User｜>{{QUERY}}<｜Assistant｜>'],
        chat_sep=['<｜end_of_sentense｜>'],
        suffix=['<｜end_of_sentense｜>'],
        system_prefix=['<｜begin▁of▁sentence｜>{{SYSTEM}}']))
