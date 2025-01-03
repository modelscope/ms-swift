# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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


register_template(DeepseekTemplateMeta(LLMTemplateType.deepseek, ))

register_template(
    TemplateMeta(
        LLMTemplateType.deepseek_coder,
        prefix=['{{SYSTEM}}'],
        prompt=['### Instruction:\n{{QUERY}}\n### Response:\n'],
        chat_sep=['\n<|EOT|>\n'],
        suffix=['\n<|EOT|>'],
        stop_words=['<|EOT|>'],
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
        encoded = {'output': output, 'input_ids': new_input_ids, 'labels': new_labels}
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs['pixel_values'] = inputs['pixel_values'].to(dtype=self.config.torch_dtype)
        inputs_embeds = model.prepare_inputs_embeds(**inputs)
        return {'inputs_embeds': inputs_embeds}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        output = self.fetch_inputs(batch, ['output'])['output']
        batched_output = dict(self.processor.batchify(output))
        res = super()._data_collator(batch, padding_to=padding_to)
        return {**batched_output, **res}


@dataclass
class DeepseekVLTemplateMeta(DeepseekTemplateMeta):
    default_system: Optional[str] = ('You are a helpful language and vision assistant. '
                                     'You are able to understand the visual content that the user provides, '
                                     'and assist the user with a variety of tasks using natural language.')
    placeholder_tokens: List[str] = field(default_factory=lambda: ['<image_placeholder>'])


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
        LLMTemplateType.deepseek_v2_5,
        prefix=['<｜begin▁of▁sentence｜>{{SYSTEM}}'],
        prompt=['<｜User｜>{{QUERY}}<｜Assistant｜>'],
        chat_sep=['<｜end▁of▁sentence｜>'],
        suffix=['<｜end▁of▁sentence｜>']))


class DeepseekVL2Template(DeepseekVLTemplate):
    image_placeholder = ['<image>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from deepseek_vl2.models.processing_deepseek_vl_v2 import VLChatProcessorOutput
        encoded = Template._encode(self, inputs)
        images = inputs.images
        processor = self.processor
        input_ids, labels = encoded['input_ids'], encoded['labels']
        images_seq_mask = [False] * len(input_ids)
        idx_list = findall(input_ids, processor.image_token_id)  # '<image>'
        _, images_list, _, images_spatial_crop, num_image_tokens = processor.tokenize_with_images(
            '<image>' * len(images), images, cropping=len(images) <= 2)
        new_num_tokens = 0
        for idx, n_image_tokens in zip(idx_list, num_image_tokens):
            image_tokens = [processor.image_token_id] * n_image_tokens
            input_ids = input_ids[:idx] + image_tokens + input_ids[idx + 1:]
            if labels is not None:
                labels = labels[:idx] + [-100] * n_image_tokens + labels[idx + 1:]
            images_seq_mask = images_seq_mask[:idx] + [True] * n_image_tokens + images_seq_mask[idx + 1:]
            new_num_tokens += n_image_tokens - 1

        output = VLChatProcessorOutput(
            sft_format=None,
            input_ids=torch.tensor(input_ids),
            target_ids=torch.tensor(input_ids),
            images=torch.stack(images_list) if images_list else torch.zeros((0, 3, 384, 384)),
            images_seq_mask=torch.tensor(images_seq_mask),
            images_spatial_crop=torch.tensor(images_spatial_crop),
            num_image_tokens=num_image_tokens)
        output.images = output.images.to(dtype=self.config.torch_dtype)
        encoded = {'output': output, 'input_ids': input_ids, 'labels': labels}
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs['images_seq_mask'] = inputs['images_seq_mask'].to(torch.bool)
        inputs['images_spatial_crop'] = inputs['images_spatial_crop'].to(torch.long)
        inputs_embeds = model.prepare_inputs_embeds(**inputs)
        return {'inputs_embeds': inputs_embeds}


register_template(
    TemplateMeta(
        MLLMTemplateType.deepseek_vl2,
        prefix=['<｜begin▁of▁sentence｜>{{SYSTEM}}'],
        prompt=['<|User|>: {{QUERY}}\n\n<|Assistant|>:'],
        chat_sep=['<｜end▁of▁sentence｜>'],
        suffix=['<｜end▁of▁sentence｜>'],
        template_cls=DeepseekVL2Template,
        placeholder_tokens=['<image>']))
