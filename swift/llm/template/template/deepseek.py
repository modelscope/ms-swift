# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Context, findall, gather_list
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta

register_template(
    TemplateMeta(
        LLMTemplateType.deepseek,
        prefix=[['bos_token_id']],
        prompt=['User: {{QUERY}}\n\nAssistant:'],
        chat_sep=[['eos_token_id']],
        system_prefix=[['bos_token_id'], '{{SYSTEM}}\n\n'],
    ))

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

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images')
        processor = self.tokenizer.processor
        input_ids, labels = inputs['input_ids'], inputs['labels']
        idx_list = findall(input_ids, processor.image_id)  # '<image_placeholder>'
        new_input_ids, new_labels = [], []
        lo = 0
        for hi in idx_list:
            new_input_ids += input_ids[lo:hi]
            if labels is not None:
                new_labels += labels[lo:hi]
            new_input_ids += [processor.image_id] * processor.num_image_tokens
            new_labels += [-100] * processor.num_image_tokens
            lo = hi + 1
        new_input_ids += input_ids[lo:]
        if labels is not None:
            new_labels += labels[lo:]
        else:
            new_labels = None
        from deepseek_vl.models.processing_vlm import VLChatProcessorOutput
        images_outputs = processor.image_processor(images, return_tensors='pt')
        output = VLChatProcessorOutput(
            sft_format=None,
            input_ids=torch.tensor(new_input_ids),
            pixel_values=images_outputs.pixel_values,
            num_image_tokens=torch.tensor([processor.num_image_tokens] * len(idx_list)))
        batched_output = dict(processor.batchify([output]))
        batched_output['pixel_values'] = batched_output['pixel_values'].to(dtype=self.model.dtype)
        inputs = {'input_ids': new_input_ids, 'labels': new_labels, '_data': batched_output}
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        inputs_embeds = model.prepare_inputs_embeds(**data)[0]
        return {'inputs_embeds': inputs_embeds}


register_template(
    TemplateMeta(
        MLLMTemplateType.deepseek_vl,
        prefix=['<｜begin▁of▁sentence｜>{{SYSTEM}}\n\n'],
        prompt=['User: {{QUERY}}\n\nAssistant:'],
        chat_sep=['<｜end▁of▁sentence｜>'],
        suffix=['<｜end▁of▁sentence｜>'],
        template_cls=DeepseekVLTemplate,
        default_system=('You are a helpful language and vision assistant. '
                        'You are able to understand the visual content that the user provides, '
                        'and assist the user with a variety of tasks using natural language.'),
        skip_prompt=False))

register_template(
    TemplateMeta(
        LLMTemplateType.deepseek2,
        prefix=[[100000]],
        prompt=['User: {{QUERY}}\n\nAssistant:'],
        chat_sep=[[100001]],
        suffix=[[100001]],
        system_prefix=[[100000], '{{SYSTEM}}\n\n']))

register_template(
    TemplateMeta(
        LLMTemplateType.deepseek2_5,
        prefix=['<｜begin▁of▁sentence｜>'],
        prompt=['<｜User｜>{{QUERY}}<｜Assistant｜>'],
        chat_sep=['<｜end_of_sentense｜>'],
        suffix=['<｜end_of_sentense｜>'],
        system_prefix=['<｜begin▁of▁sentence｜>{{SYSTEM}}']))
