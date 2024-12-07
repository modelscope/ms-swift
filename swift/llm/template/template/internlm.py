# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.utils import get_env_args
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Prompt, Word
from .utils import ChatmlTemplateMeta

INTERNLM_SYSTEM = (
    'You are an AI assistant whose name is InternLM (书生·浦语).\n'
    '- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). '
    'It is designed to be helpful, honest, and harmless.\n'
    '- InternLM (书生·浦语) can understand and communicate fluently in the language chosen '
    'by the user such as English and 中文.')

register_template(
    TemplateMeta(
        LLMTemplateType.internlm,
        prefix=['<s>'],
        prompt=['<|User|>:{{QUERY}}\n<|Bot|>:'],
        chat_sep=['<eoa>\n'],
        suffix=['<eoa>'],
        default_system=INTERNLM_SYSTEM,
        system_prefix=['<s><|System|>:{{SYSTEM}}\n']))

register_template(ChatmlTemplateMeta(LLMTemplateType.internlm2, default_system=INTERNLM_SYSTEM))


class InternLMXComposer2Template(Template):
    image_placeholder = ['</s>']
    version = 'v2'
    skip_prompt = False
    use_model = True

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        model = self.model
        encoded = super()._encode(inputs)
        if len(encoded) == 0:
            return encoded
        images = inputs.images or []

        if self.version == 'v2.5':
            hd_num = 24
            if len(images) > 1:
                hd_num = 6
            hd_num = get_env_args('hd_num', int, hd_num)
            Image_transform = get_class_from_dynamic_module('ixc_utils.Image_transform', model.model_dir)
            images = [Image_transform(image, hd_num=hd_num) for image in images]
        elif self.version == 'v2-4khd':
            hd_num = 55
            hd_num = get_env_args('hd_num', int, hd_num)
            HD_transform = get_class_from_dynamic_module('ixc_utils.HD_transform', model.model_dir)
            images = [HD_transform(image, hd_num=hd_num) for image in images]
        images = [model.vis_processor(image).to(model.dtype) for image in images]
        encoded['images'] = images
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = inputs['input_ids'][0].tolist()
        labels = inputs.get('labels')
        images = inputs['images']
        if len(images) > 0:  # ignore <s>
            input_ids = input_ids[1:]
            if labels is not None:
                labels = labels[1:]
        input_ids.append(2)  # add dummy </s>
        if labels is not None:
            labels = labels[0].tolist()
            labels.append(2)
        else:
            labels = []
        res_inputs_embeds = []
        res_labels = []
        wrap_im_mask = []
        pre_i, i, idx = 0, 0, 0
        device = model.device
        internlm2_model = model.model
        if not hasattr(internlm2_model, 'tok_embeddings'):
            internlm2_model = internlm2_model.model
        tok_embeddings = internlm2_model.tok_embeddings
        if len(images) > 0:
            images = torch.concat([model.img2emb(image[None])[0] for image in images], dim=0)
        while i < len(input_ids):
            if input_ids[i] == 2:  # replace_token
                res_input_ids = torch.tensor([1] + input_ids[pre_i:i], device=device)
                res_inputs_embeds.append(tok_embeddings(res_input_ids[None])[0])
                wrap_im_mask += [0] * len(res_input_ids)
                res_labels += [-100] + labels[pre_i:i]
                if len(images) > 0 and idx < images.shape[0]:
                    res_inputs_embeds.append(images[idx].to(device))
                    wrap_im_mask += [1] * images.shape[1]
                    res_labels += [-100] * images.shape[1]
                idx += 1
                i += 1
                pre_i = i
                continue
            i += 1
        if len(labels) == 0:
            res_labels = None
        res_inputs_embeds = torch.concat(res_inputs_embeds, dim=0)
        wrap_im_mask = torch.tensor(wrap_im_mask, dtype=torch.bool, device=device)[None]
        return {'inputs_embeds': res_inputs_embeds, 'im_mask': wrap_im_mask, 'labels': res_labels}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        if 'im_mask' in batch[0]:
            im_mask = [b['im_mask'][0] for b in batch]
            im_mask = self._pad_sequence(im_mask, 0)
            res['im_mask'] = im_mask
        return res


@dataclass
class Xcomposer2TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<s>'])
    prompt: Prompt = field(
        default_factory=lambda: ['[UNUSED_TOKEN_146]user\n{{QUERY}}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['[UNUSED_TOKEN_145]\n'])
    suffix: Prompt = field(default_factory=lambda: ['[UNUSED_TOKEN_145]'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<s>[UNUSED_TOKEN_146]system\n{{SYSTEM}}[UNUSED_TOKEN_145]\n'])
    stop_words: List[Word] = field(default_factory=lambda: ['<|im_end|>'])


register_template(
    Xcomposer2TemplateMeta(
        MLLMTemplateType.xcomposer2,
        template_cls=InternLMXComposer2Template,
        default_system=('You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
                        '- InternLM-XComposer (浦语·灵笔) is a conversational language model that is developed by '
                        'Shanghai AI Laboratory (上海人工智能实验室). '
                        'It is designed to be helpful, honest, and harmless.\n'
                        '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen '
                        'by the user such as English and 中文.'),
    ))


class InternLMXComposer2_5Template(InternLMXComposer2Template):
    system = ('You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
              '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model '
              'that is developed by Shanghai AI Laboratory (上海人工智能实验室). '
              'It is designed to be helpful, honest, and harmless.\n'
              '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen '
              'by the user such as English and 中文.\n'
              '- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively '
              'based on the provided image.')
    version = 'v2.5'


class InternLMXComposer2_4khdTemplate(InternLMXComposer2Template):
    version = 'v2-4khd'


register_template(
    Xcomposer2TemplateMeta(
        MLLMTemplateType.xcomposer2_5,
        template_cls=InternLMXComposer2_5Template,
        default_system=InternLMXComposer2_5Template.system))

register_template(
    Xcomposer2TemplateMeta(
        MLLMTemplateType.xcomposer2_4khd,
        template_cls=InternLMXComposer2_4khdTemplate,
        default_system=InternLMXComposer2_5Template.system))
