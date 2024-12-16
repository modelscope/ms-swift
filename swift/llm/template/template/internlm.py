# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch
from PIL import Image
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.utils import get_env_args
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, Word
from ..vision_utils import load_file
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

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'video':
            inputs.images.insert(inputs.image_idx, inputs.videos[index])
            inputs.image_idx += 1
        return self.image_placeholder

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        model = self.model
        encoded = super()._encode(inputs)
        images = inputs.images or []

        if self.version == 'v2.5':
            hd_num = 24
            if len(images) > 1:
                hd_num = 6
            hd_num = get_env_args('hd_num', int, hd_num)
            images_origin = images
            images = []
            for image in images_origin:
                if isinstance(image, Image.Image):
                    Image_transform = get_class_from_dynamic_module('ixc_utils.Image_transform', model.model_dir)
                    images.append(Image_transform(image, hd_num=hd_num))
                else:
                    load_video = get_class_from_dynamic_module('ixc_utils.load_video', model.model_dir)
                    frame2img = get_class_from_dynamic_module('ixc_utils.frame2img', model.model_dir)
                    Video_transform = get_class_from_dynamic_module('ixc_utils.Video_transform', model.model_dir)
                    image = load_video(load_file(image))
                    image = frame2img(image, model.font)
                    images.append(Video_transform(image, hd_num=hd_num))
        elif self.version == 'v2-4khd':
            hd_num = 55
            hd_num = get_env_args('hd_num', int, hd_num)
            HD_transform = get_class_from_dynamic_module('ixc_utils.HD_transform', model.model_dir)
            images = [HD_transform(image, hd_num=hd_num) for image in images]
        images = [model.vis_processor(image).to(model.dtype) for image in images]
        encoded['images'] = images
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        batch_size = len(inputs['input_ids'])
        res = []
        im_mask = []
        length = inputs['length']
        for i in range(batch_size):
            input_ids = inputs['input_ids'][i].tolist()[:length[i]]
            input_ids.append(2)  # add dummy </s>
            labels = inputs.get('labels')
            if labels is not None:
                labels = labels[i].tolist()[:length[i]]
                labels.append(2)
            else:
                labels = []
            images = inputs['images'][i]
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
            add_bos = False
            while i < len(input_ids):
                if input_ids[i] == 2:  # replace_token
                    res_input_ids = torch.tensor(([1] if add_bos else []) + input_ids[pre_i:i], device=device)
                    if not add_bos and self.version != 'v2.5':
                        add_bos = True
                    res_inputs_embeds.append(tok_embeddings(res_input_ids[None])[0])
                    wrap_im_mask += [0] * len(res_input_ids)
                    res_labels += ([-100] if add_bos else []) + labels[pre_i:i]
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
            im_mask.append(torch.tensor(wrap_im_mask, dtype=torch.bool, device=device))
            res.append({'inputs_embeds': torch.concat(res_inputs_embeds, dim=0), 'labels': res_labels})
        res = Template._data_collator(self, res)
        res['im_mask'] = self._pad_sequence(im_mask, 0)
        return res

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        res['length'] = [len(b['input_ids']) for b in batch]
        res.update(self.fetch_inputs(batch, ['images']))
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
