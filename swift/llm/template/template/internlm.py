# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Literal, Optional, Tuple

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Context, findall, gather_list
from .utils import DEFAULT_SYSTEM

INTERNLM_SYSTEM = (
    'You are an AI assistant whose name is InternLM (书生·浦语).\n'
    '- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). '
    'It is designed to be helpful, honest, and harmless.\n'
    '- InternLM (书生·浦语) can understand and communicate fluently in the language chosen '
    'by the user such as English and 中文.')

register_template(
    TemplateType.internlm,
    Template(['<s>'], ['<|User|>:{{QUERY}}\n<|Bot|>:'], ['<eoa>\n'], ['<eoa>'], INTERNLM_SYSTEM,
             ['<s><|System|>:{{SYSTEM}}\n']))


class Internlm2Template(ChatmlTemplate):
    system = INTERNLM_SYSTEM


register_template(TemplateType.internlm2, Internlm2Template())


class InternLMXComposer2Template(Template):
    INTERNLM_XCOMPOSER_SYSTEM = (
        'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
        '- InternLM-XComposer (浦语·灵笔) is a conversational language model that is developed by '
        'Shanghai AI Laboratory (上海人工智能实验室). '
        'It is designed to be helpful, honest, and harmless.\n'
        '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen '
        'by the user such as English and 中文.')
    image_placeholder = ['</s>']

    def __init__(self, version):
        prefix = ['<s>']
        prompt = ['[UNUSED_TOKEN_146]user\n{{QUERY}}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n']
        chat_sep = ['[UNUSED_TOKEN_145]\n']
        suffix = ['[UNUSED_TOKEN_145]']
        system_prefix = ['<s>[UNUSED_TOKEN_146]system\n{{SYSTEM}}[UNUSED_TOKEN_145]\n']
        super().__init__(prefix, prompt, chat_sep, suffix, self.INTERNLM_XCOMPOSER_SYSTEM, system_prefix)
        self.version = version

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        dtype = self.model.dtype
        images = example.get('images') or []

        if self.version == 'v2.5':
            hd_num = 24
            if len(images) > 1:
                hd_num = 6
            hd_num = get_env_args('hd_num', int, hd_num)
            Image_transform = get_class_from_dynamic_module('ixc_utils.Image_transform', self.tokenizer.model_dir)
            images = [Image_transform(image, hd_num=hd_num) for image in images]
        elif self.version == 'v2-4khd':
            hd_num = 55
            hd_num = get_env_args('hd_num', int, hd_num)
            HD_transform = get_class_from_dynamic_module('ixc_utils.HD_transform', self.tokenizer.model_dir)
            images = [HD_transform(image, hd_num=hd_num) for image in images]
        images = [self.model.vis_processor(image).to(dtype) for image in images]
        inputs['_data'] = {'input_ids': inputs['input_ids'], 'labels': inputs['labels'], 'images': images}
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        input_ids = data['input_ids']
        labels = data['labels']
        images = data['images']
        if len(images) > 0:  # ignore <s>
            input_ids = input_ids[1:]
            if labels is not None:
                labels = labels[1:]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        input_ids.append(2)  # add dummy </s>
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()
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

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        if 'im_mask' in batch[0]:
            im_mask = [b['im_mask'][0] for b in batch]
            im_mask = self.pad_sequence(im_mask, 0, self.padding_side)
            res['im_mask'] = im_mask
        return res

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids


register_template(
    TemplateType.internlm_xcomposer2, InternLMXComposer2Template(version='v2'), use_model=True, lazy_tokenize=True)


class InternLMXComposer2_5Template(InternLMXComposer2Template):
    INTERNLM_XCOMPOSER_SYSTEM = (
        'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
        '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model '
        'that is developed by Shanghai AI Laboratory (上海人工智能实验室). '
        'It is designed to be helpful, honest, and harmless.\n'
        '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen '
        'by the user such as English and 中文.\n'
        '- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively '
        'based on the provided image.')


register_template(
    TemplateType.internlm_xcomposer2_5,
    InternLMXComposer2_5Template(version='v2.5'),
    use_model=True,
    lazy_tokenize=True)

register_template(
    TemplateType.internlm_xcomposer2_4khd,
    InternLMXComposer2_5Template(version='v2-4khd'),
    use_model=True,
    lazy_tokenize=True)
