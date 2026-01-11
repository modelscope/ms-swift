# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Literal, Optional

import torch

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, findall


class MolmoTemplate(Template):
    placeholder_tokens = ['<im_patch>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        return []

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        # image
        images_inputs = self.processor.process(images=inputs.images or None, text='')
        images_input_ids = images_inputs.pop('input_ids').tolist()
        user_token = self._tokenize(' User')
        assert len(user_token) == 1
        idx = findall(images_input_ids, user_token[0])
        assert len(idx) == 1
        labels = encoded['labels']
        encoded['input_ids'] = images_input_ids[:idx[0]] + encoded['input_ids']
        if labels:
            encoded['labels'] = [-100] * idx[0] + labels
        if 'images' in images_inputs:
            images_inputs['images'] = images_inputs['images'].to(self.model_info.torch_dtype)
        encoded.update(images_inputs)
        return encoded

    def generate(self, model, **kwargs):
        kwargs.pop('attention_mask', None)
        generation_config = kwargs.pop('generation_config')
        batch = {
            k: kwargs.pop(k, None)
            for k in ['input_ids', 'attention_mask', 'images', 'image_input_idx', 'image_masks']
        }
        return model.generate_from_batch(batch, generation_config, **kwargs)

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        # prepare batchfy inputs
        keys = ['images', 'image_input_idx', 'image_masks']
        images_res = self.fetch_inputs(batch, keys)
        for key in keys:
            val = images_res.get(key)
            if val:
                images_res[key] = torch.stack(val)
        res.update(images_res)
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.molmo,
        prefix=[],
        prompt=[' User: {{QUERY}} Assistant:'],
        chat_sep=None,
        suffix=['<|endoftext|>'],
        template_cls=MolmoTemplate,
    ))
