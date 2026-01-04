# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Literal

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, findall
from .utils import TemplateMeta


class DotsOCRTemplate(Template):
    image_token_id = 151665
    placeholder_tokens = ['<|imgpad|>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image
        assert media_type == 'image'
        inputs.images[index] = fetch_image({'image': inputs.images[index]})
        if self.mode == 'lmdeploy':
            return ['<|img|>', [-100], '<|endofimg|>']
        else:
            return ['<|img|><|imgpad|><|endofimg|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)

        images = inputs.images
        media_token = self.image_token_id
        media_inputs = processor.image_processor(images=images, videos=None, return_tensors='pt', do_resize=False)
        media_grid_thw = media_inputs['image_grid_thw']
        idx_list = findall(input_ids, media_token)
        merge_length = processor.image_processor.merge_size**2

        def _get_new_tokens(i):
            token_len = (media_grid_thw[i].prod() // merge_length)
            return [media_token] * token_len

        input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list, _get_new_tokens)
        encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded


register_template(
    TemplateMeta(
        MLLMTemplateType.dots_ocr,
        prefix=[''],
        prompt=['<|user|>{{QUERY}}<|endofuser|><|assistant|>'],
        chat_sep=['<|endofassistant|>'],
        suffix=['<|endofassistant|>'],
        system_prefix=['<|system|>{{SYSTEM}}<|endofsystem|>\n'],
        template_cls=DotsOCRTemplate,
    ))
