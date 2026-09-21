# Copyright (c) ModelScope Contributors. All rights reserved.
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


class MonkeyOCRv2Template(Template):
    use_model = False
    support_padding_free = False
    norm_bbox = 'none'
    placeholder_tokens = ['<|IMAGE|>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image', 'MonkeyOCRv2 only supports image input'
        from qwen_vl_utils import fetch_image
        inputs.images[index] = fetch_image({'image': inputs.images[index]})
        return ['<|IMAGE|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)

        images = inputs.images
        if images:
            image_processor = getattr(processor, 'image_processor', None) or processor
            media_inputs = image_processor(images=images, return_tensors='pt')
            media_grid_thw = media_inputs.get('image_grid_thw')

            image_token_id = self._tokenize('<|IMAGE|>')
            idx_list = findall(input_ids, image_token_id)
            if idx_list and media_grid_thw is not None:
                merge_size = getattr(image_processor, 'merge_size', 2)

                def _get_new_tokens(i):
                    token_len = media_grid_thw[i].prod() // (merge_size**2)
                    return [image_token_id] * token_len

                input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                    _get_new_tokens)

            encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded


register_template(
    TemplateMeta(
        MLLMTemplateType.monkeyocrv2,
        prefix=[''],
        prompt=['{{QUERY}}'],
        chat_sep=['\n'],
        suffix=[''],
        system_prefix=None,
        default_system=None,
        template_cls=MonkeyOCRv2Template,
    ))
