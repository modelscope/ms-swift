# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import torch
from PIL import Image
from typing import Any, Dict, List, Literal, Optional, Tuple

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, findall
from .utils import ChatmlTemplateMeta


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


class Molmo2Template(Template):
    """Molmo2 template for image and video understanding.

    Uses ChatML format with BOS auto-insertion.
    Media placeholders (<|image|>, <|video|>) are expanded via _extend_tokens.
    Video loading/sampling is delegated entirely to processor.video_processor.
    """

    use_model = True

    placeholder_tokens = [
        '<|image|>',
        '<|video|>',
        '<im_patch>',
        '<frame_start>',
        '<frame_end>',
    ]

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<|image|>']
        if media_type == 'video':
            return ['<|video|>']
        return []

    def _prepare_mm_inputs(self, inputs: StdTemplateInputs) -> Tuple[Dict[str, Any], List[List[int]], List[List[int]]]:
        media_inputs: Dict[str, Any] = {}
        image_expansions: List[List[int]] = []
        video_expansions: List[List[int]] = []
        tokenizer = self.tokenizer

        if inputs.images:
            images = [img if isinstance(img, Image.Image) else Image.open(img).convert('RGB')
                      for img in inputs.images]
            image_inputs = self.processor.image_processor(images=images, return_tensors='pt')
            for image_grid in image_inputs['image_grids']:
                image_tokens = self.processor.get_image_tokens(image_grid.cpu().numpy())
                image_expansions.append(tokenizer.encode(''.join(image_tokens), add_special_tokens=False))
            media_inputs.update(image_inputs)

        if inputs.videos:
            if len(inputs.videos) != 1:
                raise ValueError('Molmo2 currently only supports single-video samples.')
            video_inputs = self.processor.video_processor(
                videos=inputs.videos,
                return_tensors='pt',
                return_metadata=True,
            )
            video_metadata = video_inputs.pop('video_metadata')
            for video_grid, metadata in zip(video_inputs['video_grids'], video_metadata):
                video_string = self.processor.get_video_string(
                    video_grid.cpu().numpy(),
                    np.asarray(metadata.timestamps, dtype=np.float32),
                )
                video_expansions.append(tokenizer.encode(video_string, add_special_tokens=False))
            media_inputs.update(video_inputs)

        return media_inputs, image_expansions, video_expansions

    def _build_token_type_ids(self, input_ids: List[int]) -> List[int]:
        image_token_ids = {int(token_id) for token_id in getattr(self.processor, 'image_token_ids', [])}
        return [1 if token_id in image_token_ids else 0 for token_id in input_ids]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        media_inputs, image_expansions, video_expansions = self._prepare_mm_inputs(inputs)

        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale')

        # Expand image placeholders
        image_placeholder = self.tokenizer.encode('<|image|>', add_special_tokens=False)
        idx_list = findall(input_ids, image_placeholder)
        if idx_list:
            input_ids, labels, loss_scale = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, lambda i: image_expansions[i])

        # Expand video placeholders
        video_placeholder = self.tokenizer.encode('<|video|>', add_special_tokens=False)
        idx_list = findall(input_ids, video_placeholder)
        if idx_list:
            input_ids, labels, loss_scale = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, lambda i: video_expansions[i])

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        encoded['token_type_ids'] = self._build_token_type_ids(input_ids)
        encoded.update(media_inputs)
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        for key in ['image_grids', 'video_grids', 'image_token_pooling', 'video_token_pooling', 'image_num_crops']:
            value = self.concat_tensor(batch, key, 0)
            if value is not None:
                res[key] = value
        video_metadata = self.gather_list(batch, 'video_metadata')
        if video_metadata:
            res['video_metadata'] = video_metadata
        return res


register_template(
    ChatmlTemplateMeta(
        MLLMTemplateType.molmo2,
        template_cls=Molmo2Template,
    ))
