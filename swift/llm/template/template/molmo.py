# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional

import torch

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import findall


class MolmoTemplate(Template):
    system = None
    use_model = True
    image_placeholder = ['<|image|>']
    DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
    DEFAULT_IM_START_TOKEN = '<im_start>'
    DEFAULT_IM_END_TOKEN = '<im_end>'
    DEFAULT_IM_COL_TOKEN = '<im_col>'

    def __init__(self, *args, **kwargs):
        Template.__init__(self, *args, **kwargs)
        self.processor_kwargs = {
            'images_kwargs': {
                'max_crops': 12,
                'overlap_margins': [4, 4],
                'base_image_input_size': [336, 336],
                'image_token_length_w': 12,
                'image_token_length_h': 12,
                'image_patch_size': 14,
                'image_padding_mask': True,
            },
            'text_kwargs': {
                'style': 'long_caption',
                'system_prompt': 'none',
                'message_format': 'role',
                'always_start_with_space': True,
                'sequence_length': 1536,
                'padding': False,
            }
        }

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if len(encoded) == 0:
            return encoded
        # image
        raw_image = inputs.images
        res = {}
        labels = encoded['labels']
        if raw_image:
            image_id = self.tokenizer.convert_tokens_to_ids(self.image_placeholder)
            idx_list = findall(encoded['input_ids'], image_id)
            res = self._process_images(raw_image, encoded['input_ids'], idx_list, labels)
            import numpy as np
            if 'image_input_idx' in res:
                # Shift patch mapping up by one since we added BOS
                image_input_idx = res['image_input_idx']
                res['image_input_idx'] = np.where(image_input_idx < 0, image_input_idx, image_input_idx + 1)
            encoded['input_ids'] = res.pop('input_ids').tolist()
            if labels:
                encoded['labels'] = [-100] + res.pop('labels')  # add one label for BOS

            for k, v in res.items():
                res[k] = torch.from_numpy(v).unsqueeze(0)
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        encoded['input_ids'] = [bos] + encoded['input_ids']
        res.update({'input_ids': encoded['input_ids']})
        # prepare meta inputs
        encoded.update(self.prepare_meta_inputs(res))

        return encoded

    def _process_images(self, images: List, tokens: List, idx_list: List = None, labels: List = None) -> torch.Tensor:
        from PIL.Image import Image
        import numpy as np
        if images is not None:
            image_arrays = []
            for image in images:
                if isinstance(image, Image):
                    image = image.convert('RGB')
                    image_arrays.append(np.array(image))
                else:
                    assert len(image.shape) == 3 and image.shape[-1] == 3
                    image_arrays.append(image.astype(np.uint8))
            images = image_arrays
            # For now only support inserting images at the start
        if idx_list is None:
            idx_list = [-1] * len(images)
        image_patch_token_id = self.processor.special_token_ids[self.DEFAULT_IMAGE_PATCH_TOKEN]
        image_col_token_id = self.processor.special_token_ids[self.DEFAULT_IM_COL_TOKEN]
        image_start_token_id = self.processor.special_token_ids[self.DEFAULT_IM_START_TOKEN]
        image_end_token_id = self.processor.special_token_ids[self.DEFAULT_IM_END_TOKEN]
        sequence_length = self.processor_kwargs['text_kwargs']['sequence_length']
        res = self.processor.image_processor.multimodal_preprocess(
            images=images,
            image_idx=idx_list,
            tokens=np.asarray(tokens).astype(np.int32),
            sequence_length=sequence_length,
            image_patch_token_id=image_patch_token_id,
            image_col_token_id=image_col_token_id,
            image_start_token_id=image_start_token_id,
            image_end_token_id=image_end_token_id,
            **self.processor_kwargs['images_kwargs'])
        if labels is not None:
            new_labels = []
            cur_idx = 0
            for input_id in res['input_ids']:
                if input_id in (image_start_token_id, image_end_token_id, image_col_token_id, image_patch_token_id):
                    new_labels.append(-100)
                    if tokens[cur_idx] == self.tokenizer.convert_tokens_to_ids(self.image_placeholder)[0]:
                        cur_idx += 1
                else:
                    new_labels.append(labels[cur_idx])
                    cur_idx += 1
            res['labels'] = new_labels
        return res

    def prepare_meta_inputs(self, data: Any) -> Dict[str, Any]:

        # prepare batch inputs
        input_ids = torch.tensor(data['input_ids']).unsqueeze(0)
        batch_size, seq_len = input_ids.shape
        attention_mask = None
        mask_len = seq_len
        max_new_tokens = None
        if not self.is_training:
            generation_config = self.model.generation_config
            max_new_tokens = generation_config.max_new_tokens
            if not max_new_tokens:
                max_new_tokens = 0
            mask_len = mask_len + max_new_tokens if self.model.config.use_position_ids else mask_len
        position_ids: Optional[torch.Tensor] = None
        append_last_valid_logits: Optional[torch.Tensor] = None
        if self.model.config.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1
            position_ids = torch.clamp(torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1, min=0)
            append_last_valid_logits = attention_mask.long().sum(dim=-1) - 1
            if max_new_tokens:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((batch_size, max_new_tokens))],
                    dim=1,
                )
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, mask_len)
        if self.is_training:
            # no batch_size before data_collator
            attention_mask = attention_mask.squeeze(0)
            position_ids = position_ids.squeeze(0)
        data.update({
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'append_last_valid_logits': append_last_valid_logits,
        })
        if 'images' in data:
            data['images'] = data['images'].to(self.model.dtype)
        return data

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to=padding_to)
        # prepare batchfy inputs
        keys = ['images', 'image_input_idx', 'image_masks', 'append_last_valid_logits']
        for key in keys:
            batch_input = [b[key] for b in batch if b.get(key) is not None]
            res[key] = torch.concat(batch_input)

        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.molmo,
        prefix=[],
        prompt=[' User: {{QUERY}} Assistant:'],
        chat_sep=['<|endoftext|>'],
        suffix=['<|endoftext|>'],
        template_cls=MolmoTemplate,
        placeholder_tokens=['<|image|>'],
    ))
