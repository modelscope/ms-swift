from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall


@dataclass
class HunYuanVLTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<｜hy_begin▁of▁sentence｜>'])
    prompt: Prompt = field(default_factory=lambda: ['{{QUERY}}<｜hy_User｜>'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<｜hy_Assistant｜><｜hy_begin▁of▁sentence｜>'])
    suffix: Prompt = field(default_factory=lambda: ['<｜hy_Assistant｜>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<｜hy_begin▁of▁sentence｜>{{SYSTEM}}<｜hy_place▁holder▁no▁3｜>'])


class HunYuanVLTemplate(Template):
    image_token_id = 120120
    placeholder_tokens = ['<｜hy_place▁holder▁no▁102｜>']

    # support_padding_free = True  # position_ids with batch_dim of 0 does not support padding_free

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        if self.mode == 'vllm':
            return ['<｜hy_place▁holder▁no▁100｜><｜hy_place▁holder▁no▁102｜><｜hy_place▁holder▁no▁101｜>']
        return [[-100]]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        idx_list = findall(input_ids, -100)
        processor = self.processor
        images = inputs.images
        if images:
            image_inputs = processor.image_processor(images=images, return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            merge_size = processor.image_processor.merge_size

            def _get_new_tokens(i):
                grid_h, grid_w = image_grid_thw[i][-2:]
                patch_h = grid_h // merge_size
                patch_w = grid_w // merge_size
                img_tokens: List[int] = [self.image_token_id] * (patch_h * (patch_w + 1) + 2)
                return img_tokens

            encoded['input_ids'], encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, _get_new_tokens)
            encoded['pixel_values'] = image_inputs['pixel_values']
            encoded['image_grid_thw'] = image_grid_thw

            input_ids = encoded['input_ids']
            position_ids = torch.arange(len(input_ids))
            position_ids_w = torch.arange(len(input_ids))
            position_ids_h = torch.arange(len(input_ids))
            position_ids_t = torch.arange(len(input_ids))
            image_tokens_cumsum = [0]
            for i in range(len(image_grid_thw)):
                grid_h, grid_w = image_grid_thw[i][-2:]
                patch_h = grid_h // merge_size
                patch_w = grid_w // merge_size
                num_image_tokens = patch_h * (patch_w + 1) + 2
                image_tokens_cumsum.append(image_tokens_cumsum[-1] + int(num_image_tokens))
                image_token_pos_indices = torch.where(torch.tensor(input_ids) == self.image_token_id)
                start_pos = image_token_pos_indices[0][image_tokens_cumsum[i]] + 1
                replace_num = (patch_w + 1) * patch_h
                position_ids_w[start_pos:start_pos + replace_num] = torch.tensor(
                    list(range(patch_w + 1)) * patch_h, dtype=torch.int64)
                patch_h_list = []
                for h in range(patch_h):
                    patch_h_list += [h] * (patch_w + 1)
                position_ids_h[start_pos:start_pos + replace_num] = torch.tensor(patch_h_list, dtype=torch.int64)
                position_ids_t[start_pos:start_pos + replace_num] = 0
            position_ids = torch.stack([position_ids, position_ids_w, position_ids_h, position_ids_t]).unsqueeze(0)
            encoded['position_ids'] = position_ids
            attention_mask = torch.tensor(input_ids).ne(processor.pad_id)
            encoded['attention_mask'] = attention_mask
        return encoded

    def _pad_3d_position_ids(self,
                             position_ids: List[torch.Tensor],
                             padding_value: float = 0.,
                             batch_dim: int = 1) -> torch.Tensor:
        batch_dim = 0
        return super()._pad_3d_position_ids(position_ids, padding_value, batch_dim)


register_template(HunYuanVLTemplateMeta(MLLMTemplateType.hunyuan_ocr, template_cls=HunYuanVLTemplate))
