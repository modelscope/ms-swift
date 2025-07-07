# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.llm import to_device
from swift.utils import is_deepspeed_enabled
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Word, findall
from .qwen import Qwen2VLTemplate
from .utils import ChatmlTemplateMeta


@dataclass
class KeyeTemplateMeta(ChatmlTemplateMeta):
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])


class KeyeVLTemplate(Template):
    image_token_id = 151655
    video_token_id = 151656
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from keye_vl_utils import fetch_image, fetch_video
        # from qwen_vl_utils import fetch_image, fetch_video
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            if getattr(self, 'mode', None) == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            video = inputs.videos[index]
            if os.path.isdir(video):
                video = [os.path.join(video, fname) for fname in os.listdir(video)]
            video = fetch_video({'video': video})
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            return ['<|vision_start|><|video_pad|><|vision_end|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from keye_vl_utils import vision_process
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        images = inputs.images
        videos = inputs.videos
        for media_type in ['images', 'videos']:
            if locals()[media_type]:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(
                        images=images, videos=None, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    if hasattr(processor, 'video_processor'):
                        processor_func = processor.video_processor
                    else:
                        processor_func = processor.image_processor
                    media_inputs = processor_func(images=None, videos=videos, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                    media_inputs['second_per_grid_ts'] = [
                        processor.image_processor.temporal_patch_size / vision_process.FPS
                    ] * len(media_grid_thw)
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    return [media_token] * token_len

                input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, _get_new_tokens)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        second_per_grid_ts = self.gather_list(batch, 'second_per_grid_ts')
        if second_per_grid_ts:
            res['second_per_grid_ts'] = second_per_grid_ts
        for media_type in ['image', 'video']:
            grid_thw = self.concat_tensor(batch, f'{media_type}_grid_thw', 0)
            if grid_thw is not None:
                res[f'{media_type}_grid_thw'] = grid_thw
        return res


# Register the Keye VL template
register_template(KeyeTemplateMeta(MLLMTemplateType.keye_vl, template_cls=KeyeVLTemplate))
