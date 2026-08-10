# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from dataclasses import dataclass, field
from qwen_vl_utils import fetch_image, fetch_video
from typing import Any, Dict, List, Optional

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..template_meta import TemplateMeta
from ..utils import Context, Prompt, Word, findall
from .utils import ChatmlTemplateMeta


@dataclass
class MiMoV2TemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = 'You are MiMo, a helpful AI assistant engineered by Xiaomi.'
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ['<|im_start|>user\n{{QUERY}}<|im_end|><|im_start|>assistant\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|im_start|>system\n{{SYSTEM}}<|im_end|>'])


class MiMoV2Template(Template):
    image_token_id = 151655
    video_token_id = 151656
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    norm_bbox = 'none'

    def replace_tag(self, media_type, index, inputs: StdTemplateInputs) -> List[Context]:
        assert media_type in {'image', 'video'}
        kwargs = {'image_patch_size': self.processor.image_processor.patch_size}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index], **inputs.chat_template_kwargs}, **kwargs)
            if self.mode == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            video = inputs.videos[index]
            video_inputs = {'video': video, **inputs.chat_template_kwargs}
            if isinstance(video, list):
                from qwen_vl_utils import vision_process
                video_inputs['sample_fps'] = vision_process.FPS
            video, _ = fetch_video(video_inputs, return_video_sample_fps=True)
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            return ['<|vision_start|><|video_pad|><|vision_end|>']

    def _encode_truncated(self, inputs: StdTemplateInputs):
        encoded = super()._encode_truncated(inputs)
        if self.mode == 'sglang':
            batched = encoded if isinstance(encoded, list) else [encoded]
            for item in batched:
                for old, new in [('images', 'image_data'), ('audios', 'audio_data'), ('videos', 'video_data')]:
                    if old in item:
                        item[new] = item.pop(old)
                for key in ['labels', 'loss_scale', 'channel']:
                    item.pop(key, None)
        return encoded

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)

        for media_type in ['images', 'videos']:
            mm_data = getattr(inputs, media_type)
            if not mm_data:
                continue
            if media_type == 'images':
                media_token = self.image_token_id
                media_inputs = processor.image_processor(images=mm_data, return_tensors='pt', do_resize=False)
                media_grid_thw = media_inputs['image_grid_thw']
            else:
                if hasattr(processor, 'video_processor'):
                    processor_func = processor.video_processor
                else:
                    processor_func = processor.image_processor
                media_inputs = processor_func(videos=mm_data, return_tensors='pt', do_resize=False)
                media_grid_thw = media_inputs['video_grid_thw']
                media_token = self.video_token_id

            idx_list = findall(input_ids, media_token)
            merge_length = processor.image_processor.merge_size**2

            def _get_new_tokens(i):
                return [media_token] * (media_grid_thw[i].prod() // merge_length)

            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                _get_new_tokens)
            encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        if 'pixel_values_videos' in res:
            res['video_pixel_values'] = res.pop('pixel_values_videos')
        return res


register_template(MiMoV2TemplateMeta(MLLMTemplateType.mimo_v2, template_cls=MiMoV2Template))
