# Copyright (c) Alibaba, Inc. and its affiliates.
import io
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import torch
from PIL import Image

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context
from .utils import ChatmlTemplateMeta


@dataclass
class ValleyTemplateMeta(ChatmlTemplateMeta):
    auto_add_bos: bool = False
    default_system: Optional[str] = ('You are Valley, a large language and vision assistant trained by ByteDance.'
                                     'You are able to understand the visual content or video that the user provides,'
                                     ' and assist the user with a variety of tasks using natural language.'
                                     'Follow the instructions carefully and explain your answers in detail.')


class ValleyTemplate(Template):
    skip_prompt = True
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        # assert media_type == 'image'
        if media_type == 'video':
            from ..vision_utils import load_video_valley
            return self.replace_video2image(load_video_valley, inputs, lambda i: [[151665, -200, 151666]])
        return [[151665, -200, 151666]]

    def preprocess_images(self, image_binary_list):
        from valley_eagle.util.mm_utils import process_anyres_image

        def byte2image(byte_data):
            return Image.open(io.BytesIO(byte_data))

        images = []
        for binary in image_binary_list:
            if isinstance(binary, Image.Image):
                images.append(binary.convert('RGB'))
            elif isinstance(binary, bytes):
                images.append(byte2image(binary))
            else:
                raise ValueError('unsupported type')
        video_pad = []
        for img in images:
            if self.model.config.anyres:
                image = process_anyres_image(img, self.tokenizer.image_processor, self.model.config.grid_pinpoints)
            else:
                image = self.tokenizer.image_processor(img, return_tensors='pt')['pixel_values'][0]
            video_pad.append(image)

        if not self.model.config.anyres:
            video = torch.stack(video_pad, dim=0)
        else:
            video = [torch.stack(img, dim=0) for img in video_pad]
        return video

    def process_images(self, inputs, images_binary):
        import re
        from qwen_vl_utils import fetch_image

        if inputs.messages[-1]['role'] == 'user':
            text = inputs.messages[-1]['content']
        elif len(inputs.messages) > 1 and inputs.messages[-2]['role'] == 'user':
            text = inputs.messages[-2]['content']
        else:
            text = ''
        video_images_tensor = self.preprocess_images(images_binary)
        img_length = len(video_images_tensor)
        video_images_tensor = [video_images_tensor]
        if img_length:
            images = [[item.to(self.model.dtype) for item in img] for img in video_images_tensor]

        messages_qwen = []
        image_list = []
        if isinstance(images_binary[0], Image.Image):
            images_pil = [img.convert('RGB') for img in images_binary]
        elif isinstance(images_binary[0], bytes):
            images_pil = [Image.open(io.BytesIO(img)).convert('RGB') for img in images_binary]
        image_sizes = torch.tensor([[x.size for x in images_pil]])
        for image_file in images_pil:
            image = fetch_image({'image': image_file})
            image_list.append(image)
        messages_qwen.append({'role': 'user', 'content': [{'type': 'text', 'text': text}]})
        messages_qwen.append({'role': 'assistant', 'content': [{'type': 'text', 'text': ''}]})
        text = self.tokenizer.qwen2vl_processor.apply_chat_template(
            messages_qwen[:-1], tokenize=False, add_generation_prompt=True)
        text_segs = re.split('<image>', text)
        text = '<|vision_start|><|image_pad|><|vision_end|>'.join(text_segs[:len(image_list) + 1]) + ''.join(
            text_segs[len(image_list) + 1:])
        data_dict_qwen2vl = self.tokenizer.qwen2vl_processor(
            text=[text], images=image_list, padding=True, return_tensors='pt')
        results = {}

        results['images'] = images
        results['image_sizes'] = image_sizes
        results['pixel_values'] = data_dict_qwen2vl['pixel_values']
        results['image_grid_thw'] = data_dict_qwen2vl['image_grid_thw']
        return results

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images or []
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        if images:
            results = self.process_images(inputs, images)
            encoded['images'] = results['images']
            encoded['image_sizes'] = results['image_sizes']
            encoded['pixel_values'] = results['pixel_values']
            encoded['image_grid_thw'] = results['image_grid_thw']
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        if 'images' in batch[0]:
            res['images'] = sum([b['images'] for b in batch if 'images' in b], start=[])
            res['image_sizes'] = torch.concat([b['image_sizes'] for b in batch if 'image_sizes' in b], dim=0)
            for media_type in ['image', 'video']:
                grid_thw = [b[f'{media_type}_grid_thw'] for b in batch if b.get(f'{media_type}_grid_thw') is not None]
                if grid_thw:
                    res[f'{media_type}_grid_thw'] = torch.concat(grid_thw)
        return res


register_template(ValleyTemplateMeta(
    MLLMTemplateType.valley,
    template_cls=ValleyTemplate,
))
