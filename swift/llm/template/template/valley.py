# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import io
from PIL import Image
import torch
import transformers
from packaging import version

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall
from ..vision_utils import load_batch, load_video_llava
from .llama import Llama3TemplateMeta
from .qwen import QwenTemplateMeta
from .utils import ChatmlTemplateMeta

class ValleyTemplate(Template):
    skip_prompt = False
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def preprocess_images(self, image_binary_list) -> torch.FloatTensor:
        byte2image = lambda byte_data: Image.open(io.BytesIO(byte_data))
        images = []
        for binary in image_binary_list:
            if isinstance(binary, Image.Image):
                images.append(binary.convert("RGB") )
            elif isinstance(binary, bytes):
                images.append(byte2image(binary))
            else:
                raise ValueError("unsupported type")
        video_pad = []
        for img in images:
            image = self.image_processor(img, return_tensors="pt")["pixel_values"][0]
            video_pad.append(image)

        video_pad = [self.black_img] if len(video_pad) == 0 else video_pad

        if not self.model.config.anyres:
            video = torch.stack(video_pad, dim=0)
        else:
            video = [torch.stack(img, dim=0) for img in video_pad]
        return video

    def process_images(self, inputs, images_binary):
        import re
        from qwen_vl_utils import fetch_image

        text = inputs.messages[-1].content[0].text
        video_images_tensor = self.preprocess_images(images_binary)
        img_length = len(video_images_tensor)
        video_images_tensor = [video_images_tensor]
        if img_length:
            images = [[item.to(self.device).half() for item in img] for img in video_images_tensor]

        messages_qwen = []
        image_list = []
        if isinstance(images_binary[0], Image.Image):
            images_pil = [img.convert("RGB") for img in images_binary]
        elif isinstance(images_binary[0], bytes):
            images_pil = [Image.open(io.BytesIO(img)).convert("RGB") for img in images_binary]
        image_sizes = [[x.size for x in images_pil]]
        for image_file in images_pil:
            image = fetch_image({"image": image_file})
            image_list.append(image)
        messages_qwen.append({"role": "user", "content": [{"type": "text", "text": text}]})
        messages_qwen.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})
        text = self.qwen2vl_processor.apply_chat_template(messages_qwen[:-1], tokenize=False, add_generation_prompt=True)
        text_segs = re.split("<image>", text)
        text = "<|vision_start|><|image_pad|><|vision_end|>".join(text_segs[: len(image_list) + 1]) + "".join(
            text_segs[len(image_list) + 1 :]
        )
        data_dict_qwen2vl = self.qwen2vl_processor(text=[text], images=image_list, padding=True, return_tensors="pt")
        results = {}

        results["images"] = images
        results["image_sizes"] = image_sizes
        results["pixel_values"] = data_dict_qwen2vl["pixel_values"].to(self.device)
        results["image_grid_thw"] = data_dict_qwen2vl["image_grid_thw"].to(self.device)
        return results

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images or []
        if images:
            results = self.process_images(inputs, images)
            encoded['images'] = results['images']
            encoded['image_sizes'] = results['image_sizes']
            encoded['pixel_values'] = results['pixel_values']
            encoded['image_grid_thw'] = results['image_grid_thw']
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = images
            res['image_sizes'] = sum([b['image_sizes'] for b in batch if 'image_sizes' in b], start=[])
        return res

register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.valley,
        template_cls=ValleyTemplate,
        default_system=("You are Valley, a large language and vision assistant trained by ByteDance."
           "You are able to understand the visual content or video that the user provides, \
and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail."),
    ))