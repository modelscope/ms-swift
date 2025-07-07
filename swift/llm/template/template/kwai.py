# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import List, Literal

import torch

from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Word
from .qwen import Qwen2VLTemplate
from .utils import ChatmlTemplateMeta


@dataclass
class KeyeTemplateMeta(ChatmlTemplateMeta):
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])


class KeyeVLTemplate(Qwen2VLTemplate):
    """
    Keye-VL template inheriting from Qwen2VLTemplate

    Keye-VL is based on Qwen3-8B language model with SigLIP vision encoder.
    It uses the same vision token format as Qwen2VL:
    - Images: <|vision_start|><|image_pad|><|vision_end|>
    - Videos: <|vision_start|><|video_pad|><|vision_end|>
    - Supports 3D RoPE for unified text/image/video processing

    Key difference: Uses keye_vl_utils.process_vision_info for vision processing
    """

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """Override replace_tag to use keye_vl_utils instead of qwen_vl_utils"""
        from keye_vl_utils import fetch_image, fetch_video

        assert media_type in {'image', 'video'}
        if media_type == 'image':
            # Use keye_vl_utils.fetch_image
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            if getattr(self, 'mode', None) == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            # Use keye_vl_utils.fetch_video
            video = inputs.videos[index]
            if os.path.isdir(video):
                video = [os.path.join(video, fname) for fname in os.listdir(video)]
            video = fetch_video({'video': video})
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            return ['<|vision_start|><|video_pad|><|vision_end|>']


# Register the Keye VL template
register_template(KeyeTemplateMeta(MLLMTemplateType.keye_vl, template_cls=KeyeVLTemplate))
