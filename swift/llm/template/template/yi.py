# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional

import torch

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta

register_template(ChatmlTemplateMeta(
    LLMTemplateType.yi_coder,
    default_system=DEFAULT_SYSTEM,
))

yi_vl_default_system = (
    'This is a chat between an inquisitive human and an AI assistant. Assume the role of the AI assistant. '
    "Read all the images carefully, and respond to the human's questions with informative, "
    'helpful, detailed and polite answers. '
    '这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。'
    '仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。')


class YiVLTemplate(Template):
    image_placeholder = [[-200], '\n']
    use_model = True

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        model = self.model
        from llava.mm_utils import expand2square
        if not hasattr(model, 'vision_tower'):
            model = model.model
        image_processor = model.vision_tower.image_processor
        images = inputs.images or []
        for i, image in enumerate(images):
            background_color = tuple(int(x * 255) for x in image_processor.image_mean)
            image = expand2square(image, background_color)
            images[i] = image
        if images:
            image_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values']
            encoded['images'] = image_tensor.to(model.dtype)
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.yi_vl,
        prefix=[],
        prompt=[[8308], ' Human: {{QUERY}}\n', [8308], ' Assistant:'],
        chat_sep=['\n'],
        suffix=['\n', [8308]],
        default_system=yi_vl_default_system,
        template_cls=YiVLTemplate,
        system_prefix=['{{SYSTEM}}\n\n']))
