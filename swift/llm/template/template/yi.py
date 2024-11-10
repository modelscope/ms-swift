from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Context, findall, gather_list
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

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        inputs.pop('loss_scale', None)
        from llava.mm_utils import expand2square
        model = self.model.model
        if not hasattr(model, 'vision_tower'):
            model = model.model
        image_processor = model.vision_tower.image_processor
        images = example.get('images') or []
        for i, image in enumerate(images):
            background_color = tuple(int(x * 255) for x in image_processor.image_mean)
            image = expand2square(image, background_color)
            images[i] = image
        if images:
            image_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values']
            inputs['images'] = image_tensor.to(model.dtype)
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        has_images = [(b == -200).sum() for b in res['input_ids']]
        assert all([
            h > 0 for h in has_images
        ]) or not any([h > 0
                       for h in has_images]), 'YIVL does not support mix-batch nlp dataset and multi-modal dataset'
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.yi_vl,
        prefix=[],
        prompt=[[8308], 'Human: {{QUERY}}\n', [8308], 'Assistant:'],
        chat_sep=['\n'],
        suffix=['\n', [8308]],
        default_system=yi_vl_default_system,
        template_cls=YiVLTemplate,
        system_prefix=['{{SYSTEM}}\n\n']))
