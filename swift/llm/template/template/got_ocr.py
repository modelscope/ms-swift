# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Literal, Optional, Tuple

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Context, gather_list
from .qwen import QwenTemplateMeta


class GOTImageEvalProcessor:

    def __init__(self, image_size=384, mean=None, std=None):
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            self.normalize,
        ])

    def __call__(self, item):
        return self.transform(item)


class GOT_OCR2Template(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        # OCR:
        # OCR with format:
        assert media_type == 'image'
        return ['<img>' + '<imgpad>' * 256 + '</img>\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, tokenizer_kwargs = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example['images']
        image_processor_high = GOTImageEvalProcessor(image_size=1024)
        for i, image in enumerate(images):
            images[i] = image_processor_high(image)[None].to(self.model.dtype)
        if images:
            inputs['images'] = images
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = gather_list(batch, 'images')
        if images:
            res['images'] = images
        return res


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.got_ocr2,
        default_system='        You should follow the instructions carefully and explain your answers in detail.',
        template_cls=GOT_OCR2Template,
    ))
