# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Literal, Optional

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context
from ..vision_utils import load_file
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
    placeholder_tokens = ['<imgpad>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        # 'OCR: '
        # 'OCR with format: '
        assert media_type == 'image'
        return ['<img>' + '<imgpad>' * 256 + '</img>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        image_processor_high = GOTImageEvalProcessor(image_size=1024)
        for i, image in enumerate(images):
            images[i] = image_processor_high(image)[None].to(self.model_info.torch_dtype)
        if images:
            encoded['images'] = images
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        images = self.gather_list(batch, 'images')
        if images:
            res['images'] = images
        return res


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.got_ocr2,
        default_system='        You should follow the instructions carefully and explain your answers in detail.',
        template_cls=GOT_OCR2Template,
    ))


class GOT_OCR2HfTemplate(Template):
    placeholder_tokens = ['<imgpad>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        # 'OCR: '
        # 'OCR with format: '
        assert media_type == 'image'
        return ['<img>' + '<imgpad>' * 256 + '</img>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:  # 暂时照抄上面
        encoded = super()._encode(inputs)
        images = inputs.images
        if images:
            encoded['images'] = images
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        images = self.gather_list(batch, 'images')
        _inputs = self.processor(images, return_tensors='pt')
        _inputs.pop('input_ids')  # this does not contain the response, so cannot be used when training
        _inputs.pop('attention_mask')  # this does not contain the response, so cannot be used when training

        res.update(_inputs.data)
        return res


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.got_ocr2_hf,
        default_system='        You should follow the instructions carefully and explain your answers in detail.',
        template_cls=GOT_OCR2HfTemplate,
    ))


class StepAudioTemplate(Template):
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio', f'media_type: {media_type}'
        from utils import load_audio
        audio_wav, sr = load_audio(load_file(inputs.audios[index]))
        audio_tokens = self.model.encoder(audio_wav, sr)
        return audio_tokens


register_template(
    TemplateMeta(
        MLLMTemplateType.step_audio,
        template_cls=StepAudioTemplate,
        prefix=['<s>'],
        prompt=['<|BOT|>human\n{{QUERY}}<|EOT|><|BOT|>assistant\n'],
        system_prefix=['<s><|BOT|>system\n{{SYSTEM}}<|EOT|>'],
        chat_sep=['<|EOT|>'],
        suffix=['<|EOT|>'],
    ))
