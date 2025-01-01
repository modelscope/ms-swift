# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

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


class LlavaHfTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return ['<image>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        if images:
            image_processor = self.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.config.torch_dtype)
            encoded['pixel_values'] = image_inputs['pixel_values']
            if 'image_sizes' in image_inputs:
                encoded['image_sizes'] = image_inputs['image_sizes']
        return encoded


register_template(
    TemplateMeta(
        MLLMTemplateType.llava1_5_hf,
        prefix=['<s>'],
        prompt=['USER: {{QUERY}}\nASSISTANT:'],
        chat_sep=['</s>'],
        suffix=['</s>'],
        template_cls=LlavaHfTemplate,
    ))


class LlavaVideoHfTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<image>\n']
        assert media_type == 'video'
        media_file = inputs.videos[index]
        if media_file.rsplit('.', 1)[-1] in {'jpg', 'png'}:
            return ['<image>\n']
        else:
            inputs.videos[index] = load_video_llava(inputs.videos[index])
            return ['<video>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images or []
        videos = inputs.videos or []
        if len(videos) > 0:
            video_processor = self.processor.video_processor
            video_inputs = video_processor(videos, return_tensors='pt').to(self.config.torch_dtype)
            encoded['pixel_values_videos'] = video_inputs['pixel_values_videos']
        if len(images) > 0:
            image_processor = self.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.config.torch_dtype)
            encoded['pixel_values'] = image_inputs['pixel_values']
            encoded['image_sizes'] = image_inputs['image_sizes']
        return encoded


register_template(
    TemplateMeta(
        MLLMTemplateType.llava_next_video_hf,
        prefix=['{{SYSTEM}} '],
        prompt=['USER: {{QUERY}} ASSISTANT:'],
        chat_sep=[' '],
        suffix=[['eos_token_id']],
        template_cls=LlavaVideoHfTemplate,
        auto_add_bos=True,
    ))


class Llava1_6HfTemplate(LlavaHfTemplate):

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        for b in batch:
            pixel_values = b.get('pixel_values')
            if pixel_values is not None:
                b['pixel_values'] = pixel_values.squeeze(0)  # 5d -> 4d
        res = super()._data_collator(batch, padding_to=padding_to)
        return res


@dataclass
class LlavaMistralTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<s>[INST] '])
    prompt: Prompt = field(default_factory=lambda: ['{{QUERY}} [/INST]'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['</s>[INST] '])
    suffix: Prompt = field(default_factory=lambda: ['</s>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<<SYS>>\n{{system}}\n<</SYS>>\n\n'])


register_template(LlavaMistralTemplateMeta(MLLMTemplateType.llava1_6_mistral_hf, template_cls=Llava1_6HfTemplate))

register_template(
    TemplateMeta(
        MLLMTemplateType.llava1_6_vicuna_hf,
        prefix=['<s>'],
        prompt=['USER: {{QUERY}} ASSISTANT:'],
        chat_sep=['</s>'],
        suffix=['</s>'],
        default_system=('A chat between a curious human and an artificial intelligence assistant. '
                        "The assistant gives helpful, detailed, and polite answers to the human's questions."),
        system_prefix=['<s>{{SYSTEM}} '],
        template_cls=Llava1_6HfTemplate))


class LLava1_6YiHfTemplate(Llava1_6HfTemplate):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        if self.mode == 'vllm':
            return [[64000], '\n']
        else:
            return super().replace_tag(media_type, index, inputs)


register_template(ChatmlTemplateMeta(
    MLLMTemplateType.llava1_6_yi_hf,
    template_cls=LLava1_6YiHfTemplate,
))

register_template(Llama3TemplateMeta(
    MLLMTemplateType.llama3_llava_next_hf,
    template_cls=Llava1_6HfTemplate,
))

register_template(QwenTemplateMeta(MLLMTemplateType.llava_next_qwen_hf, template_cls=Llava1_6HfTemplate))


class LlavaOneVisionHfTemplate(Llava1_6HfTemplate):

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, 151646)  # <image>
        processor = self.processor
        if images:
            image_processor = processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.config.torch_dtype)
            height, width = image_inputs['pixel_values'][0].shape[-2:]
            added_tokens_len = 0
            for idx, pixel_v, image_size in zip(idx_list, image_inputs['pixel_values'], image_inputs['image_sizes']):
                orig_height, orig_width = image_size
                num_image_tokens = processor._get_number_of_features(orig_height, orig_width, height, width)
                input_ids = input_ids[:added_tokens_len
                                      + idx] + [151646] * num_image_tokens + input_ids[added_tokens_len + idx + 1:]
                if labels is not None:
                    labels = labels[:added_tokens_len + idx] + [-100] * num_image_tokens + labels[added_tokens_len + idx
                                                                                                  + 1:]
                added_tokens_len += num_image_tokens - 1
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
            encoded['pixel_values'] = image_inputs['pixel_values']
            if 'image_sizes' in image_inputs:
                encoded['image_sizes'] = image_inputs['image_sizes']
        return encoded


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.llava_onevision_hf,
        default_system=None,
        template_cls=LlavaOneVisionHfTemplate,
        placeholder_tokens=['<image>'],
    ))


class LlavaLlama3_1HfTemplate(LlavaHfTemplate):
    # DaozeZhang
    system = ('You are a helpful language and vision assistant. '
              'You are able to understand the visual content that the user provides, '
              'and assist the user with a variety of tasks using natural language.')

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if len(encoded['pixel_values'].shape) == 5:  # (1, num_patch, 3, H/W, W/H)
            encoded['pixel_values'] = torch.squeeze(encoded['pixel_values'], dim=0)  # (num_patch, 3, H/W, W/H)
        return encoded


register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.llava_llama3_1_hf,
        default_system=LlavaLlama3_1HfTemplate.system,
        template_cls=LlavaLlama3_1HfTemplate,
    ))


class LLavaLlama3HfTemplate(Template):
    # xtuner
    image_placeholder = ['<image>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        raw_image = inputs.images
        if raw_image:
            pixel_values = self.processor.image_processor(raw_image, return_tensors='pt')['pixel_values']
            encoded['pixel_values'] = pixel_values.to(self.config.torch_dtype)
        return encoded


register_template(Llama3TemplateMeta(
    MLLMTemplateType.llava_llama3_hf,
    template_cls=LLavaLlama3HfTemplate,
))


class LLavaTemplate(Template):
    skip_prompt = False
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images or []
        image_sizes = [x.size for x in images]
        from llava.mm_utils import process_images
        model = self.model.model
        if not hasattr(model, 'vision_tower'):
            model = model.model
        image_processor = model.vision_tower.image_processor
        if images:
            images_tensor = process_images(images, image_processor, model.config)
            encoded['images'] = images_tensor.to(model.dtype).squeeze(0)
            encoded['image_sizes'] = image_sizes
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = images
            res['image_sizes'] = sum([b['image_sizes'] for b in batch if 'image_sizes' in b], start=[])
        return res


register_template(LlavaMistralTemplateMeta(MLLMTemplateType.llava1_6_mistral, template_cls=LLavaTemplate))

register_template(ChatmlTemplateMeta(MLLMTemplateType.llava1_6_yi, template_cls=LLavaTemplate))

register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.llama3_llava_next,
        template_cls=LLavaTemplate,
        default_system=('You are a helpful language and vision assistant. '
                        'You are able to understand the visual content that the user provides, '
                        'and assist the user with a variety of tasks using natural language.'),
    ))

register_template(QwenTemplateMeta(MLLMTemplateType.llava_next_qwen, template_cls=LLavaTemplate))
