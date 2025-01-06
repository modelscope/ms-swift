# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Any, Dict, List, Literal

import torch
from torch import nn

from swift.utils import get_env_args, is_deepspeed_enabled
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, findall
from ..vision_utils import load_video_internvl, transform_image
from .microsoft import Phi3TemplateMeta
from .utils import ChatmlTemplateMeta


class InternvlTemplate(Template):
    skip_prompt = False
    num_image_token = 256

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if self.mode == 'vllm':
            image_context = ['<img><image></img>\n']
        else:
            image_context = ['<img>', [-100], '</img>\n']
        return image_context

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        idx_list = findall(input_ids, -100)
        pixel_values = None
        images = inputs.images
        if images:
            labels = encoded.get('labels')
            input_size = get_env_args('input_size', int, 448)
            max_num = get_env_args('max_num', int, 12)
            pixel_values_images = [transform_image(image, input_size, max_num) for image in images]
            pixel_values = torch.cat(pixel_values_images, dim=0).to(self.model_info.torch_dtype)
            image_bs = pixel_values.shape[0]

            idx, idx2 = idx_list[0], idx_list[-1]  # remove [-100, -100]
            img_tokens: List[int] = self.processor.encode(
                '<IMG_CONTEXT>', add_special_tokens=False) * self.num_image_token * image_bs
            input_ids = input_ids[:idx] + img_tokens + input_ids[idx2 + 1:]
            if labels is not None:
                labels = labels[:idx] + [-100] * len(img_tokens) + labels[idx2 + 1:]
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
        encoded['pixel_values'] = pixel_values
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        embedding = model.get_input_embeddings()
        device = embedding.weight.device
        input_ids = inputs['input_ids']
        inputs_embeds = embedding(input_ids).to(device=device)
        pixel_values = inputs.get('pixel_values')
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device)
            vit_embeds = model.extract_feature(pixel_values).to(device=device)
            selected = (input_ids == self.processor.encode('<IMG_CONTEXT>', add_special_tokens=False)[0])
            inputs_embeds[selected] = vit_embeds.reshape(-1, vit_embeds.shape[-1])
        elif is_deepspeed_enabled():
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), device=device, dtype=inputs_embeds.dtype)
            vit_embeds = model.extract_feature(dummy_pixel_values).to(device=device)
            inputs_embeds += vit_embeds.mean() * 0.
        return {'inputs_embeds': inputs_embeds}


register_template(
    ChatmlTemplateMeta(
        MLLMTemplateType.internvl,
        default_system='You are an AI assistant whose name is InternLM (书生·浦语).',
        template_cls=InternvlTemplate,
        placeholder_tokens=['<IMG_CONTEXT>'],
        auto_add_bos=True))
register_template(
    Phi3TemplateMeta(
        MLLMTemplateType.internvl_phi3,
        default_system='You are an AI assistant whose name is Phi-3.',
        template_cls=InternvlTemplate,
        placeholder_tokens=['<IMG_CONTEXT>'],
        auto_add_bos=True))


class Internvl2Template(InternvlTemplate):
    video_segments = 8

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        image_context = super().replace_tag('image', index, inputs)
        if media_type == 'image':
            return image_context
        elif media_type == 'video':
            video_segments = get_env_args('video_segments', int, self.video_segments)
            load_video = partial(load_video_internvl, num_segments=video_segments)
            return self.replace_video2image(load_video, inputs, lambda i: [f'Frame{i + 1}: '] + image_context)

    def replace_object(self, object_: Dict[str, Any], index: int, inputs: StdTemplateInputs) -> List[Context]:
        objects = inputs.objects
        if objects:
            object_ = objects[index]
            return [f'<ref>{object_["caption"]}</ref>']
        else:
            return ['<ref-object>']

    def replace_box(self, object_: Dict[str, Any], index: int, inputs: StdTemplateInputs) -> List[Context]:
        objects = inputs.objects
        if objects:
            object_ = objects[index]
            if isinstance(object_['bbox'][0], list):
                all_objects = '<box> ['
                for sub_object in object_['bbox']:
                    all_objects += (f'[{sub_object[0]}, {sub_object[1]}, ' f'{sub_object[2]}, {sub_object[3]}],')
                all_objects = all_objects[:-1]
                all_objects += '] </box>'
                return [all_objects]
            else:
                return [
                    f'<box> [[{object_["bbox"][0]}, {object_["bbox"][1]}, '
                    f'{object_["bbox"][2]}, {object_["bbox"][3]}]] </box>'
                ]
        else:
            return ['<bbox>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super(InternvlTemplate, self)._encode(inputs)
        input_ids = encoded['input_ids']
        idx_list = findall(input_ids, -100)
        labels = encoded['labels']
        images = inputs.images
        if images:
            has_video = bool(inputs.videos)
            input_size = get_env_args('input_size', int, 448)
            max_num = get_env_args('max_num', int, 12)
            video_max_num = get_env_args('video_max_num', int, 1)
            if has_video:
                max_num = video_max_num
            pixel_values = [transform_image(image, input_size, max_num) for image in images]
            num_patches = [pv.shape[0] for pv in pixel_values]
            pixel_values = torch.cat(pixel_values).to(self.config.torch_dtype)
        else:
            pixel_values = None
            num_patches = []
        assert len(num_patches) == len(
            idx_list), f'len(num_patches): {len(num_patches)}, len(idx_list): {len(idx_list)}'
        added_tokens_len = 0
        for idx, num_patch in zip(idx_list, num_patches):
            img_tokens: List[int] = self.processor.encode(
                '<IMG_CONTEXT>', add_special_tokens=False) * self.num_image_token * num_patch
            input_ids = input_ids[:idx + added_tokens_len] + img_tokens + input_ids[idx + added_tokens_len + 1:]
            if labels is not None:
                labels = labels[:idx + added_tokens_len] + [-100] * len(img_tokens) + labels[idx + added_tokens_len
                                                                                             + 1:]
            added_tokens_len += len(img_tokens) - 1
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['pixel_values'] = pixel_values
        return encoded


_internvl2_system = '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'
register_template(
    ChatmlTemplateMeta(
        MLLMTemplateType.internvl2,
        default_system=_internvl2_system,
        template_cls=Internvl2Template,
        placeholder_tokens=['<IMG_CONTEXT>'],
    ))

register_template(
    Phi3TemplateMeta(
        MLLMTemplateType.internvl2_phi3,
        default_system=_internvl2_system,
        template_cls=Internvl2Template,
        placeholder_tokens=['<IMG_CONTEXT>'],
    ))

register_template(
    ChatmlTemplateMeta(
        MLLMTemplateType.internvl2_5,
        template_cls=Internvl2Template,
        placeholder_tokens=['<IMG_CONTEXT>'],
        default_system='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'))
