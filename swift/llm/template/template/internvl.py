# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Any, Dict, List, Literal, Optional

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
from .utils import ChatmlTemplateMeta, ThinkingTemplate


class InternvlTemplate(Template):
    skip_prompt = False
    num_image_token = None
    placeholder_tokens = ['<IMG_CONTEXT>']
    support_padding_free = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if self.mode == 'vllm':
            image_context = ['<image>\n']
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
            if self.num_image_token is None:
                self.num_image_token = int((input_size // 14)**2 * (0.5**2))
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

    def forward_context(self, model, inputs):
        model_name = model.language_model.__class__.__name__.lower()
        if self.padding_free and 'internlm2' in model_name:
            position_ids = inputs['position_ids']
            modeling_module = model.language_model.model.layers[0].attention.__class__
            return self._patch_flash_attention_forward(modeling_module, position_ids, use_new_func=True)
        else:
            return super().forward_context(model, inputs)

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
        auto_add_bos=True))
register_template(
    Phi3TemplateMeta(
        MLLMTemplateType.internvl_phi3,
        default_system='You are an AI assistant whose name is Phi-3.',
        template_cls=InternvlTemplate,
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

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<ref>{ref}</ref>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<box>[{bbox}]</box>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super(InternvlTemplate, self)._encode(inputs)
        input_ids = encoded['input_ids']
        idx_list = findall(input_ids, -100)
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        images = inputs.images
        if images:
            has_video = bool(inputs.videos)
            input_size = get_env_args('input_size', int, 448)
            if self.num_image_token is None:
                self.num_image_token = int((input_size // 14)**2 * (0.5**2))
            max_num = get_env_args('max_num', int, 12)
            video_max_num = get_env_args('video_max_num', int, 1)
            if has_video:
                max_num = video_max_num
            pixel_values = [transform_image(image, input_size, max_num) for image in images]
            num_patches = [pv.shape[0] for pv in pixel_values]
            pixel_values = torch.cat(pixel_values).to(self.model_info.torch_dtype)
        else:
            pixel_values = None
            num_patches = []
        assert len(num_patches) == len(
            idx_list), f'len(num_patches): {len(num_patches)}, len(idx_list): {len(idx_list)}'

        def _get_new_tokens(i):
            img_tokens: List[int] = self.processor.encode(
                '<IMG_CONTEXT>', add_special_tokens=False) * self.num_image_token * num_patches[i]
            return img_tokens

        encoded['input_ids'], encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
            input_ids, labels, loss_scale, idx_list, _get_new_tokens)
        encoded['pixel_values'] = pixel_values
        return encoded


_internvl2_system = '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'
register_template(
    ChatmlTemplateMeta(
        MLLMTemplateType.internvl2,
        default_system=_internvl2_system,
        template_cls=Internvl2Template,
    ))

register_template(
    Phi3TemplateMeta(
        MLLMTemplateType.internvl2_phi3,
        default_system=_internvl2_system,
        template_cls=Internvl2Template,
    ))

register_template(
    ChatmlTemplateMeta(
        MLLMTemplateType.internvl2_5,
        template_cls=Internvl2Template,
        default_system='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'))


class InternS1Template(Internvl2Template, ThinkingTemplate):
    image_token_id = 152957
    InternS1DefaultThinkinngSystem = ('You are an expert reasoner with extensive experience in all areas. '
                                      'You approach problems through systematic thinking and rigorous reasoning. '
                                      'Your response should reflect deep understanding and precise logical thinking, '
                                      'making your solution path and reasoning clear to others. '
                                      'Please put your thinking process within <think>...</think> tags.')

    def _swift_encode(self, inputs: StdTemplateInputs):
        if inputs.system is None and self.template_meta.response_prefix == '<think>':
            inputs.system = self.InternS1DefaultThinkinngSystem

        return super()._swift_encode(inputs)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from transformers.image_utils import make_flat_list_of_images
        import numpy as np
        encoded = super(InternvlTemplate, self)._encode(inputs)
        input_ids = encoded['input_ids']
        idx_list = findall(input_ids, -100)
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        images = inputs.images
        if inputs.videos:
            # TODO
            raise NotImplementedError('Video is not supported yet.')
        if images:
            # InternS1Processor
            images = make_flat_list_of_images(images)
            image_inputs = self.processor.image_processor(images=images, crop_to_patches=True, return_tensors='pt')
            image_num_patches = image_inputs.pop('num_patches')
            pixel_values = image_inputs.pop('pixel_values')
            image_num_patches_indices = np.cumsum(image_num_patches)
            # has_video = bool(inputs.videos) # TODO:video
        else:
            pixel_values = None
            image_num_patches_indices = []
        assert len(image_num_patches_indices) == len(
            idx_list), f'len(num_patches): {len(num_patches)}, len(idx_list): {len(idx_list)}'

        def _get_new_tokens(i):
            start = image_num_patches_indices[i - 1] if i > 0 else 0
            end = image_num_patches_indices[i]
            image_seq_length = self.processor.image_seq_length
            img_tokens: List[int] = self.processor.encode(
                '<IMG_CONTEXT>', add_special_tokens=False) * image_seq_length * image_num_patches[start:end]
            return img_tokens

        encoded['input_ids'], encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
            input_ids, labels, loss_scale, idx_list, _get_new_tokens)
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
            vit_embeddings = model.model.vision_tower.embeddings
            lm_embeddings = model.model.language_model.get_input_embeddings()
            vit_embeds = vit_embeddings(pixel_values)[0].to(device=device)
            special_image_mask = inputs_embeds == lm_embeddings(
                torch.tensor(self.image_token_id, dtype=torch.long, device=inputs_embeds.device))
            special_image_mask = special_image_mask.all(-1)
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_features = model.model.get_image_features(
                pixel_values, vision_feature_layer=-1, vision_feature_select_strategy='default')
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        elif is_deepspeed_enabled():
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), device=device, dtype=inputs_embeds.dtype)
            vit_embeds = model.model.vision_tower.embeddings(dummy_pixel_values)[0].to(device=device)
            inputs_embeds += vit_embeds.mean() * 0.
        return {'inputs_embeds': inputs_embeds}


# disable_thinking: response_prefix=''
register_template(
    ChatmlTemplateMeta(MLLMTemplateType.interns1, template_cls=InternS1Template, response_prefix='<think>'))
