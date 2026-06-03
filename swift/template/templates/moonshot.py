# Copyright (c) ModelScope Contributors. All rights reserved.

import torch
from dataclasses import dataclass, field
from PIL import Image
from torch import nn as nn
from typing import Any, Dict, List, Literal, Optional

from swift.utils import is_deepspeed_enabled
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall


@dataclass
class MoonlightTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda:
                           ['<|im_user|>user<|im_middle|>{{QUERY}}<|im_end|><|im_assistant|>assistant<|im_middle|>'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<|im_system|>system<|im_middle|>{{SYSTEM}}<|im_end|>'])
    default_system: Optional[str] = 'You are a helpful assistant'


register_template(MoonlightTemplateMeta(LLMTemplateType.moonlight))

register_template(
    MoonlightTemplateMeta(
        LLMTemplateType.kimi_k2, default_system='You are Kimi, an AI assistant created by Moonshot AI.'))


class KimiVLTemplate(Template):
    placeholder_tokens = ['<|media_pad|>']
    support_padding_free = True
    skip_prompt = False

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<|media_start|>image<|media_content|><|media_pad|><|media_end|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        media_token = self._tokenize('<|media_pad|>')[0]
        idx_list = findall(input_ids, media_token)
        if inputs.images:
            image_processor = self.processor.image_processor
            image_inputs = image_processor(inputs.images, return_tensors='pt')
            image_grid_hws = image_inputs['image_grid_hws']
            merge_length = image_processor.merge_kernel_size[0] * image_processor.merge_kernel_size[1]

            def _get_new_tokens(i):
                token_len = (image_grid_hws[i].prod() // merge_length)
                return [media_token] * token_len

            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                _get_new_tokens)

            encoded['loss_scale'] = loss_scale
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
            encoded.update(image_inputs)
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        image_grid_hws = self.concat_tensor(batch, 'image_grid_hws', 0)
        if image_grid_hws is not None:
            res['image_grid_hws'] = image_grid_hws
        return res

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = inputs['input_ids']
        pixel_values = inputs.get('pixel_values')
        inputs_embeds = model.get_input_embeddings()(input_ids)

        if pixel_values is not None and pixel_values.size(0) > 0:
            pixel_values = pixel_values.to(model.vision_tower.dtype)
            image_features: torch.Tensor = model._extract_image_features(pixel_values, inputs['image_grid_hws'])
            inputs_embeds = inputs_embeds.to(image_features[0].dtype).clone()
            inputs_embeds = model._merge_with_image_features(inputs_embeds, input_ids, image_features)
        elif is_deepspeed_enabled():
            image_processor = self.processor.image_processor
            dummy_image = Image.new('RGB', (32, 32), (0, 0, 0))
            image_inputs = image_processor([dummy_image], return_tensors='pt')
            pixel_values = image_inputs['pixel_values'].to(model.vision_tower.dtype)
            image_features: torch.Tensor = model._extract_image_features(pixel_values, image_inputs['image_grid_hws'])
            inputs_embeds = inputs_embeds + image_features.mean() * 0.
        return {'inputs_embeds': inputs_embeds}


register_template(MoonlightTemplateMeta(MLLMTemplateType.kimi_vl, template_cls=KimiVLTemplate))


class KimiK25Template(Template):
    placeholder_tokens = ['<|media_pad|>', '<|kimi_k25_video_placeholder|>']
    jinja_enable_thinking_key = 'thinking'
    support_padding_free = True

    def _get_system(self, inputs: StdTemplateInputs) -> Optional[str]:
        system = super()._get_system(inputs)
        if system is not None and '<|im_middle|>' not in system:  # compat agent
            system = f'system<|im_middle|>{system}'
        return system

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<|media_start|>image<|media_content|><|media_pad|><|media_end|>']
        raise ValueError(f'KimiK25Template does not currently support {media_type}. '
                         'Please open an issue to request support.')

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        media_token = self._tokenize('<|media_pad|>')[0]
        idx_list = findall(input_ids, media_token)
        if inputs.images:
            image_processor = self.processor.image_processor
            medias = [{'type': 'image', 'image': img} for img in inputs.images]
            image_inputs = image_processor.preprocess(medias, return_tensors='pt')
            grid_thws = image_inputs['grid_thws']
            media_proc_cfg = getattr(image_processor, 'media_proc_cfg', {})
            merge_kernel_size = media_proc_cfg.get('merge_kernel_size', (2, 2))
            if isinstance(merge_kernel_size, (list, tuple)):
                merge_length = merge_kernel_size[0] * merge_kernel_size[1]
            else:
                merge_length = merge_kernel_size * merge_kernel_size

            def _get_new_tokens(i):
                token_len = (grid_thws[i].prod() // merge_length).item()
                return [media_token] * token_len

            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                _get_new_tokens)
            encoded['loss_scale'] = loss_scale
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
            encoded.update(image_inputs)
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        grid_thws = self.concat_tensor(batch, 'grid_thws', 0)
        if grid_thws is not None:
            res['grid_thws'] = grid_thws
        return res

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = inputs['input_ids']
        pixel_values = inputs.get('pixel_values')
        inputs_embeds = model.get_input_embeddings()(input_ids)

        base_model = self.get_base_model(model)
        if pixel_values is not None and pixel_values.size(0) > 0:
            vision_tower = base_model.vision_tower
            vision_dtype = next(vision_tower.parameters()).dtype
            pixel_values = pixel_values.to(device=inputs_embeds.device, dtype=vision_dtype)
            grid_thws = inputs.get('grid_thws')
            if grid_thws is None:
                raise KeyError('pixel_values present in inputs but grid_thws is missing')
            grid_thws = grid_thws.to(inputs_embeds.device)
            # vision_tower produces un-projected features; the mm_projector maps them to
            # the language hidden size (mirrors KimiK25ForConditionalGeneration.forward).
            image_features: list = base_model._extract_image_features(pixel_values, grid_thws)
            if getattr(base_model, 'mm_projector', None):
                image_features = base_model.mm_projector(image_features)
            all_features = torch.cat(image_features, dim=0).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            media_token_id = (
                getattr(model.config, 'media_placeholder_token_id', None)
                or self.tokenizer.convert_tokens_to_ids('<|media_pad|>'))
            inputs_embeds = inputs_embeds.clone()
            mask = input_ids.reshape(-1) == media_token_id
            inputs_embeds.reshape(-1, inputs_embeds.size(-1))[mask] = all_features
        elif is_deepspeed_enabled():
            image_processor = self.processor.image_processor
            dummy_image = Image.new('RGB', (32, 32), (0, 0, 0))
            dummy_inputs = image_processor.preprocess([{'type': 'image', 'image': dummy_image}], return_tensors='pt')
            vision_tower = base_model.vision_tower
            vision_dtype = next(vision_tower.parameters()).dtype
            dummy_pixels = dummy_inputs['pixel_values'].to(device=inputs_embeds.device, dtype=vision_dtype)
            dummy_grid = dummy_inputs['grid_thws'].to(inputs_embeds.device)
            image_features = base_model._extract_image_features(dummy_pixels, dummy_grid)
            if getattr(base_model, 'mm_projector', None):
                image_features = base_model.mm_projector(image_features)
            if image_features:
                # nan_to_num guards against a non-finite value from the all-zero dummy
                # pass leaking into the text batch (NaN * 0 == NaN in IEEE-754).
                zero_term = torch.nan_to_num(torch.cat(image_features, dim=0).mean() * 0.)
                inputs_embeds = inputs_embeds + zero_term.to(dtype=inputs_embeds.dtype)
        return {'inputs_embeds': inputs_embeds}


register_template(
    MoonlightTemplateMeta(
        MLLMTemplateType.kimi_k25,
        template_cls=KimiK25Template,
        system_prefix=['<|im_system|>{{SYSTEM}}<|im_end|>'],
        default_system=None,
        is_thinking=True,
        thinking_prefix='<think>',
        non_thinking_prefix='<think></think>',
        history_thinking_prefix='<think></think>',
        agent_template='kimi_k25',
    ))
