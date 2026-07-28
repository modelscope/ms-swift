# Copyright (c) ModelScope Contributors. All rights reserved.

import math
import torch
from dataclasses import dataclass, field
from PIL import Image
from torch import nn as nn
from typing import Any, Dict, List, Literal, Optional

from swift.utils import is_deepspeed_enabled, to_device
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
    skip_prompt = False

    def _get_system(self, inputs: StdTemplateInputs) -> Optional[str]:
        system = super()._get_system(inputs)
        if system is not None and '<|im_middle|>' not in system:  # compat agent
            system = f'system<|im_middle|>{system}'
        return system

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<|media_begin|>image<|media_content|><|media_pad|><|media_end|>\n']
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
            image_inputs = image_processor([{
                'type': 'image',
                'image': image
            } for image in inputs.images],
                                           return_tensors='pt')
            grid_thws = image_inputs['grid_thws']
            merge_length = math.prod(self.config.vision_config.merge_kernel_size)

            def _get_new_tokens(i):
                token_len = (grid_thws[i].prod() // merge_length)
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

        if pixel_values is not None and pixel_values.size(0) > 0:
            pixel_values = pixel_values.to(model.vision_tower.dtype)
            image_features: torch.Tensor = model._extract_image_features(pixel_values, inputs['grid_thws'])
            if model.mm_projector:
                image_features = model.mm_projector(image_features)
            image_features = torch.cat(image_features, dim=0)
            inputs_embeds = inputs_embeds.to(image_features.dtype)
            image_mask = (input_ids == self.config.media_placeholder_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
        elif is_deepspeed_enabled():
            image_processor = self.processor.image_processor
            dummy_image = Image.new('RGB', (32, 32), (0, 0, 0))
            image_inputs = image_processor([{'type': 'image', 'image': dummy_image}], return_tensors='pt')
            image_inputs = to_device(image_inputs, inputs_embeds.device)
            pixel_values = image_inputs['pixel_values'].to(model.vision_tower.dtype)
            image_features: torch.Tensor = model._extract_image_features(pixel_values, image_inputs['grid_thws'])
            if model.mm_projector:
                image_features = model.mm_projector(image_features)
            image_features = torch.cat(image_features, dim=0)
            inputs_embeds = inputs_embeds + image_features.mean() * 0.
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


class KimiK3Template(Template):
    # Kimi-K3 renders chats in XTML: structural markers (<|open|>/<|close|>/<|sep|>/<|end_of_msg|>)
    # are special tokens while tag names (message/think/response) are plain text.
    # See `encoding_k3.py` in the model repo.
    placeholder_tokens = ['<|media_pad|>']
    support_padding_free = True
    skip_prompt = False
    jinja_enable_thinking_key = 'thinking'

    think_open = '<|open|>think<|sep|>'
    think_close = '<|close|>think<|sep|>'
    response_open = '<|open|>response<|sep|>'
    response_close = '<|close|>response<|sep|>'
    valid_thinking_efforts = {'low', 'high', 'max'}

    def __init__(self, *args, **kwargs):
        # Kimi K3 always has thinking enabled (the base class defaults hybrid-thinking
        # templates with a non_thinking_prefix to False).
        if kwargs.get('enable_thinking') is None:
            kwargs['enable_thinking'] = True
        super().__init__(*args, **kwargs)

    def _thinking_to_xtml(self, text: str, complete: bool = True) -> str:
        """Convert the inline `<think>...</think>` convention into the K3 think/response channels.

        The <think> channel is structural: every assistant message carries the open/close
        tags even when there is no reasoning content (aligned with `encoding_k3.py`).
        A complete message also owns the response-channel close, so that a tool-calls
        section can follow it (chat_sep/suffix only close the enclosing message).
        """
        if text.startswith(self.think_open):  # already converted
            return text
        if text.startswith('<think>'):
            body = text[len('<think>'):]
            think, sep, response = body.partition('</think>')
            if not sep:  # bare thinking prefix (generation prompt)
                return self.think_open + think
            res = self.think_open + think + self.think_close + self.response_open + response
        else:
            res = self.think_open + self.think_close + self.response_open + text
        if complete:
            res += self.response_close
        return res

    def _xtml_to_thinking(self, text: str) -> str:
        """Inverse of `_thinking_to_xtml`: map the K3 think/response channels back to
        the inline `<think>...</think>` convention used across swift, so that decoded
        responses round-trip through `_preprocess_inputs` in multi-turn conversations.
        A tool-calls section (`<|open|>tools<|sep|>...`) is left untouched for
        `get_toolcall` to parse.
        """
        text = text.replace(self.think_close + self.response_open, '</think>')
        text = text.replace(self.think_open, '<think>')
        # Unpaired markers (e.g. truncated or non-standard generations).
        text = text.replace(self.think_close, '</think>')
        text = text.replace(self.response_open, '')
        return text.replace(self.response_close, '')

    def decode_generate_ids(self, generate_ids, **kwargs):
        response = super().decode_generate_ids(generate_ids, **kwargs)
        if isinstance(response, str):
            response = self._xtml_to_thinking(response)
        return response

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_inputs(inputs)
        # Convert every assistant message into the XTML think/response channel form
        # ahead of the base thinking machinery, so that the rendered response always
        # equals the message content (required by loss_scale).
        for message in inputs.messages:
            if message['role'] == 'assistant' and isinstance(message['content'], str):
                message['content'] = self._thinking_to_xtml(message['content'])

    def _add_non_thinking_prefix(self, inputs, thinking_prefix='<think>') -> None:
        return super()._add_non_thinking_prefix(inputs, thinking_prefix=self.think_open)

    def _remove_thinking_content(self, content: str, thinking_suffix='</think>') -> str:
        content = content.split(self.think_close)[-1].strip()
        if content.startswith(self.response_open):
            content = content[len(self.response_open):]
        return self.template_meta.history_thinking_prefix + content

    def _get_preserve_thinking(self, inputs=None):
        preserve_thinking = None if inputs is None else inputs.chat_template_kwargs.get('preserve_thinking')
        if preserve_thinking is None:
            preserve_thinking = self.preserve_thinking
        if preserve_thinking is None:
            # K3 was trained in preserved-thinking-history mode: keep historical
            # reasoning by default (pass `--preserve_thinking false` to drop it).
            preserve_thinking = True
        return preserve_thinking

    def _get_system(self, inputs: StdTemplateInputs) -> Optional[str]:
        system = super()._get_system(inputs)
        if system is not None and '<|sep|>' not in system:  # compat agent
            system = f'role="system"<|sep|>{system}'
        return system

    def _get_response_prefix(self, inputs=None):
        response_prefix = super()._get_response_prefix(inputs)
        if not response_prefix:
            return response_prefix
        # A user-supplied response_prefix may still use the inline convention;
        # a generation prefix must keep the response channel open.
        return self._thinking_to_xtml(response_prefix, complete=False)

    def _get_thinking_effort(self, inputs=None):
        # `reasoning_effort` is the K3 API field name; `thinking_effort` is used by the
        # chat encoder (encoding_k3.py). Accept both via chat_template_kwargs.
        kwargs = {} if inputs is None else inputs.chat_template_kwargs
        thinking_effort = kwargs.get('thinking_effort') or kwargs.get('reasoning_effort')
        if thinking_effort is not None:
            assert thinking_effort in self.valid_thinking_efforts, (
                f'Unsupported thinking_effort={thinking_effort!r}; '
                f'supported values are {sorted(self.valid_thinking_efforts)}.')
        return thinking_effort

    def _swift_encode(self, inputs: StdTemplateInputs):
        res_context_list, loss_scale_list, answer_len = super()._swift_encode(inputs)
        thinking_effort = self._get_thinking_effort(inputs)
        if thinking_effort is not None:
            # Aligned with `_internal_system_message` in encoding_k3.py: the
            # thinking-effort system message is rendered before all messages,
            # but after the tool-declare system message when tools are present.
            context = ('<|open|>message role="system" type="thinking-effort"<|sep|>'
                       '`thinking_effort` guides on how much to think in your '
                       'thinking channel (not including the response channel), '
                       'supported values include `low`, `medium`, `high`, and `max`.\n'
                       f'Now the system is invoked with `thinking_effort={thinking_effort}`.'
                       '<|close|>message<|sep|><|end_of_msg|>')
            first = res_context_list[0] if res_context_list else None
            if inputs.tools and isinstance(first, str) and 'type="tool-declare"' in first:
                end = first.index('<|end_of_msg|>') + len('<|end_of_msg|>')
                res_context_list[0] = first[:end] + context + first[end:]
            else:
                res_context_list.insert(0, context)
                loss_scale_list.insert(0, 0.)
        return res_context_list, loss_scale_list, answer_len

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            # The image prompt carries the original resolution, e.g.
            # `<|media_begin|>image {W}x{H}<|media_content|><|media_pad|><|media_end|>`.
            image = inputs.images[index]
            width, height = image.size
            return [self.processor.image_processor.make_image_prompt(width, height)]
        raise ValueError(f'KimiK3Template does not currently support {media_type}. '
                         '(The official KimiK3Processor only supports images.)')

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        media_token = self._tokenize('<|media_pad|>')[0]
        idx_list = findall(input_ids, media_token)
        if inputs.images:
            image_processor = self.processor.image_processor
            medias = [{'type': 'image', 'image': image} for image in inputs.images]
            image_inputs = image_processor.preprocess(medias, return_tensors='pt')
            # Pre-expand `<|media_pad|>` so labels stay aligned; the model-side
            # `_merge_input_ids_with_image_features` (which changes the sequence length
            # inside forward) is bypassed by `_post_encode`.
            num_tokens_list = [image_processor.media_tokens_calculator(media) for media in medias]

            def _get_new_tokens(i):
                return [media_token] * num_tokens_list[i]

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

        if pixel_values is not None and pixel_values.size(0) > 0:
            pixel_values = pixel_values.to(model.vision_tower.patch_embed.proj.weight.dtype)
            image_features = model._extract_image_features(pixel_values, inputs['grid_thws'])
            if model.mm_projector:
                image_features = model.mm_projector(image_features)
            image_features = torch.cat(image_features, dim=0)
            inputs_embeds = inputs_embeds.to(image_features.dtype).clone()
            image_mask = (input_ids == self.config.media_placeholder_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
        elif is_deepspeed_enabled():
            image_processor = self.processor.image_processor
            dummy_image = Image.new('RGB', (32, 32), (0, 0, 0))
            image_inputs = image_processor.preprocess([{'type': 'image', 'image': dummy_image}], return_tensors='pt')
            image_inputs = to_device(image_inputs, inputs_embeds.device)
            pixel_values = image_inputs['pixel_values'].to(model.vision_tower.patch_embed.proj.weight.dtype)
            image_features = model._extract_image_features(pixel_values, image_inputs['grid_thws'])
            if model.mm_projector:
                image_features = model.mm_projector(image_features)
            image_features = torch.cat(image_features, dim=0)
            inputs_embeds = inputs_embeds + image_features.mean() * 0.
        return {'inputs_embeds': inputs_embeds}


@dataclass
class KimiK3TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: [
        '<|open|>message role="user"<|sep|>{{QUERY}}<|close|>message<|sep|><|end_of_msg|>'
        '<|open|>message role="assistant"<|sep|>'
    ])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|close|>message<|sep|><|end_of_msg|>'])
    suffix: Prompt = field(default_factory=lambda: ['<|close|>message<|sep|><|end_of_msg|>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<|open|>message {{SYSTEM}}<|close|>message<|sep|><|end_of_msg|>'])
    default_system: Optional[str] = None


register_template(
    KimiK3TemplateMeta(
        MLLMTemplateType.kimi_k3,
        template_cls=KimiK3Template,
        # Kimi K3 always has thinking enabled; the <think> channel is structural
        # (assistant messages carry it even when the reasoning is empty).
        is_thinking=True,
        thinking_prefix='<|open|>think<|sep|>',
        non_thinking_prefix='<|open|>think<|sep|><|close|>think<|sep|><|open|>response<|sep|>',
        history_thinking_prefix='<|open|>think<|sep|><|close|>think<|sep|><|open|>response<|sep|>',
        agent_template='kimi_k3',
    ))
