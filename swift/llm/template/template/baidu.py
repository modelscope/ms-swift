# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall
from .utils import ThinkingTemplate


@dataclass
class ERNIETemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<|begin_of_sentence|>'])
    prompt: Prompt = field(default_factory=lambda: ['User: {{QUERY}}\nAssistant: '])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|end_of_sentence|>'])
    suffix: Prompt = field(default_factory=lambda: ['</s>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|begin_of_sentence|>{{SYSTEM}}\n'])


register_template(ERNIETemplateMeta(LLMTemplateType.ernie))


class ErnieThinkingTemplate(ThinkingTemplate):

    def _swift_prepare_inputs(self, inputs) -> None:
        super()._swift_prepare_inputs(inputs)
        for message in inputs.messages:
            if message['role'] == 'assistant':
                if '<response>' not in message['content']:
                    if '</think>' in message['content']:
                        message['content'] = message['content'].replace('</think>', '</think>\n\n<response>\n')
                        message['content'] = message['content'] + '\n</response>'
                        if '<think>\n' not in message['content']:
                            message['content'] = message['content'].replace('<think>', '<think>\n')
                    else:
                        message['content'] = '<response>\n' + message['content'] + '\n</response>\n'


@dataclass
class ERNIEThinkingTemplateMeta(TemplateMeta):
    prefix: Prompt = field(
        default_factory=lambda:
        ['<|im_start|>system\n'
         '<global_setting>\n'
         'think_mode=True\n'
         '</global_setting><|im_end|>\n\n'])
    prompt: Prompt = field(
        default_factory=lambda: ['<|im_start|>user\n'
                                 '{{QUERY}}<|im_end|>\n\n'
                                 '<|im_start|>assistant\n'])
    response_prefix: Optional[str] = '<think>\n'
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>\n\n'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: [
        '<|im_start|>system\n'
        '<system_setting>\n'
        '{{SYSTEM}}\n'
        '</system_setting>\n\n'
        '<global_setting>\n'
        'think_mode=True\n'
        '</global_setting><|im_end|>\n\n'
    ])


register_template(ERNIEThinkingTemplateMeta(LLMTemplateType.ernie_thinking, template_cls=ErnieThinkingTemplate))


class PaddleOCRTemplate(Template):
    image_placeholder = ['<image>']
    image_token = '<|IMAGE_PLACEHOLDER|>'
    image_token_id = 100295
    skip_prompt = False

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        if self.mode == 'vllm':
            assert NotImplementedError
        return ['<|IMAGE_START|>', [-100], '<|IMAGE_END|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        idx_list = findall(input_ids, -100)
        processor = self.processor
        images = inputs.images
        if images:
            image_inputs = processor.image_processor(images=images, return_tensors='pt')
            image_inputs['pixel_values'] = image_inputs['pixel_values']
            image_grid_thw = image_inputs['image_grid_thw']
            merge_size = processor.image_processor.merge_size**2

            def _get_new_tokens(i):
                img_tokens: List[int] = [self.image_token_id] * (image_grid_thw[i].prod() // merge_size)
                return img_tokens

            encoded['input_ids'], encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, _get_new_tokens)
            encoded['pixel_values'] = image_inputs['pixel_values']
            encoded['image_grid_thw'] = image_grid_thw

        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        embedding = model.get_input_embeddings()
        device = embedding.weight.device
        input_ids = inputs['input_ids']
        inputs_embeds = embedding(input_ids).to(device=device)
        pixel_values = inputs.get('pixel_values')
        image_grid_thw = inputs.get('image_grid_thw')
        if pixel_values is not None:
            siglip_position_ids = list()
            image_grid_hws = list()
            sample_indices = list()
            cu_seqlens = [0]
            pixel_values = pixel_values.unsqueeze(0).to(device=device)
            for idx, thw in enumerate(image_grid_thw):
                thw_tuple = tuple(thw.detach().cpu().numpy().tolist())
                numel = np.prod(thw_tuple)
                image_grid_hws.append(thw_tuple)
                image_position_ids = torch.arange(numel) % np.prod(thw_tuple[1:])
                siglip_position_ids.append(image_position_ids)
                sample_indices.append(torch.full((numel, ), idx, dtype=torch.int64))
                cu_seqlens.append(cu_seqlens[-1] + numel)

            siglip_position_ids = torch.concat(siglip_position_ids, dim=0).to(pixel_values.device)
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32).to(pixel_values.device)
            sample_indices = torch.concat(sample_indices, dim=0).to(pixel_values.device)

            vision_outputs = model.visual(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_hws,
                position_ids=siglip_position_ids,
                vision_return_embed_list=True,
                interpolate_pos_encoding=True,
                sample_indices=sample_indices,
                cu_seqlens=cu_seqlens,
                return_pooler_output=False,
                use_rope=True,
                window_size=-1,
            )
            image_embeds = vision_outputs.last_hidden_state
            image_embeds = model.mlp_AR(image_embeds, image_grid_thw)
            n_image_tokens = (input_ids == self.image_token_id).sum().item()
            image_embeds = torch.cat(image_embeds, dim=0)
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError('Image features and image tokens do not match: tokens: '
                                 f'{n_image_tokens}, features {n_image_features}')

            mask = input_ids == self.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        return {'inputs_embeds': inputs_embeds}


register_template(ERNIETemplateMeta(MLLMTemplateType.paddle_ocr, template_cls=PaddleOCRTemplate))


class ERNIE_VLTemplate(Template):
    placeholder_tokens = ['<|IMAGE_PLACEHOLDER|>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [f'Picture {index+1}:<|IMAGE_PLACEHOLDER|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded['loss_scale']
        image_token = self._tokenize('<|IMAGE_PLACEHOLDER|>')[0]
        idx_list = findall(input_ids, image_token)
        if idx_list:
            split_token = self._tokenize('\n')[0]
            new_inputs = self.processor(
                text=['\n'.join(['<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>'] * len(idx_list))],
                images=inputs.images,
                videos=inputs.videos,
                padding=True,
                return_tensors='pt',
            )
            splited_tokens = self._split_list(new_inputs['input_ids'][0].tolist(), split_token)
            # Insert image tokens into input_ids
            input_ids_len = len(input_ids)
            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                lambda i: splited_tokens[i])
            idx_list.append(input_ids_len)
            splited_tokens.append([])
            token_type_ids = []
            position_ids = []
            text_i, image_i, n_text_token = 0, 0, 0
            for i, idx in enumerate(idx_list):
                image_idx = image_i + len(splited_tokens[i])
                text_len = idx - text_i
                token_type_ids.append(torch.tensor([0] * (text_len))[None])
                token_type_ids.append(new_inputs['token_type_ids'][:, image_i:image_idx])
                text_position_ids = torch.arange(0, text_len)[None, :, None]
                start_idx = 0
                if position_ids:
                    start_idx = position_ids[-1][0, -1].max() + 1
                position_ids.append(torch.concat([text_position_ids + start_idx for _ in range(3)], dim=2))
                n_text_token += text_len
                position_ids.append(new_inputs['position_ids'][:, image_i:image_idx] + n_text_token)
                text_i = idx + 1
                n_text_token -= 1  # '\n'
                image_i = image_idx + 1
            token_type_ids = torch.cat(token_type_ids, dim=1)
            position_ids = torch.cat(position_ids, dim=1)
            encoded.update(new_inputs)
            encoded['token_type_ids'] = token_type_ids
            encoded['position_ids'] = position_ids
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
            encoded['loss_scale'] = loss_scale
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = {}
        for key in ['images', 'grid_thw', 'image_type_ids']:
            res[key] = self.concat_tensor(batch, key, 0)
        res.update(super()._data_collator(batch, padding_to=padding_to))
        return res

    def generate(self, model, *args, **kwargs):
        kwargs['use_cache'] = False
        return super().generate(model, *args, **kwargs)


register_template(
    ERNIETemplateMeta(MLLMTemplateType.ernie_vl, template_cls=ERNIE_VLTemplate, response_prefix='<think>'))

ERNIE_VL_SYSTEM = ('You are a multimodal AI assistant called ERNIE developed by Baidu based on the PaddlePaddle '
                   'framework.')
register_template(
    ERNIETemplateMeta(
        MLLMTemplateType.ernie_vl_thinking,
        template_cls=ERNIE_VLTemplate,
        response_prefix='\n<think>\n',
        default_system=ERNIE_VL_SYSTEM))
