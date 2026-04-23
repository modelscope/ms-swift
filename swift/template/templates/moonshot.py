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
    default_system: str = 'You are a helpful assistant'


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
    use_model = True

    def _get_system(self, inputs: StdTemplateInputs) -> Optional[str]:
        system = super()._get_system(inputs)
        if system is not None and '<|im_middle|>' not in system:
            system = f'system<|im_middle|>{system}'
        return system

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<|media_begin|>image<|media_content|><|media_pad|><|media_end|>']
        elif media_type == 'video':
            return ['<|kimi_k25_video_placeholder|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)

        if not inputs.images and not inputs.videos:
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
            encoded['loss_scale'] = loss_scale
            return encoded

        image_processor = self.processor.image_processor
        media_token = self._tokenize('<|media_pad|>')[0]
        video_ph_token = self._tokenize('<|kimi_k25_video_placeholder|>')[0]

        # Determine media ordering from input_ids to maintain correct correspondence
        media_order = []  # list of ('image', idx) or ('video', idx)
        img_counter = 0
        vid_counter = 0
        for token in input_ids:
            if token == media_token:
                media_order.append(('image', img_counter))
                img_counter += 1
            elif token == video_ph_token:
                media_order.append(('video', vid_counter))
                vid_counter += 1

        # Process videos into chunks of temporal_merge_kernel_size frames
        video_chunks_map = {}  # vid_idx -> list of chunk MediaInput dicts
        video_prompts_map = {}  # vid_idx -> concatenated chunk prompt string
        if inputs.videos:
            temporal_merge = image_processor.num_frames_per_chunk
            sample_fps = image_processor.media_proc_cfg.get('sample_fps', 2.0)
            chunk_duration = temporal_merge / sample_fps
            for vid_idx, video_frames in enumerate(inputs.videos):
                chunks = []
                prompt_parts = []
                for i in range(0, len(video_frames), temporal_merge):
                    chunk_frames = video_frames[i:i + temporal_merge]
                    chunk_num = i // temporal_merge
                    start_time = chunk_num * chunk_duration
                    hours = int(start_time // 3600)
                    minutes = int((start_time % 3600) // 60)
                    seconds = start_time % 60
                    timestamp = f'{hours:02d}:{minutes:02d}:{seconds:06.3f}'
                    chunk_prompt = image_processor.make_chunk_prompt(timestamp)
                    prompt_parts.append(chunk_prompt)
                    chunks.append({
                        'type': 'video_chunk',
                        'video_chunk': list(chunk_frames),
                        'prompt': chunk_prompt,
                    })
                video_chunks_map[vid_idx] = chunks
                video_prompts_map[vid_idx] = ''.join(prompt_parts)

        # Step 1: Replace video placeholders with chunk prompt tokens
        if video_prompts_map:
            video_idx_list = findall(input_ids, video_ph_token)

            def _get_video_tokens(i):
                return self._tokenize(video_prompts_map[i])

            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, video_idx_list,
                                                                _get_video_tokens)

        # Build medias list in conversation order (must match media_pad token order)
        medias = []
        for media_type, idx in media_order:
            if media_type == 'image':
                medias.append({'type': 'image', 'image': inputs.images[idx]})
            elif media_type == 'video':
                medias.extend(video_chunks_map.get(idx, []))

        # Step 2: Process all medias through image_processor and expand media_pad tokens
        if medias:
            image_inputs = image_processor.preprocess(medias, return_tensors='pt')
            grid_thws = image_inputs['grid_thws']
            idx_list = findall(input_ids, media_token)
            merge_kernel_size = image_processor.media_proc_cfg['merge_kernel_size']
            if isinstance(merge_kernel_size, (list, tuple)):
                kernel_h, kernel_w = merge_kernel_size
            else:
                kernel_h = kernel_w = merge_kernel_size

            def _get_new_tokens(i):
                t, h, w = grid_thws[i].tolist()
                # tpool_patch_merger pools temporal dim, so token count excludes t
                token_len = (h // kernel_h) * (w // kernel_w)
                return [media_token] * token_len

            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                _get_new_tokens)
            encoded.update(image_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
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

        if pixel_values is not None and len(pixel_values) > 0 and input_ids.shape[1] != 1:
            inputs_embeds = model.get_input_embeddings()(input_ids)
            image_features = model._extract_image_features(pixel_values, inputs['grid_thws'])
            image_features = model.mm_projector(image_features)
            inputs_embeds = inputs_embeds.to(image_features[0].dtype).clone()
            inputs_embeds, attention_mask, labels, position_ids = model._merge_input_ids_with_image_features(
                image_features,
                inputs_embeds,
                input_ids,
                inputs.get('attention_mask'),
                inputs.get('labels'),
            )
            return {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
                'labels': labels,
                'position_ids': position_ids,
            }
        else:
            inputs_embeds = model.get_input_embeddings()(input_ids)
            if is_deepspeed_enabled():
                image_processor = self.processor.image_processor
                dummy_image = Image.new('RGB', (32, 32), (0, 0, 0))
                dummy_media = [{'type': 'image', 'image': dummy_image}]
                dummy_inputs = image_processor.preprocess(dummy_media, return_tensors='pt')
                target_dtype = model.vision_tower.patch_embed.proj.weight.dtype
                dummy_pv = dummy_inputs['pixel_values'].to(dtype=target_dtype, device=inputs_embeds.device)
                dummy_thws = dummy_inputs['grid_thws'].to(device=inputs_embeds.device)
                image_features = model._extract_image_features(dummy_pv, dummy_thws)
                image_features = model.mm_projector(image_features)
                inputs_embeds = inputs_embeds + image_features[0].mean() * 0.
            return {'inputs_embeds': inputs_embeds}


register_template(
    MoonlightTemplateMeta(
        MLLMTemplateType.kimi_k25,
        template_cls=KimiK25Template,
        default_system=None,
        is_thinking=True,
        thinking_prefix='<think>\n',
        non_thinking_prefix='<think></think>',
        history_thinking_prefix='<think></think>',
        agent_template='kimi_k25',
    ))
