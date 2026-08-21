# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import torch
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple

from swift.utils import get_env_args
from ..base import MaxLengthError, Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context
from .utils import ChatmlTemplateMeta


class MossVLTemplate(Template):
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    support_padding_free = False
    _vision_tokens = ('<|image_pad|>', '<|vision_start|>', '<|vision_end|>')
    _processor_runtime_keys = {
        'min_pixels',
        'max_pixels',
        'video_fps',
        'min_frames',
        'max_frames',
        'num_extract_threads',
    }
    _processor_env_keys = {
        'video_min_pixels': ('video_min_pixels', int),
        'video_max_pixels': ('video_max_pixels', int),
        'video_fps': ('fps', float),
        'min_frames': ('fps_min_frames', int),
        'max_frames': ('fps_max_frames', int),
        'num_extract_threads': ('num_extract_threads', int),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.sequence_parallel_size > 1:
            raise NotImplementedError('MOSS-VL does not support sequence parallel yet.')
        if self.truncation_strategy == 'split':
            raise ValueError(
                'MOSS-VL does not support truncation_strategy="split" because multimodal tensors cannot be split '
                'with the generic text-only path. Use "left", "right", or "raise" instead.')

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<|image|>']
        if media_type == 'video':
            return ['<|video|>']
        raise ValueError(f'MOSS-VL does not support media_type={media_type!r}.')

    def _is_binary_loss(self) -> bool:
        is_binary = self.is_binary_loss_scale
        if is_binary is None:
            is_binary = self.loss_scale.is_binary_loss_scale
        return is_binary

    def _context_to_text(self, context: Context) -> str:
        if isinstance(context, str):
            return context
        if isinstance(context, dict):
            context = context['token_ids']
        return self.tokenizer.decode(context, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def _render_text_and_spans(self, inputs: StdTemplateInputs) -> Tuple[str, List[List[int]]]:
        inputs.messages = deepcopy(inputs.messages)
        self._swift_prepare_inputs(inputs)
        if self.template_backend == 'jinja':
            if self.is_training:
                raise ValueError('MOSS-VL SFT requires template_backend="swift" to build labels_spans.')
            context_list, loss_scale_list, _ = self._jinja_encode(inputs)
        else:
            context_list, loss_scale_list, _ = self._swift_encode(inputs)
        context_list, loss_scale_list = self._simplify_context_list(context_list, loss_scale_list, inputs)

        if self.is_training and not self._is_binary_loss():
            raise ValueError('MOSS-VL currently supports binary loss-scale strategies only. '
                             f'Current loss_scale={self._loss_scale!r}.')
        if any(loss_weight not in {0, 1} for loss_weight in loss_scale_list):
            raise ValueError(f'MOSS-VL labels_spans cannot represent non-binary loss weights: {loss_scale_list}.')

        parts = []
        spans = []
        offset = 0
        for context, loss_weight in zip(context_list, loss_scale_list):
            text = self._context_to_text(context)
            parts.append(text)
            if self.is_training and loss_weight == 1 and text:
                spans.append([offset, offset + len(text)])
            offset += len(text)
        rendered = ''.join(parts)

        # Native MOSS-VL supervises every assistant <|im_end|>, but not the following newline.
        for span in spans:
            if rendered[span[1]:].startswith('<|im_end|>'):
                span[1] += len('<|im_end|>')
            if rendered[span[0]:span[1]].endswith('<|im_end|>\n'):
                span[1] -= 1

        merged_spans = []
        for span in spans:
            if merged_spans and span[0] <= merged_spans[-1][1]:
                merged_spans[-1][1] = max(merged_spans[-1][1], span[1])
            elif span[0] < span[1]:
                merged_spans.append(span)
        return rendered, merged_spans

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        rendered, labels_spans = self._render_text_and_spans(inputs)
        processor_kwargs = dict(inputs.mm_processor_kwargs)
        runtime_kwargs = {**self.chat_template_kwargs, **inputs.chat_template_kwargs}
        for key, (env_name, type_func) in self._processor_env_keys.items():
            if key not in runtime_kwargs and os.getenv(env_name.upper()) is not None:
                runtime_kwargs[key] = get_env_args(env_name, type_func, None)
        if self.max_pixels is not None:
            processor_kwargs.setdefault('max_pixels', self.max_pixels)
        for key in self._processor_runtime_keys:
            if key in runtime_kwargs:
                processor_kwargs.setdefault(key, runtime_kwargs[key])

        # MossVLProcessor does not route video_max_pixels to its video processor.
        # Translate the common runtime knobs to the nested native `size` contract.
        video_min_pixels = processor_kwargs.pop('video_min_pixels', runtime_kwargs.get('video_min_pixels'))
        video_max_pixels = processor_kwargs.pop('video_max_pixels', runtime_kwargs.get('video_max_pixels'))
        if video_min_pixels is not None or video_max_pixels is not None:
            videos_kwargs = dict(processor_kwargs.get('videos_kwargs') or {})
            size = dict(videos_kwargs.get('size') or getattr(self.processor.video_processor, 'size', {}) or {})
            if video_min_pixels is not None:
                size['shortest_edge'] = video_min_pixels
            if video_max_pixels is not None:
                size['longest_edge'] = video_max_pixels
            videos_kwargs['size'] = size
            processor_kwargs['videos_kwargs'] = videos_kwargs
        vision_chunked_length = int(
            processor_kwargs.pop(
                'vision_chunked_length',
                runtime_kwargs.get(
                    'vision_chunked_length',
                    get_env_args('mossvl_vision_chunked_length', int, 64),
                ),
            ))
        processor_kwargs.update({
            'text': [rendered],
            'padding': False,
            'return_tensors': 'pt',
        })
        if inputs.images:
            processor_kwargs['images'] = inputs.images
        if inputs.videos:
            processor_kwargs['videos'] = inputs.videos
        if self.is_training:
            processor_kwargs['labels_spans'] = [labels_spans]

        processor_outputs = self.processor(**processor_kwargs)
        encoded = {
            'input_ids': processor_outputs['input_ids'][0].tolist(),
            'pixel_values': processor_outputs['pixel_values'],
            'grid_thw': processor_outputs['grid_thw'],
            'cross_attention_mask': processor_outputs['cross_attention_mask'],
            'media_nums_per_sample': list(processor_outputs['media_nums_per_sample']),
            'vision_chunked_length': vision_chunked_length,
        }
        if self.is_training:
            encoded['labels'] = processor_outputs['labels'][0].tolist()
        else:
            encoded['labels'] = None
        encoded['loss_scale'] = None
        return encoded

    def _vision_signature(self, input_ids: List[int]) -> Tuple[int, ...]:
        token_ids = []
        for token in self._vision_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != self.tokenizer.unk_token_id and token_id not in token_ids:
                token_ids.append(token_id)
        return tuple(input_ids.count(token_id) for token_id in token_ids)

    def _truncate(self, input_ids: List[int], labels: Optional[List[int]], encoded: Dict[str, Any],
                  truncation_strategy: Literal['left', 'right']):
        if truncation_strategy == 'right':
            text_slice = slice(0, self.max_length)
        else:
            text_slice = slice(len(input_ids) - self.max_length, len(input_ids))
        truncated_input_ids = input_ids[text_slice]

        if self._vision_signature(input_ids) != self._vision_signature(truncated_input_ids):
            raise MaxLengthError(
                'MOSS-VL truncation removed vision special/frame tokens while media tensors are still present. '
                'Reduce media size/frame count, increase max_length, or change the truncation boundary.')

        cross_attention_mask = encoded.get('cross_attention_mask')
        if cross_attention_mask is not None:
            if cross_attention_mask.shape[-2] != len(input_ids):
                raise ValueError(
                    'MOSS-VL cross_attention_mask text dimension does not match input_ids before truncation: '
                    f'{cross_attention_mask.shape[-2]} != {len(input_ids)}.')
            encoded['cross_attention_mask'] = cross_attention_mask[..., text_slice, :]

        if labels is not None:
            labels = labels[text_slice]
            if labels:
                labels[0] = -100
        return truncated_input_ids, labels

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        grid_thw = self.concat_tensor(batch, 'grid_thw', 0)
        if grid_thw is not None:
            res['grid_thw'] = grid_thw

        media_nums_per_sample = []
        for row in batch:
            media_nums_per_sample.extend(row.get('media_nums_per_sample') or [])
        if media_nums_per_sample:
            res['media_nums_per_sample'] = media_nums_per_sample

        chunk_lengths = [row.get('vision_chunked_length') for row in batch]
        chunk_lengths = [length for length in chunk_lengths if length is not None]
        if chunk_lengths:
            if len(set(chunk_lengths)) != 1:
                raise ValueError(f'vision_chunked_length must be identical within a batch: {chunk_lengths}.')
            res['vision_chunked_length'] = chunk_lengths[0]
        return res

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        masks = [row.get('cross_attention_mask') for row in batch]
        if not all(mask is not None for mask in masks):
            raise ValueError('Every MOSS-VL sample must contain cross_attention_mask, including text-only samples.')

        target_text_length = res['input_ids'].shape[1]
        target_frame_length = max(mask.shape[-1] for mask in masks)
        padding_right = (self.padding_side if self.is_training else 'left') == 'right'
        padded_masks = []
        for row, mask in zip(batch, masks):
            if mask.ndim == 4:
                if mask.shape[0] != 1:
                    raise ValueError(f'Expected a single-sample cross_attention_mask, got shape={tuple(mask.shape)}.')
                mask = mask[0]
            if mask.shape[-2] != len(row['input_ids']):
                raise ValueError('MOSS-VL cross_attention_mask text dimension does not match input_ids in collator: '
                                 f'{mask.shape[-2]} != {len(row["input_ids"])}.')
            text_padding = target_text_length - mask.shape[-2]
            frame_padding = target_frame_length - mask.shape[-1]
            if padding_right:
                pad = (0, frame_padding, 0, text_padding)
            else:
                pad = (0, frame_padding, text_padding, 0)
            padded_masks.append(F.pad(mask.to(torch.bool), pad, value=True))
        res['cross_attention_mask'] = torch.stack(padded_masks)
        return res


register_template(
    ChatmlTemplateMeta(
        MLLMTemplateType.moss_vl,
        template_cls=MossVLTemplate,
        default_system=None,
        agent_template='hermes',
    ))
