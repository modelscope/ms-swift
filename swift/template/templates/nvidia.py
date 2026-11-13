# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from dataclasses import asdict
from PIL import Image

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..utils import findall
from ..vision_utils import load_audio, load_image, load_video_hf
from .qwen import QwenTemplateMeta


def _nemotron_transcribe_audio_tokens(audio_mask, subsampling_factor):
    # Matches Nemotron3_5TranscribeAudioModel._get_feat_extract_output_lengths.
    return (audio_mask.sum(-1) + subsampling_factor - 1) // subsampling_factor


class NemoSpeechLMTemplate(Template):
    placeholder_tokens = ['<|audio|>']
    support_padding_free = False

    def replace_tag(self, media_type, index, inputs):
        if media_type != 'audio':
            raise ValueError(f'Nemotron Transcribe does not support {media_type}.')
        return ['<|audio|>']

    def _encode(self, inputs):
        encoded = super()._encode(inputs)
        if not inputs.audios:
            return encoded
        feature_extractor = self.processor.feature_extractor
        sampling_rate = feature_extractor.sampling_rate
        audios = [
            audio if isinstance(audio, (np.ndarray, torch.Tensor)) else load_audio(audio, sampling_rate=sampling_rate)
            for audio in inputs.audios
        ]
        audios = [torch.as_tensor(audio, dtype=torch.float32).squeeze() for audio in audios]
        if any(audio.ndim != 1 or audio.numel() == 0 for audio in audios):
            raise ValueError('Nemotron Transcribe expects non-empty mono audio waveforms.')

        audio_inputs = feature_extractor(
            audios, sampling_rate=sampling_rate, return_attention_mask=True, return_tensors='pt', padding=True)
        input_features = audio_inputs['input_features']
        input_features_mask = audio_inputs.get('attention_mask')
        if input_features_mask is None:
            input_features_mask = torch.ones(input_features.shape[:2], dtype=torch.bool)
        input_features_mask = input_features_mask.bool()
        subsampling_factor = int(self.processor.subsampling_factor)
        padding = -input_features.shape[1] % subsampling_factor
        if padding:
            input_features = F.pad(input_features, (0, 0, 0, padding))
            input_features_mask = F.pad(input_features_mask, (0, padding))
        audio_token_counts = _nemotron_transcribe_audio_tokens(input_features_mask, subsampling_factor)

        audio_token_id = self._tokenize('<|audio|>')
        indices = findall(encoded['input_ids'], audio_token_id)
        if len(indices) != len(audios):
            raise ValueError(f'Audio placeholders ({len(indices)}) != audio items ({len(audios)}).')

        def _get_new_audio_tokens(index):
            return audio_token_id * int(audio_token_counts[index])

        input_ids, labels, loss_scale = self._extend_tokens(encoded['input_ids'], encoded['labels'],
                                                            encoded.get('loss_scale'), indices, _get_new_audio_tokens)
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        encoded['input_features'] = input_features
        encoded['input_features_mask'] = input_features_mask
        if (inputs.is_multimodal and self.truncation_strategy == 'split' and self.max_length is not None
                and len(input_ids) > self.max_length):
            raise ValueError('Nemotron Transcribe audio cannot use truncation_strategy=split; '
                             'use raise, left or right.')
        return encoded

    def _data_collator_mm_data(self, batch):
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        input_features_mask = [b['input_features_mask'] for b in batch if b.get('input_features_mask') is not None]
        if not input_features:
            return {}
        max_frames = max(feature.shape[1] for feature in input_features)
        input_features = [F.pad(feature, (0, 0, 0, max_frames - feature.shape[1])) for feature in input_features]
        input_features_mask = [F.pad(mask, (0, max_frames - mask.shape[1])) for mask in input_features_mask]
        return {
            'input_features': torch.cat(input_features, dim=0),
            'input_features_mask': torch.cat(input_features_mask, dim=0),
        }


class NemotronHOmniTemplate(Template):
    # Private markers exist only during expansion. Both modalities become official token 18.
    image_sentinel = -200
    video_sentinel = -201
    image_patch = -202
    video_patch = -203
    placeholder_tokens = ['<image>']

    def __init__(self, *args, enable_thinking=True, **kwargs):
        super().__init__(*args, enable_thinking=True if enable_thinking is None else enable_thinking, **kwargs)

    @staticmethod
    def _normalize_messages(messages):
        # Run before StdTemplateInputs discards structured content and reasoning on tool calls.
        media = {'images': [], 'videos': [], 'audios': []}
        for message in messages:
            content = message.get('content')
            if isinstance(content, list) and all(isinstance(part, dict) for part in content):
                text = ''.join(part.get('text') or '' for part in content if part['type'] in {'text', 'input_text'})
                image_count = sum(part['type'] in {'image', 'image_url', 'input_image'} for part in content)
                index, rendered = 0, ''
                for part in content:
                    kind = part['type'].removeprefix('input_').removesuffix('_url')
                    if kind == 'text':
                        rendered += part.get('text') or ''
                    elif kind in {'image', 'video', 'audio'}:
                        value = part.get(part['type'], part.get(kind))
                        if value is None:
                            value = part.get(kind + '_url')
                        if isinstance(value, dict):
                            value = value['url']
                        if value is not None:
                            media[kind + 's'].append(value)
                        if kind == 'image':
                            index += 1
                        if f'<{kind}>' not in text:
                            if kind == 'image' and image_count > 1:
                                rendered += f'<image {index}>'
                            rendered += f'<{kind}>'
                message['content'] = rendered
            reasoning = message.pop('reasoning_content', None)
            if message['role'] == 'assistant' and isinstance(reasoning, str) and reasoning.strip():
                message['content'] = '<think>\n' + reasoning + '</think>' + (message.get('content') or '')
        return media

    def encode(self, inputs, **kwargs):
        from swift.infer_engine import InferRequest
        if isinstance(inputs, InferRequest):
            inputs = asdict(inputs)
        if isinstance(inputs, dict):
            inputs = deepcopy(inputs)
            for key, messages in list(inputs.items()):
                if key == 'messages' or key.endswith('_messages'):
                    media = self._normalize_messages(messages)
                    for kind, values in media.items():
                        if values:
                            target = key[:-len('messages')] + kind
                            if inputs.get(target):
                                raise ValueError('Do not combine structured media with top-level media lists.')
                            inputs[target] = values
        return super().encode(inputs, **kwargs)

    def replace_tag(self, media_type, index, inputs):
        if self.mode in {'vllm', 'sglang', 'lmdeploy'}:
            raise NotImplementedError('Nemotron media ownership currently supports the HF and Megatron backends only.')
        if media_type == 'image':
            return [[self.image_sentinel]]
        if media_type == 'video':
            return [[self.video_sentinel]]
        raise ValueError(f'nemotron_h_omni does not support {media_type}.')

    @staticmethod
    def _load_video(video):
        from transformers.video_utils import VideoMetadata
        metadata = None
        if isinstance(video, dict):
            metadata = video.get('video_metadata')
            video = video['video']
        if isinstance(video, str):
            videos, loaded_metadata = load_video_hf([video])
            video = videos[0]
            if metadata is None:
                metadata = loaded_metadata[0]
        # HF returns [None, ...] for decoded arrays. Only real clip metadata carries timestamps.
        while isinstance(metadata, (list, tuple)):
            if all(item is None for item in metadata):
                metadata = None
            elif len(metadata) == 1:
                metadata = metadata[0]
            else:
                raise ValueError('Expected one VideoMetadata per clip, not per-frame metadata.')
        if isinstance(metadata, dict):
            metadata = VideoMetadata(**metadata)
        if metadata is not None and not isinstance(metadata, VideoMetadata):
            raise TypeError('video_metadata must be VideoMetadata, a metadata dict, or None.')
        frames = []
        for frame in video:
            if isinstance(frame, (str, Image.Image)):
                frame = load_image(frame)
            else:
                if isinstance(frame, torch.Tensor):
                    frame = frame.detach().cpu().numpy()
                frame = np.asarray(frame)
                if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
                    frame = frame.transpose(1, 2, 0)
                frame = Image.fromarray(frame).convert('RGB')
            frames.append(frame)
        if not frames:
            raise ValueError('A video clip must contain at least one frame.')
        if len({frame.size for frame in frames}) != 1:
            raise ValueError('Frames within one clip must have the same size; different clips may differ.')
        return frames, metadata

    def _encode(self, inputs):
        encoded = super()._encode(inputs)
        ids, labels = encoded['input_ids'], encoded['labels']
        encoded.pop('loss_scale', None)
        processor = self.processor
        for kind, sentinel, patch in [('image', self.image_sentinel, self.image_patch),
                                      ('video', self.video_sentinel, self.video_patch)]:
            media = getattr(inputs, kind + 's')
            indices = findall(ids, sentinel)
            if len(indices) != len(media):
                raise ValueError(f'{kind} placeholders ({len(indices)}) != media items ({len(media)}).')
            if not media:
                continue
            replacements, pixels = [], []
            for item in media:
                if kind == 'image':
                    result = processor.image_processor(images=[item])
                    tokens = self._tokenize(processor.image_start_token
                                            + processor.image_token * int(result['num_tokens'][0])
                                            + processor.image_end_token)
                    values = result['pixel_values']
                else:
                    frames, metadata = self._load_video(item)
                    result = processor(text=[processor.video_token], videos=[frames], video_metadata=metadata)
                    tokens = result['input_ids'][0]
                    values = result['pixel_values_videos']
                if not isinstance(values, torch.Tensor):
                    values = torch.stack([torch.as_tensor(value) for value in values])
                if values.ndim != 4 or values.shape[1] != 3:
                    raise ValueError(f'Expected [frames, 3, H, W] pixels, got {values.shape}.')
                pixels.append(values)
                replacements.append([patch if token == processor.image_token_id else token for token in tokens])
            ids, labels, _ = self._extend_tokens(ids, labels, None, indices, replacements.__getitem__)
            encoded['pixel_values' if kind == 'image' else 'pixel_values_videos'] = pixels
        # Use the existing sequence-aligned field, which base packing/truncation/CP/padding preserve.
        encoded['mm_token_type_ids'] = torch.tensor(
            [1 if token == self.image_patch else 2 if token == self.video_patch else 0 for token in ids])
        encoded['input_ids'] = [
            processor.image_token_id if token in (self.image_patch, self.video_patch) else token for token in ids
        ]
        encoded['labels'] = labels
        if (inputs.is_multimodal and self.truncation_strategy == 'split' and self.max_length is not None
                and len(ids) > self.max_length):
            raise ValueError('Nemotron media cannot use truncation_strategy=split; use raise, left or right.')
        return encoded

    def _data_collator_mm_data(self, batch):
        result = {}
        for key in ('pixel_values', 'pixel_values_videos'):
            values = [value for row in batch for value in row.get(key, [])]
            if values:
                result[key] = values
        return result

    def _data_collator(self, batch, *, padding_to=None):
        for row in batch:
            row.pop('loss_scale', None)
            if 'mm_token_type_ids' not in row:
                # Plain-text split truncation does not retain auxiliary sequence fields.
                if row.get('pixel_values') or row.get('pixel_values_videos'):
                    raise ValueError('Nemotron cached media rows require mm_token_type_ids; re-encode the dataset.')
                row['mm_token_type_ids'] = torch.zeros(len(row['input_ids']), dtype=torch.long)
        result = super()._data_collator(batch, padding_to=padding_to)
        types = result.pop('mm_token_type_ids')
        if types.shape != result['input_ids'].shape:
            raise ValueError('Nemotron media ownership must align with final input_ids.')
        result['image_token_mask'] = types == 1
        result['video_token_mask'] = types == 2
        return result


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.nemo_speechlm,
        template_cls=NemoSpeechLMTemplate,
        default_system=None,
        is_thinking=True,
        thinking_prefix='<think>\n',
        non_thinking_prefix='<think></think>',
        history_thinking_prefix='<think></think>',
        stop_words=['<|im_end|>'],
        agent_template='hermes',
    ))

register_template(
    QwenTemplateMeta(
        MLLMTemplateType.nemotron_h_omni,
        template_cls=NemotronHOmniTemplate,
        default_system=None,
        is_thinking=True,
        thinking_prefix='<think>\n',
        non_thinking_prefix='<think></think>',
        history_thinking_prefix='<think></think>',
        stop_words=['<|im_end|>'],
        agent_template='hermes',
    ))
