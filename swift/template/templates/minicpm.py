# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import numpy as np
import torch
from dataclasses import dataclass, field
from functools import partial
from torch import nn
from typing import Any, Dict, List, Literal, Optional

from swift.utils import get_env_args
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall
from ..vision_utils import load_audio, load_video_minicpmv_mplug_owl3
from .llama import Llama3TemplateMeta
from .qwen import Qwen2_5TemplateMeta, Qwen3MixedTemplateMeta, QwenTemplateMeta
from .utils import ChatmlTemplateMeta


@dataclass
class MinicpmTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<s>{{SYSTEM}}'])
    prompt: Prompt = field(default_factory=lambda: ['<用户>{{QUERY}}<AI>'])
    chat_sep: Optional[Prompt] = field(default_factory=list)
    suffix: Prompt = field(default_factory=lambda: ['</s>'])


register_template(MinicpmTemplateMeta(LLMTemplateType.minicpm))


def _remove_idx(arr: List[int], idx_list: List[int]) -> List[int]:
    res = []
    idx_set = set(idx_list)
    for i, x in enumerate(arr):
        if i not in idx_set:
            res.append(x)
    return res


class MiniCPMVTemplate(Template):
    is_v2_5 = False
    use_model = True
    skip_prompt = False
    placeholder_tokens = ['<unk>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if self.mode == 'vllm':
            return ['(<image>./</image>)\n']
        else:
            return [[-100]]

    async def prepare_lmdeploy_turbomind_inputs(self, inputs: Dict[str, Any]) -> None:
        images = inputs.pop('images', None) or []
        if len(images) == 0:
            return
        input_ids = inputs['input_ids']
        idx_list = findall(input_ids, -100)
        idx_list.insert(0, -1)
        new_input_ids = []
        features = []
        for i in range(len(idx_list) - 1):
            new_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]]
            context_list = ['<image>', [-100], '</image>']
            feat = [x.squeeze() for x in images[i]['embeddings'].split(1)]
            grid = images[i].get('grid')
            if len(feat) > 1 and grid is not None:
                context_list.append('<slice>')
                for j in range(grid[1]):
                    if j > 0:
                        context_list.append('\n')
                    for _ in range(grid[0]):
                        context_list += ['<image>', [-100], '</image>']
                context_list.append('</slice>\n')
            new_input_ids += self._encode_context_list(context_list)[0]
            features += feat
        new_input_ids += input_ids[idx_list[-1] + 1:]
        inputs['input_ids'] = new_input_ids
        inputs['images'] = features
        await super().prepare_lmdeploy_turbomind_inputs(inputs)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, -100)
        idx = idx_list[0]
        tgt_sizes = None
        slice_mode = getattr(self.config, 'slice_mode', False)
        if slice_mode:
            if self.is_v2_5:
                image_processor = self.processor.image_processor
                image_inputs = image_processor(images, return_tensors='pt').to(self.model_info.torch_dtype)
                placeholder = image_processor.get_slice_image_placeholder(image_inputs.image_sizes[0][0])
                pixel_values = image_inputs['pixel_values']
                tgt_sizes = image_inputs['tgt_sizes']
            else:
                images, placeholder = self.model.get_slice_image_placeholder(images[0], self.processor)
                pixel_values = [[self.model.transform(img) for img in images]]
            placeholder += '\n'
            placeholder_id = self.processor.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            input_tensor_ids = torch.tensor(input_ids)
            image_start_idx = torch.where(input_tensor_ids == self.processor.im_start_id)[0]
            image_start_idx += 1
            image_end_idx = torch.where(input_tensor_ids == self.processor.im_end_id)[0]
            valid_image_nums = max(len(image_start_idx), len(image_end_idx))
            image_bound = [
                torch.hstack(
                    [image_start_idx[:valid_image_nums].unsqueeze(-1), image_end_idx[:valid_image_nums].unsqueeze(-1)])
            ]
        else:
            placeholder = '<image>' + '<unk>' * self.config.query_num + '</image>\n'
            placeholder_id = self.processor.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            image_bound = [torch.tensor([[idx, idx + self.config.query_num]])]
            pixel_values = [[self.model.transform(images[0])]]
        encoded = {
            'input_ids': input_ids,
            'labels': labels,
            'image_bound': image_bound,
            'pixel_values': pixel_values,
            'tgt_sizes': tgt_sizes
        }
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs_embeds, _ = model.get_vllm_embedding(inputs)
        return {'inputs_embeds': inputs_embeds}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = {}
        for k in ['pixel_values', 'image_bound', 'tgt_sizes']:
            res[k] = self.gather_list(batch, k)
        res.update(super()._data_collator(batch, padding_to=padding_to))
        return res


register_template(MinicpmTemplateMeta(MLLMTemplateType.minicpmv, template_cls=MiniCPMVTemplate))


class MiniCPMV2_5Template(MiniCPMVTemplate):
    is_v2_5 = True


register_template(Llama3TemplateMeta(
    MLLMTemplateType.minicpmv2_5,
    template_cls=MiniCPMV2_5Template,
))


class MiniCPMV2_6Template(MiniCPMVTemplate):

    def init_env_args(self):
        super().init_env_args()
        self.max_num_frames = get_env_args('max_num_frames', int, 64)
        self.max_slice_nums = get_env_args('max_slice_nums', int, None)
        self.video_max_slice_nums = get_env_args('video_max_slice_nums', int, 1)  # or 2

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type in {'image', 'video'}
        load_video = partial(load_video_minicpmv_mplug_owl3, max_num_frames=self.max_num_frames)
        image_context = super().replace_tag('image', index, inputs)
        if media_type == 'image':
            return image_context
        elif media_type == 'video':
            return self.replace_video2image(load_video, inputs, lambda i: image_context)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        images = inputs.images
        use_video = bool(inputs.videos)
        use_image_id = True
        max_slice_nums = self.max_slice_nums
        if use_video:
            max_slice_nums = self.video_max_slice_nums
            use_image_id = False
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        idx_list = findall(input_ids, -100)

        image_processor = self.processor.image_processor
        image_inputs = image_processor([images], return_tensors='pt',
                                       max_slice_nums=max_slice_nums).to(self.model_info.torch_dtype)

        def _get_new_tokens(i):
            placeholder = image_processor.get_slice_image_placeholder(
                image_inputs.image_sizes[0][i], image_idx=i, max_slice_nums=max_slice_nums, use_image_id=use_image_id)
            placeholder += '\n'
            return self.processor.encode(placeholder, add_special_tokens=False)

        input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list, _get_new_tokens)

        if inputs.images:
            input_tensor_ids = torch.tensor(input_ids)
            unk_token = self.processor.encode('<unk>', add_special_tokens=False)[0]
            indices = (input_tensor_ids == unk_token).nonzero(as_tuple=True)[0].tolist()

            ranges = []
            start = indices[0]
            for i in range(1, len(indices)):
                if indices[i] != indices[i - 1] + 1:
                    ranges.append([start, indices[i - 1] + 1])
                    start = indices[i]
            ranges.append([start, indices[-1] + 1])
            image_bound = [torch.tensor(ranges)]
        else:
            image_bound = [[]]

        encoded = {
            'input_ids': input_ids,
            'labels': labels,
            'loss_scale': loss_scale,
            'image_bound': image_bound,
            'pixel_values': image_inputs['pixel_values'],
            'tgt_sizes': image_inputs['tgt_sizes']
        }
        return encoded


register_template(QwenTemplateMeta(
    MLLMTemplateType.minicpmv2_6,
    template_cls=MiniCPMV2_6Template,
))

register_template(ChatmlTemplateMeta(
    MLLMTemplateType.minicpmv4,
    template_cls=MiniCPMV2_6Template,
))

register_template(Qwen2_5TemplateMeta(
    MLLMTemplateType.minicpmo,
    template_cls=MiniCPMV2_6Template,
))


class MiniCPMV4_5Template(MiniCPMV2_6Template):

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        images = inputs.images
        use_video = bool(inputs.videos)
        use_image_id = True
        max_slice_nums = self.max_slice_nums
        if use_video:
            max_slice_nums = self.video_max_slice_nums
            use_image_id = False
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        idx_list = findall(input_ids, -100)

        image_processor = self.processor.image_processor
        image_inputs = image_processor([images], return_tensors='pt',
                                       max_slice_nums=max_slice_nums).to(self.model_info.torch_dtype)

        def _get_new_tokens(i):
            placeholder = image_processor.get_slice_image_placeholder(
                image_inputs.image_sizes[0][i], image_idx=i, max_slice_nums=max_slice_nums, use_image_id=use_image_id)
            placeholder += '\n'
            return self.processor.encode(placeholder, add_special_tokens=False)

        input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list, _get_new_tokens)

        if inputs.images:
            input_tensor_ids = torch.tensor(input_ids)
            unk_token = self.processor.encode('<unk>', add_special_tokens=False)[0]
            indices = (input_tensor_ids == unk_token).nonzero(as_tuple=True)[0].tolist()

            ranges = []
            start = indices[0]
            for i in range(1, len(indices)):
                if indices[i] != indices[i - 1] + 1:
                    ranges.append([start, indices[i - 1] + 1])
                    start = indices[i]
            ranges.append([start, indices[-1] + 1])
            image_bound = [torch.tensor(ranges)]
        else:
            image_bound = [[]]

        encoded = {
            'input_ids': input_ids,
            'labels': labels,
            'loss_scale': loss_scale,
            'image_bound': image_bound,
            'pixel_values': image_inputs['pixel_values'],
            'tgt_sizes': image_inputs['tgt_sizes'],
            'temporal_ids': image_inputs['temporal_ids'],
        }
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = {}
        for k in ['pixel_values', 'image_bound', 'tgt_sizes', 'temporal_ids']:
            res[k] = self.gather_list(batch, k)
        res.update(Template._data_collator(self, batch, padding_to=padding_to))
        return res


register_template(
    Qwen3MixedTemplateMeta(
        MLLMTemplateType.minicpmv4_5,
        template_cls=MiniCPMV4_5Template,
        is_thinking=True,
        thinking_prefix='<think>\n',
    ))


class MiniCPMO4_5Template(MiniCPMV4_5Template):
    SAMPLING_RATE = 16000
    MAX_AUDIO_DURATION = 30  # seconds

    def init_env_args(self):
        super().init_env_args()
        self.use_audio_in_video = get_env_args('use_audio_in_video', bool, False)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image' or media_type == 'video' and not self.use_audio_in_video:
            return super().replace_tag(media_type, index, inputs)
        elif media_type == 'audio':
            # Load audio from file path to numpy array at 16kHz
            if isinstance(inputs.audios[index], str):
                inputs.audios[index] = load_audio(inputs.audios[index], sampling_rate=self.SAMPLING_RATE)
            return ['<|audio_start|><|audio_end|>']
        elif media_type == 'video':
            from minicpmo.utils import get_video_frame_audio_segments
            video = inputs.videos[inputs.video_idx]
            video_segments, audio_segments, _ = get_video_frame_audio_segments(
                video, use_audio=self.use_audio_in_video, stack_frames=1)
            # Insert frames into images list at current position
            images = inputs.images
            inputs.images = images[:inputs.image_idx] + video_segments + images[inputs.image_idx:]
            # Build context list
            image_context = [[-100]]
            context_list = []
            if self.use_audio_in_video and audio_segments:
                # Insert audio segments into audios list at current position
                audios = inputs.audios
                inputs.audios = audios[:inputs.audio_idx] + audio_segments + audios[inputs.audio_idx:]
                audio_context = ['<|audio_start|><|audio_end|>']
                # Interleave: one image placeholder + one audio placeholder per second
                for i in range(len(video_segments)):
                    context_list += image_context
                    if i < len(audio_segments):
                        context_list += audio_context
                inputs.audio_idx += len(audio_segments)
            else:
                for _ in range(len(video_segments)):
                    context_list += image_context
            inputs.image_idx += len(video_segments)
            return context_list

    def _get_audio_num_tokens(self, audio_sample_len: int) -> int:
        """Compute the number of <unk> placeholder tokens for an audio of given sample count.

        This mirrors the official get_audio_placeholder logic:
        1. mel frames = ceil(audio_samples / hop_length)
        2. after CNN downsampling: (mel_frames - 1) // 2 + 1
        3. after avg pooling: (cnn_frames - pool_step) // pool_step + 1
        """
        hop_length = self.processor.audio_processor.hop_length  # 160
        pool_step = self.config.audio_pool_step  # 5
        feature_lens = math.ceil(audio_sample_len / hop_length)
        feature_lens_after_cnn = (feature_lens - 1) // 2 + 1
        output_lens = (feature_lens_after_cnn - pool_step) // pool_step + 1
        return output_lens

    def _extract_audio_features(self, audios: List[np.ndarray]):
        """Extract mel features from audio arrays using the WhisperFeatureExtractor.

        Handles chunking of long audios (>30s) into segments.
        Matches the official audio_feature_extract output format.

        Returns:
            audio_features: tensor [N, 80, max_frames] or [] if no audios
            audio_feature_lens: [tensor([l1, l2, ...])] or None
        """
        audio_processor = self.processor.audio_processor
        max_audio_inp_len = self.MAX_AUDIO_DURATION * self.SAMPLING_RATE
        all_audio_features = []
        all_audio_lens = []

        for audio in audios:
            # Chunk long audios at 30s boundaries
            if len(audio) <= max_audio_inp_len:
                chunks = [audio]
            else:
                chunks = [audio[i:i + max_audio_inp_len] for i in range(0, len(audio), max_audio_inp_len)]

            for chunk in chunks:
                audio_input = audio_processor(
                    chunk,
                    sampling_rate=self.SAMPLING_RATE,
                    return_tensors='pt',
                    padding='max_length',
                    return_attention_mask=True,
                )
                feat = audio_input['input_features']  # [1, 80, frames]
                actual_len = audio_input['attention_mask'].sum(dim=1)  # [1]
                feat = feat[:, :, :actual_len[0]]
                all_audio_features.append(feat.squeeze(0))  # [80, actual_frames]
                all_audio_lens.append(actual_len[0])

        if all_audio_features:
            # Pad and stack: [N, 80, max_frames] — same as official processor
            audio_features = torch.nn.utils.rnn.pad_sequence(
                [f.transpose(0, 1) for f in all_audio_features],
                batch_first=True,
                padding_value=0.0,
            ).transpose(1, 2)
            audio_feature_lens = [torch.hstack(all_audio_lens)]
        else:
            audio_features = []
            audio_feature_lens = None

        return audio_features, audio_feature_lens

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        # Step 1: Base encode — produces input_ids with -100 for images
        # and audio_start_id,audio_end_id pairs for audios
        encoded = Template._encode(self, inputs)
        images = inputs.images
        use_video = bool(inputs.videos)
        use_image_id = True
        max_slice_nums = self.max_slice_nums
        if use_video:
            max_slice_nums = self.video_max_slice_nums
            use_image_id = False
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)

        # Step 2: Process images — replace -100 tokens with image placeholders
        idx_list = findall(input_ids, -100)
        image_processor = self.processor.image_processor
        image_inputs = image_processor([images], return_tensors='pt',
                                       max_slice_nums=max_slice_nums).to(self.model_info.torch_dtype)

        def _get_new_tokens(i):
            placeholder = image_processor.get_slice_image_placeholder(
                image_inputs.image_sizes[0][i], image_idx=i, max_slice_nums=max_slice_nums, use_image_id=use_image_id)
            placeholder += '\n'
            return self.processor.encode(placeholder, add_special_tokens=False)

        input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list, _get_new_tokens)

        # Step 3: Process audios — expand audio_start/audio_end pairs with <unk> tokens
        tokenizer = self.processor.tokenizer
        audio_start_id = tokenizer.convert_tokens_to_ids('<|audio_start|>')
        audio_end_id = tokenizer.convert_tokens_to_ids('<|audio_end|>')
        unk_token_id = tokenizer.convert_tokens_to_ids('<unk>')

        audio_features = None
        audio_feature_lens = None

        if inputs.audios:
            audios = inputs.audios
            audio_features, audio_feature_lens = self._extract_audio_features(audios)

            # Find empty audio placeholder pairs (audio_start_id immediately followed by audio_end_id)
            audio_placeholder_positions = []
            for i in range(len(input_ids) - 1):
                if input_ids[i] == audio_start_id and input_ids[i + 1] == audio_end_id:
                    audio_placeholder_positions.append(i)

            assert len(audio_placeholder_positions) == len(audios), \
                f'Found {len(audio_placeholder_positions)} audio placeholders but have {len(audios)} audios'

            # Expand each audio placeholder with <unk> tokens
            offset = 0
            for i, audio in enumerate(audios):
                num_tokens = self._get_audio_num_tokens(len(audio))
                unk_tokens = [unk_token_id] * num_tokens
                pos = audio_placeholder_positions[i] + offset
                # Current: [..., audio_start_id, audio_end_id, ...]
                # Target:  [..., audio_start_id, unk*N, audio_end_id, ...]
                input_ids = input_ids[:pos + 1] + unk_tokens + input_ids[pos + 1:]
                if labels is not None:
                    labels = labels[:pos + 1] + [-100] * num_tokens + labels[pos + 1:]
                if loss_scale is not None:
                    scale_val = loss_scale[pos]
                    loss_scale = loss_scale[:pos + 1] + [scale_val] * num_tokens + loss_scale[pos + 1:]
                offset += num_tokens

        # Step 4: Compute image_bound using start/end token boundaries
        # This is more robust than finding consecutive <unk> tokens, especially
        # when both image and audio use <unk> as placeholder.
        input_tensor_ids = torch.tensor(input_ids)

        if images:
            im_start_id = tokenizer.convert_tokens_to_ids('<image>')
            im_end_id = tokenizer.convert_tokens_to_ids('</image>')
            slice_start_id = tokenizer.convert_tokens_to_ids('<slice>')
            slice_end_id = tokenizer.convert_tokens_to_ids('</slice>')

            start_cond = (input_tensor_ids == im_start_id) | (input_tensor_ids == slice_start_id)
            end_cond = (input_tensor_ids == im_end_id) | (input_tensor_ids == slice_end_id)
            image_start_idx = torch.where(start_cond)[0] + 1
            image_end_idx = torch.where(end_cond)[0]
            valid_image_nums = min(len(image_start_idx), len(image_end_idx))
            image_bound = [
                torch.hstack([
                    image_start_idx[:valid_image_nums].unsqueeze(-1),
                    image_end_idx[:valid_image_nums].unsqueeze(-1),
                ])
            ]
        else:
            image_bound = [[]]

        # Step 5: Compute audio_bounds
        if inputs.audios:
            audio_start_idx = torch.where(input_tensor_ids == audio_start_id)[0]
            audio_end_idx = torch.where(input_tensor_ids == audio_end_id)[0]
            assert len(audio_start_idx) == len(audio_end_idx)
            audio_bounds = [torch.hstack([
                (audio_start_idx + 1).unsqueeze(-1),
                audio_end_idx.unsqueeze(-1),
            ])]
        else:
            audio_bounds = [[]]

        encoded = {
            'input_ids': input_ids,
            'labels': labels,
            'loss_scale': loss_scale,
            'image_bound': image_bound,
            'pixel_values': image_inputs['pixel_values'],
            'tgt_sizes': image_inputs['tgt_sizes'],
            'audio_features': audio_features,
            'audio_feature_lens': audio_feature_lens,
            'audio_bounds': audio_bounds,
        }
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Vision embeddings
        inputs_embeds, _ = model.get_vllm_embedding(inputs)
        # Audio embeddings — scatter audio features into the embedding
        inputs_embeds = model.get_omni_embedding(
            inputs,
            input_embeddings=inputs_embeds,
            chunk_length=getattr(self.config, 'audio_chunk_length', 1.0),
        )
        return {'inputs_embeds': inputs_embeds}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = {}
        # Vision data
        for k in ['pixel_values', 'image_bound', 'tgt_sizes']:
            res[k] = self.gather_list(batch, k)

        # Audio data — collect from all samples
        all_audio_feats = []  # list of [N_i, 80, frames_i] tensors
        audio_feature_lens_list = []
        audio_bounds_list = []

        for b in batch:
            af = b.pop('audio_features', None)
            afl = b.pop('audio_feature_lens', None)
            ab = b.pop('audio_bounds', None)

            if af is not None and isinstance(af, torch.Tensor):
                all_audio_feats.append(af)
            if afl is not None:
                audio_feature_lens_list.extend(afl)
            if ab is not None:
                audio_bounds_list.extend(ab)

        # Re-pad audio features across the batch to the same max frame length
        if all_audio_feats:
            # Unpack per-sample tensors into individual segments, then re-pad
            segments = []
            for af in all_audio_feats:
                for i in range(af.shape[0]):
                    segments.append(af[i])  # [80, frames_i]
            res['audio_features'] = torch.nn.utils.rnn.pad_sequence(
                [s.transpose(0, 1) for s in segments],
                batch_first=True,
                padding_value=0.0,
            ).transpose(1, 2)  # [total_segments, 80, max_frames]
        else:
            res['audio_features'] = []

        res['audio_feature_lens'] = audio_feature_lens_list if audio_feature_lens_list else []
        res['audio_bounds'] = audio_bounds_list if audio_bounds_list else []

        res.update(Template._data_collator(self, batch, padding_to=padding_to))
        return res


register_template(
    Qwen3MixedTemplateMeta(
        MLLMTemplateType.minicpmo4_5,
        template_cls=MiniCPMO4_5Template,
        is_thinking=True,
    ))


class MiniCPMV4_6Template(MiniCPMV2_6Template):
    support_padding_free = True

    def init_env_args(self):
        super().init_env_args()
        self.downsample_mode = get_env_args('downsample_mode', str, None)

    def _data_collator(self, batch, *, padding_to=None):
        """Collect per-sample pixel_values / tgt_sizes intact (no dim-0 flatten)
        so that _post_encode can concatenate them along the NaViT axis."""
        res = {}
        for k in ['pixel_values', 'tgt_sizes']:
            items = [b.pop(k) for b in batch if b.get(k) is not None]
            if items:
                res[k] = items
        res.update(Template._data_collator(self, batch, padding_to=padding_to))
        return res

    def _post_encode(self, model, inputs):
        """Build inputs_embeds for v4.6.  Concatenates pixel_values from all
        samples (or packed segments) into a single NaViT tensor so that the
        vision tower is called only once, avoiding DeepSpeed ZeRO errors about
        gradients being reduced twice for the same parameter."""
        pixel_values = inputs.get('pixel_values')
        tgt_sizes = inputs.pop('tgt_sizes', None)
        input_ids = inputs.get('input_ids')

        if isinstance(pixel_values, list) and len(pixel_values) > 0 and input_ids is not None:
            embed_fn = model.model.get_input_embeddings()
            inputs_embeds = embed_fn(input_ids)
            image_token_id = getattr(model.model.config, 'image_token_id', None)
            if image_token_id is None:
                image_token_id = getattr(self.processor, 'image_token_id', None)

            # Select valid samples (those with both pixel_values and tgt_sizes)
            valid_pvs, valid_ts = [], []
            for pv, ts in zip(pixel_values, tgt_sizes or []):
                if pv is not None and pv.numel() > 0 and ts is not None:
                    valid_pvs.append(pv)
                    valid_ts.append(ts)

            if valid_pvs and image_token_id is not None:
                combined_pv = torch.cat(valid_pvs, dim=-1)
                combined_ts = torch.cat(valid_ts, dim=0)
                vision_output = model.model.get_image_features(
                    combined_pv,
                    combined_ts,
                    downsample_mode=getattr(self, 'downsample_mode', None),
                )
                features = torch.cat(
                    vision_output.pooler_output, dim=0).to(
                        device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                mask = model.model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds,
                    features,
                    image_token_id,
                )
                inputs_embeds = inputs_embeds.masked_scatter(mask, features)

            inputs['inputs_embeds'] = inputs_embeds
            inputs.pop('input_ids', None)
            inputs.pop('pixel_values', None)
        elif isinstance(pixel_values, torch.Tensor):
            inputs['target_sizes'] = tgt_sizes
            inputs['pixel_values'] = pixel_values

        return inputs

    def _unwrap_batched_grids(self, grids: Any) -> Any:
        if isinstance(grids, list) and grids and isinstance(grids[0], list):
            if grids[0] and isinstance(grids[0][0], list):
                return grids[0]
        return grids

    def _unwrap_batched_list(self, value: Any) -> Any:
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            return value[0]
        return value

    def _build_v4_6_placeholder(
        self,
        image_inputs: Dict[str, Any],
        image_idx: int,
        use_image_id: bool,
        max_slice_nums: Optional[int],
    ) -> str:
        image_processor = self.processor.image_processor
        if hasattr(image_processor, 'get_slice_image_placeholder'):
            grids = self._unwrap_batched_grids(image_inputs['grids'])
            source_tokens = self._unwrap_batched_list(image_inputs['source_image_visual_tokens'])
            patch_tokens = self._unwrap_batched_list(image_inputs['patch_visual_tokens'])
            return image_processor.get_slice_image_placeholder(
                grids[image_idx],
                image_idx=image_idx,
                max_slice_nums=max_slice_nums,
                use_image_id=use_image_id,
                source_image_visual_tokens=source_tokens[image_idx],
                patch_visual_tokens=patch_tokens[image_idx],
            )

        grids = self._unwrap_batched_grids(image_inputs['grids'])
        num_patches_per_image = image_inputs['num_patches_per_image']
        target_sizes = image_inputs['target_sizes']
        flat_index = 0
        for idx in range(image_idx):
            flat_index += num_patches_per_image[idx]
        n_patches = num_patches_per_image[image_idx]
        img_target_sizes = target_sizes[flat_index:flat_index + n_patches]

        downsample_mode = self.downsample_mode or getattr(image_processor, 'downsample_mode', None)
        token_divisor = 4 if downsample_mode == '4x' else 16
        num_tokens_per_patch = img_target_sizes.prod(-1) // token_divisor
        num_rows, num_cols = grids[image_idx]

        image_start = getattr(self.processor, 'image_start_token', '<image>')
        image_end = getattr(self.processor, 'image_end_token', '</image>')
        slice_start = getattr(self.processor, 'slice_start_token', '<slice>')
        slice_end = getattr(self.processor, 'slice_end_token', '</slice>')
        image_id_start = getattr(self.processor, 'image_id_start_token', '<image_id>')
        image_id_end = getattr(self.processor, 'image_id_end_token', '</image_id>')
        image_token = (
            getattr(self.processor, 'image_token', None)
            or getattr(getattr(self.processor, 'tokenizer', None), 'image_token', None) or '<image>')

        image_placeholder = image_start + '<|placeholder|>' * int(num_tokens_per_patch[0]) + image_end
        if use_image_id:
            image_placeholder = f'{image_id_start}{image_idx}{image_id_end}' + image_placeholder

        slice_mode = getattr(self.processor, 'slice_mode', True)
        if slice_mode and num_rows > 0 and num_cols > 0:
            per_slice_tokens = int(num_tokens_per_patch[1]) if len(num_tokens_per_patch) > 1 else 0
            slice_placeholder = slice_start + '<|placeholder|>' * per_slice_tokens + slice_end
            slices = [slice_placeholder * num_cols for _ in range(num_rows)]
            image_placeholder += '\n'.join(slices)

        return image_placeholder.replace('<|placeholder|>', image_token)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        images = inputs.images
        use_video = bool(inputs.videos)
        use_image_id = True
        max_slice_nums = self.max_slice_nums
        if use_video:
            max_slice_nums = self.video_max_slice_nums
            use_image_id = False
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        idx_list = findall(input_ids, -100)

        image_processor = self.processor.image_processor
        image_kwargs = {
            'return_tensors': 'pt',
            'downsample_mode': self.downsample_mode,
        }
        if max_slice_nums is not None:
            image_kwargs['max_slice_nums'] = max_slice_nums

        if hasattr(image_processor, 'valid_kwargs'):
            image_inputs = image_processor(images, **image_kwargs).to(self.model_info.torch_dtype)
        else:
            image_inputs = image_processor([images], **image_kwargs).to(self.model_info.torch_dtype)

        def _get_new_tokens(i):
            placeholder = self._build_v4_6_placeholder(image_inputs, i, use_image_id, max_slice_nums)
            placeholder += '\n'
            return self.processor.encode(placeholder, add_special_tokens=False)

        input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list, _get_new_tokens)

        if inputs.images:
            input_tensor_ids = torch.tensor(input_ids)
            unk_token = self.processor.image_token_id
            indices = (input_tensor_ids == unk_token).nonzero(as_tuple=True)[0].tolist()

            ranges = []
            start = indices[0]
            for i in range(1, len(indices)):
                if indices[i] != indices[i - 1] + 1:
                    ranges.append([start, indices[i - 1] + 1])
                    start = indices[i]
            ranges.append([start, indices[-1] + 1])
            image_bound = [torch.tensor(ranges)]
        else:
            image_bound = [[]]

        tgt_sizes = image_inputs.get('tgt_sizes')
        if tgt_sizes is None:
            tgt_sizes = image_inputs.get('target_sizes')

        encoded = {
            'input_ids': input_ids,
            'labels': labels,
            'loss_scale': loss_scale,
            'image_bound': image_bound,
            'pixel_values': image_inputs['pixel_values'],
            'tgt_sizes': tgt_sizes,
        }
        return encoded


register_template(ChatmlTemplateMeta(
    MLLMTemplateType.minicpmv4_6,
    template_cls=MiniCPMV4_6Template,
))
