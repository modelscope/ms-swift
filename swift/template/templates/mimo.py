# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, Word, findall
from ..vision_utils import load_audio
from .utils import ChatmlTemplateMeta


@dataclass
class MiMoV2TemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = 'You are MiMo, a helpful AI assistant engineered by Xiaomi.'
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ['<|im_start|>user\n{{QUERY}}<|im_end|><|im_start|>assistant\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|im_start|>system\n{{SYSTEM}}<|im_end|>'])


def _wav_to_log_mel(wav: torch.Tensor, *, sr, n_mels, n_fft, hop, win, f_min, f_max) -> torch.Tensor:
    from torchaudio.transforms import MelSpectrogram

    mel_transform = MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        power=1.0,
        center=True,
    )
    spec = mel_transform(wav[None, :])
    return torch.log(torch.clip(spec, min=1e-7)).squeeze().transpose(0, 1)


def _pad_codes_to_group(codes: torch.Tensor, group_size: int, audio_channels: int) -> torch.Tensor:
    codes = codes[:, :audio_channels]
    T = codes.shape[0]
    pad_T = ((T + group_size - 1) // group_size) * group_size
    if pad_T > T:
        codes = torch.cat([codes, codes[-1:].expand(pad_T - T, -1)], dim=0)
    return codes


class MiMoV2Template(Template):
    image_token_id = 151655
    video_token_id = 151656
    audio_token_id = 151669
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>', '<|audio_pad|>']
    norm_bbox = 'none'
    use_model = True

    def init_env_args(self) -> None:
        super().init_env_args()
        pc = getattr(self.config, 'processor_config', None) or {}
        self.audio_sampling_rate = pc.get('audio_sampling_rate', 24000)

    @property
    def _audio_cfg(self) -> dict:
        return getattr(self.config, 'audio_config', None) or {}

    def replace_tag(self, media_type, index, inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image, fetch_video

        assert media_type in {'image', 'video', 'audio'}
        if media_type == 'audio':
            return ['<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>']

        kwargs = {'image_patch_size': self.processor.image_processor.patch_size}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index], **inputs.chat_template_kwargs}, **kwargs)
            if self.mode == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            if self.mode == 'sglang':
                return ['<|vision_start|><|video_pad|><|vision_end|>']
            video = inputs.videos[index]
            video_inputs = {'video': video, **inputs.chat_template_kwargs}
            if isinstance(video, list):
                from qwen_vl_utils import vision_process
                video_inputs['sample_fps'] = vision_process.FPS
            video, _ = fetch_video(video_inputs, return_video_sample_fps=True)
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            return ['<|vision_start|><|video_pad|><|vision_end|>']

    def _encode_truncated(self, inputs: StdTemplateInputs):
        encoded = super()._encode_truncated(inputs)
        if self.mode == 'sglang':
            batched = encoded if isinstance(encoded, list) else [encoded]
            for item in batched:
                for old, new in [('images', 'image_data'), ('audios', 'audio_data'), ('videos', 'video_data')]:
                    if old in item:
                        item[new] = item.pop(old)
                for key in ['labels', 'loss_scale', 'channel']:
                    item.pop(key, None)
                mm_keys = {'image_data', 'audio_data', 'video_data'}
                if item.keys() & mm_keys and 'input_ids' in item:
                    ids = item['input_ids']
                    if hasattr(ids, 'tolist'):
                        ids = ids.tolist()
                    item['prompt'] = self.tokenizer.decode(ids, skip_special_tokens=False)
        return encoded

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)

        if inputs.audios:
            audio_tokenizer = self.model.audio_tokenizer
            at_config = audio_tokenizer.config
            group_size = self._audio_cfg.get('group_size', 4)
            audio_channels = self._audio_cfg.get('audio_channels', 20)
            segment_size = self._audio_cfg.get('audio_segment_size', 6000)

            # wav → mel
            mels = [
                _wav_to_log_mel(
                    torch.from_numpy(load_audio(p, self.audio_sampling_rate)).float(),
                    sr=at_config.sampling_rate,
                    n_mels=at_config.n_mels,
                    n_fft=at_config.nfft,
                    hop=at_config.hop_length,
                    win=at_config.window_size,
                    f_min=at_config.fmin,
                    f_max=at_config.fmax,
                ) for p in inputs.audios
            ]

            # mel → codes
            tokenize_audio_batch = sys.modules[type(self.model).__module__].tokenize_audio_batch
            code_list = tokenize_audio_batch(
                mels,
                audio_tokenizer.encoder,
                segment_size=segment_size,
                device=audio_tokenizer.device,
            )

            # pad per-audio so placeholder count matches model's per-group output
            padded_codes_list = [_pad_codes_to_group(c, group_size, audio_channels) for c in code_list]

            idx_list = findall(input_ids, self.audio_token_id)

            def _get_new_audio_tokens(i):
                return [self.audio_token_id] * (padded_codes_list[i].shape[0] // group_size)

            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                _get_new_audio_tokens)
            encoded['audio_codes'] = torch.cat(padded_codes_list, dim=0)

        for media_type in ['images', 'videos']:
            mm_data = getattr(inputs, media_type)
            if not mm_data:
                continue
            if media_type == 'images':
                media_token = self.image_token_id
                media_inputs = processor.image_processor(images=mm_data, return_tensors='pt', do_resize=False)
                media_grid_thw = media_inputs['image_grid_thw']
            else:
                if hasattr(processor, 'video_processor'):
                    processor_func = processor.video_processor
                else:
                    processor_func = processor.image_processor
                media_inputs = processor_func(videos=mm_data, return_tensors='pt', do_resize=False)
                media_grid_thw = media_inputs['video_grid_thw']
                media_token = self.video_token_id

            idx_list = findall(input_ids, media_token)
            merge_length = processor.image_processor.merge_size**2

            def _get_new_tokens(i):
                return [media_token] * (media_grid_thw[i].prod() // merge_length)

            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                _get_new_tokens)
            encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        if 'pixel_values_videos' in res:
            res['video_pixel_values'] = res.pop('pixel_values_videos')
        audio_codes = [b['audio_codes'] for b in batch if b.get('audio_codes') is not None]
        if audio_codes:
            res['audio_codes'] = torch.cat(audio_codes, dim=0)
        return res


register_template(MiMoV2TemplateMeta(MLLMTemplateType.mimo_v2, template_cls=MiMoV2Template))
