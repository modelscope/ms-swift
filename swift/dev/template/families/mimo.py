# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from ..base import Template
from ..inputs import StdTemplateInputs
from ..media import load_audio
from ..register import register_template
from ..utils import findall
from .utils import ChatmlTemplateMeta


@dataclass
class MiMoV2TemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = "You are MiMo, a helpful AI assistant engineered by Xiaomi."
    auto_add_bos: bool = False
    stop_words: List = field(default_factory=lambda: ["<|endoftext|>"])
    prefix: List = field(default_factory=list)
    prompt: List = field(default_factory=lambda: ["<|im_start|>user\n{{QUERY}}<|im_end|>"])
    chat_sep: Optional[List] = field(default_factory=lambda: ["<|im_end|>"])
    suffix: List = field(default_factory=lambda: ["<|im_end|>"])
    system_prefix: Optional[List] = field(default_factory=lambda: ["<|im_start|>system\n{{SYSTEM}}<|im_end|>"])


class MiMoV2Template(Template):
    image_token_id = 151655
    video_token_id = 151656
    audio_token_id = 151669
    placeholder_tokens = ["<|image_pad|>", "<|video_pad|>", "<|audio_pad|>"]
    norm_bbox = "none"
    use_model = True
    support_padding_free = True

    def _get_model(self):
        if self.model is not None:
            return self.get_base_model(self.model)
        if self.dummy_model is None:
            from ...model.loader.base import LoaderModelInfo
            from ...model.loader.mimo import MiMoV2Loader

            loader = MiMoV2Loader(LoaderModelInfo(model_type='mimo_v2', model_dir=self.model_info.model_dir))
            with torch.device('meta'):
                model = loader.build_model_from_config(self.config, self.processor, attn_implementation='eager')
            self.dummy_model = loader.process_model(model)
        return self.dummy_model

    def init_env_args(self):
        super().init_env_args()
        config = getattr(self.config, "processor_config", None) or {}
        self.audio_sampling_rate = config.get("audio_sampling_rate", 24000)

    @staticmethod
    def _wav_to_log_mel(wav, config):
        from torchaudio.transforms import MelSpectrogram

        transform = MelSpectrogram(
            sample_rate=config.sampling_rate, n_fft=config.nfft, hop_length=config.hop_length,
            win_length=config.window_size, f_min=config.fmin, f_max=config.fmax, n_mels=config.n_mels,
            power=1.0, center=True,
        )
        return transform(wav[None, :]).clamp_min(1e-7).log().squeeze(0).transpose(0, 1)

    @staticmethod
    def _pad_codes_to_group(codes, group_size, audio_channels):
        codes = codes[:, :audio_channels]
        if codes.shape[0] == 0:
            raise ValueError("MiMo audio tokenizer returned an empty code sequence.")
        padding = -codes.shape[0] % group_size
        if padding:
            codes = torch.cat([codes, codes[-1:].expand(padding, -1)], dim=0)
        return codes

    def replace_tag(self, media_type, index, inputs: StdTemplateInputs):
        from qwen_vl_utils import fetch_image, fetch_video

        if media_type == "audio":
            return ["<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>"]
        if media_type == "image":
            inputs.images[index] = fetch_image(
                {"image": inputs.images[index], **inputs.chat_template_kwargs},
                image_patch_size=self.processor.image_processor.patch_size,
            )
            inputs.mm_processor_kwargs["do_resize"] = False
            return ["<|vision_start|><|image_pad|><|vision_end|>"]
        if media_type != "video":
            raise ValueError(f"MiMo-V2 does not support {media_type}.")
        video = inputs.videos[index]
        options = {"video": video, **inputs.chat_template_kwargs}
        if isinstance(video, list):
            from qwen_vl_utils import vision_process

            options["sample_fps"] = vision_process.FPS
        video, _ = fetch_video(options, return_video_sample_fps=True)
        inputs.videos[index] = video.to(torch.uint8) if isinstance(video, torch.Tensor) else video
        inputs.mm_processor_kwargs["do_resize"] = False
        return ["<|vision_start|><|video_pad|><|vision_end|>"]

    def _encode(self, inputs: StdTemplateInputs):
        encoded = super()._encode(inputs)
        ids, labels, scales = encoded["input_ids"], encoded["labels"], encoded.get("loss_scale")
        if inputs.audios:
            model = self._get_model()
            audio_tokenizer = getattr(model, "audio_tokenizer", None)
            if audio_tokenizer is None or any(parameter.is_meta for parameter in audio_tokenizer.parameters()):
                raise ValueError("MiMo audio encoding requires a loaded audio tokenizer; bind the real model.")
            config = getattr(self.config, "audio_config", None) or {}
            group_size, channels = config.get("group_size", 4), config.get("audio_channels", 20)
            mels = [self._wav_to_log_mel(
                torch.as_tensor(load_audio(audio, self.audio_sampling_rate), dtype=torch.float32),
                audio_tokenizer.config,
            ) for audio in inputs.audios]
            tokenize = sys.modules[type(model).__module__].tokenize_audio_batch
            with torch.no_grad():
                codes = tokenize(mels, audio_tokenizer.encoder, segment_size=config.get("audio_segment_size", 6000),
                                 device=audio_tokenizer.device)
            codes = [self._pad_codes_to_group(value, group_size, channels) for value in codes]
            indices = findall(ids, self.audio_token_id)
            if len(indices) != len(codes):
                raise ValueError("MiMo audio placeholders must match audio code sequences.")
            ids, labels, scales = self._extend_tokens(
                ids, labels, scales, indices, lambda i: [self.audio_token_id] * (codes[i].shape[0] // group_size),
            )
            encoded["audio_codes"] = torch.cat(codes, dim=0)
        for kind in ("images", "videos"):
            media = getattr(inputs, kind)
            if not media:
                continue
            if kind == "images":
                values = self.processor.image_processor(images=media, return_tensors="pt", do_resize=False)
                grid, token = values["image_grid_thw"], self.image_token_id
            else:
                processor = getattr(self.processor, "video_processor", self.processor.image_processor)
                values = processor(videos=media, return_tensors="pt", do_resize=False)
                grid, token = values["video_grid_thw"], self.video_token_id
                if "pixel_values_videos" in values:
                    values["video_pixel_values"] = values.pop("pixel_values_videos")
            indices = findall(ids, token)
            if len(indices) != len(grid):
                raise ValueError(f"MiMo {kind} placeholders must match media grids.")
            merge_length = self.processor.image_processor.merge_size**2
            ids, labels, scales = self._extend_tokens(
                ids, labels, scales, indices, lambda i: [token] * int(grid[i].prod() // merge_length),
            )
            encoded.update(values)
        encoded.update(input_ids=ids, labels=labels, loss_scale=scales)
        return encoded

    def _data_collator_mm_data(self, batch):
        codes = [row.pop("audio_codes") for row in batch if row.get("audio_codes") is not None]
        videos = [row.pop("video_pixel_values") for row in batch if row.get("video_pixel_values") is not None]
        result = super()._data_collator_mm_data(batch)
        if codes:
            result["audio_codes"] = torch.cat(codes, dim=0)
        if videos:
            result["video_pixel_values"] = torch.cat(videos, dim=0)
        return result


register_template(MiMoV2TemplateMeta("mimo_v2", template_cls=MiMoV2Template))
