# Copyright (c) Alibaba, Inc. and its affiliates.
import itertools
from functools import partial
from typing import Any, Dict, List, Literal, Optional

from swift.utils import get_env_args
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, findall
from ..vision_utils import load_batch, load_file
from .qwen import QwenTemplateMeta


class GOTImageEvalProcessor:

    def __init__(self, image_size=384, mean=None, std=None):
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            self.normalize,
        ])

    def __call__(self, item):
        return self.transform(item)


class GOT_OCR2Template(Template):
    placeholder_tokens = ['<imgpad>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        # 'OCR: '
        # 'OCR with format: '
        assert media_type == 'image'
        return ['<img>' + '<imgpad>' * 256 + '</img>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        image_processor_high = GOTImageEvalProcessor(image_size=1024)
        for i, image in enumerate(images):
            images[i] = image_processor_high(image)[None].to(self.model_info.torch_dtype)
        if images:
            encoded['images'] = images
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        images = self.gather_list(batch, 'images')
        if images:
            res['images'] = images
        return res


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.got_ocr2,
        default_system='        You should follow the instructions carefully and explain your answers in detail.',
        template_cls=GOT_OCR2Template,
    ))


class GOT_OCR2HfTemplate(Template):
    placeholder_tokens = ['<imgpad>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        # 'OCR: '
        # 'OCR with format: '
        assert media_type == 'image'
        return ['<img>' + '<imgpad>' * 256 + '</img>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:  # 暂时照抄上面
        encoded = super()._encode(inputs)
        images = inputs.images
        if images:
            encoded['images'] = images
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        images = self.gather_list(batch, 'images')
        _inputs = self.processor(images, return_tensors='pt')
        _inputs.pop('input_ids')  # this does not contain the response, so cannot be used when training
        _inputs.pop('attention_mask')  # this does not contain the response, so cannot be used when training

        res.update(_inputs.data)
        return res


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.got_ocr2_hf,
        default_system='        You should follow the instructions carefully and explain your answers in detail.',
        template_cls=GOT_OCR2HfTemplate,
    ))


class StepAudioTemplate(Template):
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio', f'media_type: {media_type}'
        from utils import load_audio
        audio_wav, sr = load_audio(load_file(inputs.audios[index]))
        audio_tokens = self.model.encoder(audio_wav, sr)
        return audio_tokens


class StepAudio2MiniTemplate(Template):
    use_model = True

    def load_audio(self, file_path, target_rate=16000, max_length=None):
        '''
        Open an audio file and read as mono waveform, resampling as necessary
        If max_length is provided, truncate the audio to that length
        '''
        import torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != target_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)(waveform)
        audio = waveform[0]  # get the first channel

        # Truncate audio if it exceeds max_length
        if max_length is not None and audio.shape[0] > max_length:
            audio = audio[:max_length]

        return audio

    def _mel_filters(self, n_mels: int) -> 'torch.Tensor':
        '''Load the mel filterbank matrix for projecting STFT into a Mel spectrogram.'''
        import librosa
        import torch
        assert n_mels in {80, 128}, f'Unsupported n_mels: {n_mels}'
        if n_mels == 128:
            return torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=128))
        else:
            return torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=80))

    def log_mel_spectrogram(self, audio, n_mels=128, padding=479):
        '''
        Compute the log-Mel spectrogram with specific padding for StepAudio
        '''
        import torch
        import torch.nn.functional as F
        if isinstance(audio, str):
            audio = self.load_audio(audio)
        elif not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        window = torch.hann_window(400).to(audio.device)
        stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs()**2
        filters = self._mel_filters(n_mels)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def compute_token_num(self, max_feature_len):
        # First, audio goes through encoder:
        # 1. conv1: kernel=3, stride=1, padding=1 -> size unchanged
        # 2. conv2: kernel=3, stride=2, padding=1 -> size/2
        # 3. avg_pooler: kernel=2, stride=2 -> size/2
        max_feature_len = max_feature_len - 2  # remove padding
        encoder_output_dim = (max_feature_len + 1) // 2 // 2  # after conv2 and avg_pooler

        # Then through adaptor (parameters from config file):
        padding = 1
        kernel_size = 3  # from config: audio_encoder_config.kernel_size
        stride = 2  # from config: audio_encoder_config.adapter_stride
        adapter_output_dim = (encoder_output_dim + 2 * padding - kernel_size) // stride + 1
        return adapter_output_dim

    def padding_mels(self, data: List['torch.Tensor']):
        ''' Padding the data into batch data

        Parameters
        ----------
            data: List[Tensor], shape of Tensor (128, T)

        Returns:
        -------
            feats, feats lengths
        '''
        import torch
        from torch.nn.utils.rnn import pad_sequence
        sample = data
        assert isinstance(sample, list)
        feats_lengths = torch.tensor([s.size(1) - 2 for s in sample], dtype=torch.int32)
        feats = [s.t() for s in sample]
        padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)

        return padded_feats.transpose(1, 2), feats_lengths

    def audio_process(self, audio):
        results = []
        mels = []
        for i in range(0, audio.shape[0], 16000 * 25):
            mel = self.log_mel_spectrogram(audio[i:i + 16000 * 25], n_mels=128, padding=479)
            mels.append(mel)
            audio_tokens = '<audio_patch>' * self.compute_token_num(mel.shape[1])
            results.append(f'<audio_start>{audio_tokens}<audio_end>')
        audio_ids = self._tokenize(''.join(results))
        return audio_ids, mels

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        return ['<audio_patch>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        sampling_rate = get_env_args('sampling_rate', int, 16000)
        inputs.audios = load_batch(inputs.audios, partial(self.load_audio, target_rate=sampling_rate))

        audio_token = self._tokenize('<audio_patch>')[0]
        idx_list = findall(input_ids, audio_token)

        if idx_list:
            audio_inputs = []
            mels = []
            for audio in inputs.audios:
                audio_input, mel = self.audio_process(audio)
                audio_inputs.append(audio_input)
                mels.extend(mel)

            def _get_new_audio_tokens(i):
                return audio_inputs[i]

            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                _get_new_audio_tokens)
            encoded['input_ids'] = input_ids  # Add labels to the batch
            encoded['labels'] = labels  # Add labels to the batch
            encoded['loss_scale'] = loss_scale
            encoded['mels'] = mels
            wavs, wav_lens = self.padding_mels(mels)
            # audio_tokens = [151688, 151690, 151689]
            # for audio_token_id in audio_tokens:
            #     labels[labels == audio_token_id] = -100  # Mask image token IDs in labels

        else:
            wavs = None
            wav_lens = None

        encoded['wavs'] = wavs
        encoded['wav_lens'] = wav_lens

        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        batch_wavs, batch_wav_lens = self.padding_mels(list(itertools.chain.from_iterable([e['mels'] for e in batch])))
        res = super()._data_collator(batch, padding_to=padding_to)
        res['wav_lens'] = batch_wav_lens
        res['wavs'] = batch_wavs

        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.step_audio2_mini,
        template_cls=StepAudio2MiniTemplate,
        prefix=[],
        prompt=['<|BOT|>human\n{{QUERY}}<|EOT|><|BOT|>assistant\n'],
        system_prefix=['<|BOT|>system\n{{SYSTEM}}<|EOT|>'],
        chat_sep=['<|EOT|>'],
        suffix=['<|EOT|>'],
    ))

register_template(
    TemplateMeta(
        MLLMTemplateType.step_audio,
        template_cls=StepAudioTemplate,
        prefix=['<s>'],
        prompt=['<|BOT|>human\n{{QUERY}}<|EOT|><|BOT|>assistant\n'],
        system_prefix=['<s><|BOT|>system\n{{SYSTEM}}<|EOT|>'],
        chat_sep=['<|EOT|>'],
        suffix=['<|EOT|>'],
    ))
