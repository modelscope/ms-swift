# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn.functional as F

from swift.utils import get_env_args
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Word, findall
from ..vision_utils import load_batch
from .qwen import QwenTemplateMeta


class MiDashengLMTemplate(Template):
    placeholder_tokens = ['<|AUDIO|>']
    skip_prompt = False

    def init_env_args(self):
        super().init_env_args()
        self.sampling_rate = get_env_args('sampling_rate', int, 16000)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        return ['<|AUDIO|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from transformers.audio_utils import load_audio
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        inputs.audios = load_batch(inputs.audios, partial(load_audio, sampling_rate=self.sampling_rate))
        audio_token = self._tokenize('<|AUDIO|>')[0]
        idx_list = findall(input_ids, audio_token)
        if idx_list:
            split_token = self._tokenize('\n')[0]
            audio_inputs = self.processor(text='\n'.join(['<|AUDIO|>'] * len(inputs.audios)), audio=inputs.audios)
            splited_tokens = self._split_list(audio_inputs['input_ids'][0].tolist(), split_token)

            encoded['input_ids'], encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
                input_ids, encoded['labels'], encoded['loss_scale'], idx_list, lambda i: splited_tokens[i])
            encoded['input_values'] = audio_inputs['input_values']
            encoded['audio_length'] = audio_inputs['audio_length']
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)

        input_values = [b['input_values'] for b in batch if b.get('input_values') is not None]
        audio_lengths = [b['audio_length'] for b in batch if b.get('audio_length') is not None]

        if input_values:
            res['audio_length'] = torch.concat(audio_lengths)
            for i in range(len(input_values)):
                pad_len = (res['audio_length'].max() - input_values[i].shape[1]).item()
                input_values[i] = F.pad(input_values[i], (0, pad_len), 'constant', 0)
            res['input_values'] = torch.concat(input_values)

        return res


register_template(QwenTemplateMeta(MLLMTemplateType.midashenglm, template_cls=MiDashengLMTemplate))
