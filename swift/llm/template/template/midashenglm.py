# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import torch

from swift.utils import get_env_args, is_deepspeed_enabled
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Word, findall
from ..vision_utils import load_batch
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta


@dataclass
class QwenTemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])


class MiDashengLMTemplate(Template):
    placeholder_tokens = ['<|AUDIO|>']
    skip_prompt = False

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        return ['<|AUDIO|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from transformers.audio_utils import load_audio
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        sampling_rate = get_env_args('sampling_rate', int, 16000)
        inputs.audios = load_batch(inputs.audios, partial(load_audio, sampling_rate=sampling_rate))
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
            res['input_values'] = torch.concat(input_values)
            if audio_lengths:
                res['audio_length'] = torch.concat(audio_lengths)

        return res


register_template(QwenTemplateMeta(MLLMTemplateType.midashenglm, template_cls=MiDashengLMTemplate))
