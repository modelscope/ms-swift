# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn.functional as F
from functools import partial
from typing import Any, Dict, List, Literal

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, findall
from ..vision_utils import load_audio, load_batch


class SophonRealtimeTemplate(Template):
    """ChatML as rendered by the released tokenizer (no think block, no default system prompt).

    Each `<audio>` becomes one `<|audio|>` token per 16 mel frames (6.25 tokens per second); the model replaces
    them with the projected audio embeddings. Audio may also appear in assistant turns (as context).
    """
    audio_token_id = 151669
    sampling_rate = 16000
    support_padding_free = False

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        return [[self.audio_token_id]]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        idx_list = findall(input_ids, self.audio_token_id)
        if idx_list:
            audios = load_batch(inputs.audios, partial(load_audio, sampling_rate=self.sampling_rate))
            feats = self.processor.audio_features(audios)
            n_tokens = feats.pop('tokens_per_clip')
            encoded['input_ids'], encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
                input_ids, encoded['labels'], encoded['loss_scale'], idx_list,
                lambda i: [self.audio_token_id] * n_tokens[i])
            encoded.update(feats)
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        audio_values = [b['audio_values'] for b in batch if b.get('audio_values') is not None]
        if audio_values:
            max_len = max(v.shape[-1] for v in audio_values)
            res['audio_values'] = torch.concat([F.pad(v, (0, max_len - v.shape[-1])) for v in audio_values])
            for key in ['audio_lens', 'audio_token_len']:
                res[key] = torch.concat([b[key] for b in batch if b.get(key) is not None])
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.sophon_realtime,
        prefix=[],
        prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        chat_sep=['<|im_end|>\n'],
        suffix=['<|im_end|>'],
        system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
        stop_words=['<|endoftext|>'],
        agent_template='hermes',
        template_cls=SophonRealtimeTemplate,
    ))
