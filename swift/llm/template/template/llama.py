# Copyright (c) Alibaba, Inc. and its affiliates.

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn

from swift.utils import get_env_args
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt
from ..vision_utils import load_batch

# ref: https://github.com/facebookresearch/llama/blob/main/llama/generation.py
LLAMA_DEFAULT_SYSTEM = (
    'You are a helpful, respectful and honest assistant. '
    'Always answer as helpfully as possible, while being safe. '
    'Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n\n'
    'If a question does not make any sense, or is not factually coherent, '
    'explain why instead of answering something not correct. '
    "If you don't know the answer to a question, please don't share false information.")

register_template(
    TemplateMeta(
        LLMTemplateType.llama, ['<s>[INST] '], ['{{QUERY}} [/INST]'], ['</s><s>[INST] '], ['</s>'],
        default_system=LLAMA_DEFAULT_SYSTEM,
        system_prefix=['<s>[INST] <<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n']))


@dataclass
class Llama3TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<|begin_of_text|>'])
    prompt: Prompt = field(default_factory=lambda: [
        '<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    ])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|eot_id|>'])
    suffix: Prompt = field(default_factory=lambda: ['<|eot_id|>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{SYSTEM}}<|eot_id|>'])
    tool_prompt: Optional[Prompt] = field(default_factory=lambda: [
        '<|start_header_id|>tool<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    ])
    default_tools_prompt: str = 'toolbench'


register_template(Llama3TemplateMeta(LLMTemplateType.llama3))


def _get_llama3_2_prefix() -> Prompt:
    now = dt.datetime.now()
    date_string = now.strftime('%d %b %Y')
    date_prompt = f'Cutting Knowledge Date: December 2023\nToday Date: {date_string}'
    return [f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{date_prompt}\n\n' '{{SYSTEM}}<|eot_id|>']


@dataclass
class Llama3_2TemplateMeta(Llama3TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: _get_llama3_2_prefix())
    system_prefix: Optional[Prompt] = None


register_template(Llama3_2TemplateMeta(LLMTemplateType.llama3_2))


class Llama3_2VisionTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return ['<|image|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from transformers.models.mllama.processing_mllama import (get_cross_attention_token_mask,
                                                                  convert_sparse_cross_attention_mask_to_dense)
        encoded = super()._encode(inputs)
        images = inputs.images
        if images:
            input_ids = encoded['input_ids']
            processor = self.processor
            image_features = processor.image_processor(images, return_tensors='pt')
            num_tiles = image_features.pop('num_tiles')
            encoded.update(image_features)

            cross_attention_token_mask = [get_cross_attention_token_mask(input_ids, processor.image_token_id)]
            cross_attention_mask = convert_sparse_cross_attention_mask_to_dense(
                cross_attention_token_mask,
                num_tiles=num_tiles,
                max_num_tiles=processor.image_processor.max_image_tiles,
                length=len(input_ids),
            )
            encoded['cross_attention_mask'] = torch.tensor(cross_attention_mask)

        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        for key in ['aspect_ratio_ids', 'aspect_ratio_mask']:
            value = [b[key] for b in batch if b.get(key) is not None]
            if value:
                res[key] = torch.concat(value)

        cross_attention_mask = [
            b['cross_attention_mask'][0] for b in batch if b.get('cross_attention_mask') is not None
        ]
        if cross_attention_mask:
            res['cross_attention_mask'] = self._pad_sequence(cross_attention_mask, 0)
        return res


register_template(Llama3_2TemplateMeta(MLLMTemplateType.llama3_2_vision, template_cls=Llama3_2VisionTemplate))

register_template(
    Llama3TemplateMeta(
        LLMTemplateType.reflection,
        default_system=('You are a world-class AI system, capable of complex reasoning and reflection. '
                        'Reason through the query inside <thinking> tags, and then provide your final '
                        'response inside <output> tags. If you detect that you made a mistake in your reasoning '
                        'at any point, correct yourself inside <reflection> tags.')))


class Llama3_1OmniTemplate(Template):
    skip_prompt = False
    audio_placeholder = [[-200]]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        import whisper
        encoded = super()._encode(inputs)
        audios = inputs.audios
        if audios:
            audios = load_batch(audios, whisper.load_audio)
            n_mels = get_env_args('n_mels', int, 128)
            for i, audio in enumerate(audios):
                audio = whisper.pad_or_trim(audio)
                audios[i] = whisper.log_mel_spectrogram(audio, n_mels=n_mels).permute(1, 0)
            audios = torch.stack(audios)
            encoded.update({'speech': audios, 'speech_lengths': torch.tensor([[audios.shape[1]]])})

        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        speech = inputs.get('speech')
        input_ids = inputs['input_ids']
        labels = inputs.get('labels')
        if speech is not None:
            speech_lengths = inputs['speech_lengths']
            speech = speech.to(model.dtype)
            inputs_embeds, labels = model.prepare_inputs_labels_for_speech_and_text(input_ids, None, None, None, labels,
                                                                                    speech, speech_lengths)[4:]
        else:
            inputs_embeds = model.get_model().embed_tokens(input_ids)
        res = {'inputs_embeds': inputs_embeds}
        if labels is not None:
            res['labels'] = labels[0]
        return res


register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.llama3_1_omni,
        default_system=('You are a helpful language and speech assistant. '
                        'You are able to understand the speech content that the user provides, '
                        'and assist the user with a variety of tasks using natural language.'),
        template_cls=Llama3_1OmniTemplate,
    ))
