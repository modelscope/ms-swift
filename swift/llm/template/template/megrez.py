# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall


@dataclass
class MegrezTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<|role_start|>system<|role_end|>{{SYSTEM}}<|turn_end|>'])
    prompt: Prompt = field(default_factory=lambda:
                           ['<|role_start|>user<|role_end|>{{QUERY}}<|turn_end|><|role_start|>assistant<|role_end|>'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|turn_end|>'])
    suffix: Prompt = field(default_factory=lambda: ['<|turn_end|>'])
    default_system: str = '你是Megrez-3B-Instruct，将针对用户的问题给出详细的、积极的回答。'


register_template(MegrezTemplateMeta(LLMTemplateType.megrez))


class MegrezOmniTemplate(Template):
    skip_prompt = False
    placeholder_tokens = ['<|unk|>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return [[-1], '\n']
        elif media_type == 'audio':
            return [f'Audio {index + 1}: ', [-2], '\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']

        for mm_key in ['images', 'audios']:
            mm_data = getattr(inputs, mm_key)
            if not mm_data:
                continue
            if mm_key == 'images':
                idx_list = findall(input_ids, -1)
                encoding = self.processor.process_image(
                    mm_data,
                    return_tensors='pt',
                )
                text = self.processor.insert_image_feature_placeholders(
                    '<s>'.join(['(<image>./</image>)'] * len(mm_data)), encoding)
                encoded['image_encoding'] = encoding
            else:
                idx_list = findall(input_ids, -2)
                encoding = self.processor.process_audio(
                    mm_data,
                    return_tensors='pt',
                )
                text = self.processor.insert_audio_feature_placeholders(
                    '<s>'.join(['(<audio>./</audio>)'] * len(mm_data)), encoding)
                encoded['audio_encoding'] = encoding

            padding = text.split('<s>')

            def _get_new_tokens(i):
                return self._tokenize(padding[i])

            input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, _get_new_tokens)
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        _, inputs_embeds, _ = model.compose_embeddings(inputs)
        inputs.pop('position_ids', None)
        return {'inputs_embeds': inputs_embeds}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        new_batch = []
        for b in batch:
            text_encodings = {'input_ids': torch.tensor(b['input_ids'])}
            multimodal_inputs = {'image_encoding': b.get('image_encoding'), 'audio_encoding': b.get('audio_encoding')}
            new_batch.append(self.processor.merge_encodings(text_encodings, multimodal_inputs))
        res.update(self.processor.data_collator(new_batch))
        return res


register_template(MegrezTemplateMeta(MLLMTemplateType.megrez_omni, template_cls=MegrezOmniTemplate))
