# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, align_image_inputs, findall, split_tokens
import torch.nn.functional as F

@dataclass
class ERNIETemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<|begin_of_sentence|>'])
    prompt: Prompt = field(default_factory=lambda: ['User: {{QUERY}}\nAssistant: '])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|end_of_sentence|>'])
    suffix: Prompt = field(default_factory=lambda: ['</s>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|begin_of_sentence|>{{SYSTEM}}\n'])


register_template(ERNIETemplateMeta(LLMTemplateType.ernie))


class ERNIETemplate(Template):
    placeholder_tokens = ['<|IMAGE_PLACEHOLDER|>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [f'Picture {index+1}:<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        if inputs.images:
            text = self.processor.decode(input_ids).replace('<|IMAGE_PLACEHOLDER|>', '<|image@placeholder|>')
            new_inputs = self.processor(
                text=[text],
                images=inputs.images,
                videos=inputs.videos,
                padding=True,
                return_tensors='pt',
            )
            encoded.update(new_inputs)
            new_input_ids = new_inputs['input_ids'][0].tolist()
            encoded['input_ids'], encoded['labels'] = align_image_inputs(input_ids, labels, new_input_ids,
                                                                         self.placeholder_tokens[0])
            encoded['position_ids'] = encoded['position_ids'][0]
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = {}
        for key in ['images', 'grid_thw', 'image_type_ids']:
            res[key] = self.concat_tensor(batch, key, 0)
        res.update(super()._data_collator(batch, padding_to=padding_to))
        return res


register_template(ERNIETemplateMeta(MLLMTemplateType.ernie_vl, template_cls=ERNIETemplate))
