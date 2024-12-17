# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional, Literal

from ..base import Template
from ..constant import MLLMTemplateType, LLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from dataclasses import field, dataclass
from ..utils import Prompt, Context

@dataclass
class MegrezTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<|role_start|>system<|role_end|>{{SYSTEM}}<|turn_end|>'])
    prompt: Prompt = field(default_factory=lambda: ['<|role_start|>user<|role_end|>{{QUERY}}<|turn_end|><|role_start|>assistant<|role_end|>'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|turn_end|>'])
    suffix: Prompt = field(default_factory=lambda: ['<|turn_end|>'])
    default_system: str = '你是Megrez-3B-Instruct，将针对用户的问题给出详细的、积极的回答。'


register_template(MegrezTemplateMeta(LLMTemplateType.megrez))

class MegrezOmniTemplate(Template):
    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        print()

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        return encoded
    
    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        return res

register_template(
    MegrezTemplateMeta(
        MLLMTemplateType.megrez_omni,
        placeholder_tokens=['<|unk|>'],
        template_cls=MegrezOmniTemplate
    )
)