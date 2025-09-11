# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Optional

from ..constant import LLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Prompt


@dataclass
class ERNIETemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<|begin_of_sentence|>'])
    prompt: Prompt = field(default_factory=lambda: ['User: {{QUERY}}\nAssistant: '])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|end_of_sentence|>'])
    suffix: Prompt = field(default_factory=lambda: ['</s>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|begin_of_sentence|>{{SYSTEM}}\n'])


register_template(ERNIETemplateMeta(LLMTemplateType.ernie))


@dataclass
class ERNIEThinkingTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<|im_start|>system\n'])
    prompt: Prompt = field(default_factory=lambda: ['<global_setting>\n'
                                                    'think_mode=True\n'
                                                    '</global_setting><|im_end|>\n\n'
                                                    '<|im_start|>user\n'
                                                    '{{QUERY}}<|im_end|>\n\n'
                                                    '<|im_start|>assistant\n'
                                                    '<think>\n\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>\n\n'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|im_start|>system\n'
                                                                     '<system_setting>\n'
                                                                     '{{SYSTEM}}\n'
                                                                     '</system_setting>\n\n'])


register_template(ERNIEThinkingTemplateMeta(LLMTemplateType.ernie_thinking))
