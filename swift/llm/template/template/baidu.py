# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Optional

from ..base import Template
from ..constant import LLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Prompt
from .utils import ThinkingTemplate


@dataclass
class ERNIETemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<|begin_of_sentence|>'])
    prompt: Prompt = field(default_factory=lambda: ['User: {{QUERY}}\nAssistant: '])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|end_of_sentence|>'])
    suffix: Prompt = field(default_factory=lambda: ['</s>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|begin_of_sentence|>{{SYSTEM}}\n'])


register_template(ERNIETemplateMeta(LLMTemplateType.ernie))


class ErnieThinkingTemplate(ThinkingTemplate):

    def _swift_prepare_inputs(self, inputs) -> None:
        super()._swift_prepare_inputs(inputs)
        for message in inputs.messages:
            if message['role'] == 'assistant':
                if '<response>' not in message['content']:
                    if '</think>' in message['content']:
                        message['content'] = message['content'].replace('</think>', '</think>\n\n<response>\n')
                        message['content'] = message['content'] + '\n</response>'
                        if '<think>\n' not in message['content']:
                            message['content'] = message['content'].replace('<think>', '<think>\n')
                    else:
                        message['content'] = '<response>\n' + message['content'] + '\n</response>\n'


@dataclass
class ERNIEThinkingTemplateMeta(TemplateMeta):
    prefix: Prompt = field(
        default_factory=lambda:
        ['<|im_start|>system\n'
         '<global_setting>\n'
         'think_mode=True\n'
         '</global_setting><|im_end|>\n\n'])
    prompt: Prompt = field(
        default_factory=lambda: ['<|im_start|>user\n'
                                 '{{QUERY}}<|im_end|>\n\n'
                                 '<|im_start|>assistant\n'])
    response_prefix: Optional[str] = '<think>\n'
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>\n\n'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: [
        '<|im_start|>system\n'
        '<system_setting>\n'
        '{{SYSTEM}}\n'
        '</system_setting>\n\n'
        '<global_setting>\n'
        'think_mode=True\n'
        '</global_setting><|im_end|>\n\n'
    ])


register_template(ERNIEThinkingTemplateMeta(LLMTemplateType.ernie_thinking, template_cls=ErnieThinkingTemplate))
