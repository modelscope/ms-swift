# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Optional

from ..base import Template
from ..constant import LLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Prompt

DEFAULT_SYSTEM = 'You are a helpful assistant.'


@dataclass
class ChatmlTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'])
    auto_add_bos: bool = True


@dataclass
class EmptyTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ['{{QUERY}}'])
    chat_sep: Optional[Prompt] = None
    auto_add_bos: bool = True


register_template(ChatmlTemplateMeta(LLMTemplateType.chatml))
register_template(EmptyTemplateMeta(LLMTemplateType.dummy))


class ThinkingTemplate(Template):

    def _swift_prepare_messages(self, messages):
        super()._swift_prepare_messages(messages)
        # Only during inference or training, and only if the loss_scale is set to 'last_round',
        # will the previous 'think' entries be deleted.
        if not self.is_training or self.loss_scale.name == 'last_round':
            for i, message in enumerate(messages):
                # Delete the content before '</think>' in all assistant turns except the last round.
                if message['role'] == 'assistant' and isinstance(message['content'], str) and i != len(messages) - 1:
                    message['content'] = message['content'].split('</think>')[-1].strip()
