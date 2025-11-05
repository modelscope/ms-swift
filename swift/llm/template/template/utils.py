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
    with_answer = False
    no_think_prefix = ''  # for hybrid thinking model
    history_think_prefix = ''
    add_no_think_prefix_after_tool = True

    def _swift_prepare_inputs(self, inputs):
        super()._swift_prepare_inputs(inputs)
        messages = inputs.messages

        if self.no_think_prefix and self.use_chat_template:
            pre_role = ''
            for message in messages:
                if message['role'] == 'assistant' and isinstance(message['content'], str):
                    if pre_role == 'tool' and not self.add_no_think_prefix_after_tool:
                        pass
                    elif not message['content'].startswith(('<think>', self.no_think_prefix)):
                        # During multi-turn SFT training/validation:
                        # If the message has no <think> block and does not start with the no_think_prefix,
                        # prepend the no_think_prefix to the content.
                        message['content'] = self.no_think_prefix + message['content']
                pre_role = message['role']

        # Only during inference or training, and only if the loss_scale is set to 'last_round',
        # will the previous 'think' entries be deleted.
        if not self.is_training or self.loss_scale.name in {'last_round', 'last_round_with_ignore_empty_think'}:
            for i, message in enumerate(messages):
                # Delete the content before '</think>' in all assistant turns except the last round.
                if message['role'] == 'assistant' and isinstance(message['content'], str) and i != len(messages) - 1:
                    if self.with_answer:
                        message['content'] = message['content'].split('<answer>')[-1].rstrip()
                        if message['content'].endswith('</answer>'):
                            message['content'] = message['content'][:-len('</answer>')]
                        message['content'] = message['content'].strip()
                    else:
                        message['content'] = self.history_think_prefix + message['content'].split(
                            '</think>')[-1].strip()


class ThinkingWithAnswerTemplate(ThinkingTemplate):
    with_answer = True
