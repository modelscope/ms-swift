# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import Optional

from ..constant import LLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Prompt

DEFAULT_SYSTEM = 'You are a helpful assistant.'


class NemotronThinkingMixin:
    """Match Nemotron-3.5's thinking layout: think by default and keep the newline after a
    historical ``</think>``."""

    def __init__(self, *args, enable_thinking=True, **kwargs):
        super().__init__(*args, enable_thinking=True if enable_thinking is None else enable_thinking, **kwargs)

    def _remove_thinking_content(self, content: str, thinking_suffix: str = '</think>') -> str:
        if thinking_suffix in content:
            return self.template_meta.history_thinking_prefix + content.split(thinking_suffix)[-1]
        return super()._remove_thinking_content(content, thinking_suffix)


@dataclass
class ChatmlTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>\n'])
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
