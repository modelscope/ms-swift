# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Optional

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
