# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import Optional

from ..constant import LLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Prompt

DEFAULT_SYSTEM = "You are a helpful assistant."


@dataclass
class ChatmlTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ["<|im_start|>user\n{{QUERY}}<|im_end|>\n"])
    response_opener: Optional[Prompt] = field(default_factory=lambda: ["<|im_start|>assistant\n"])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ["<|im_end|>\n"])
    suffix: Prompt = field(default_factory=lambda: ["<|im_end|>\n"])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"])
    # the official chatml jinjas never prepend the tokenizer's own bos (they render with
    # add_special_tokens=False); a bos belongs in prefix only when a jinja spells it out.
    auto_add_bos: bool = False


@dataclass
class EmptyTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ["{{QUERY}}"])
    chat_sep: Optional[Prompt] = None
    auto_add_bos: bool = True


register_template(ChatmlTemplateMeta(LLMTemplateType.chatml))
register_template(EmptyTemplateMeta(LLMTemplateType.dummy))
