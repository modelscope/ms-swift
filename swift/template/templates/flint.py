# Copyright (c) ModelScope Contributors. All rights reserved.
"""Flint chat template.

Flint's `chat_template.jinja` writes role markers without any separator (`<|system|>`, `<|user|>`,
`<|assistant|>`, `<|observation|>`, `<|developer|>`), keeps optional reasoning inline as
`<think>...</think>` at the start of the assistant turn, and ends a turn with `<|end_of_text|>`.
There is no thinking switch and no default system prompt: the model decides per turn whether to
reason. Tool calls use the `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
form and tool results come back as `<|observation|><tool_response>...</tool_response>`.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ..constant import LLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Prompt, Word


@dataclass
class FlintTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|system|>{{SYSTEM}}'])
    prompt: Prompt = field(default_factory=lambda: ['<|user|>{{QUERY}}<|assistant|>'])
    chat_sep: Optional[Prompt] = field(default_factory=list)
    suffix: Prompt = field(default_factory=lambda: ['<|end_of_text|>'])
    # No BOS for chat, matching the reference tokenizer (`add_bos_token: false`).
    auto_add_bos: bool = False
    agent_template: str = 'flint'
    # `<|user|>` / `<|observation|>` / `<|developer|>` are in `eos_token_id`, so generation already
    # stops on them; they stay out of the response, hence no extra stop words.
    stop_words: List[Word] = field(default_factory=list)


register_template(FlintTemplateMeta(LLMTemplateType.flint))
