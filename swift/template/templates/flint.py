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

from ..base import Template
from ..constant import LLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Prompt, Word


class FlintTemplate(Template):
    # The checkpoint's four `eos_token_id` entries: the end of turn, plus the three role markers that
    # also end a turn. All four must be dropped from the answer, but `Template.skip_stop_tokens`
    # only knows `tokenizer.eos_token_id` and the template suffix, i.e. `<|end_of_text|>`.
    stop_tokens = ['<|end_of_text|>', '<|user|>', '<|observation|>', '<|developer|>']

    def decode_generate_ids(self, generate_ids, **kwargs):
        stop_token_ids = {self._tokenize(token)[0] for token in self.stop_tokens}
        while generate_ids and generate_ids[-1] in stop_token_ids:
            generate_ids = generate_ids[:-1]
        return super().decode_generate_ids(generate_ids, **kwargs)


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
    # Generation already stops on `<|user|>` / `<|observation|>` / `<|developer|>` through the
    # checkpoint's `eos_token_id`, and `FlintTemplate` drops them from the response; nothing to add.
    stop_words: List[Word] = field(default_factory=list)


register_template(FlintTemplateMeta(LLMTemplateType.flint, template_cls=FlintTemplate))
