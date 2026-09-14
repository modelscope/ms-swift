# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import Optional

from ..base import Template
from ..constant import LLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Prompt

BOS = '<｜start▁of▁sentence｜>'
EOS = '<｜end▁of▁sentence｜>'
# chat_template.jinja hardcodes this system block: it is emitted for every request, and a
# user-provided system message is appended after it instead of replacing it.
spark_builtin_system = 'you are a helpful assistant.'


class Spark2_5Template(Template):

    def _get_system(self, inputs: StdTemplateInputs) -> Optional[str]:
        system = super()._get_system(inputs)
        if system and not inputs.tools:
            # A user system is separated from the builtin one by a blank line. With tools, the
            # separator belongs to the tools block, which is glued to the builtin system.
            system = f'\n\n{system}'
        return system


@dataclass
class Spark2_5TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: [f'{BOS}<|System|>\n{spark_builtin_system}{EOS}'])
    prompt: Prompt = field(default_factory=lambda: [f'{BOS}<|User|>{{{{QUERY}}}}{EOS}{BOS}<|Bot|>'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: [EOS])
    suffix: Prompt = field(default_factory=lambda: [EOS])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: [f'{BOS}<|System|>\n{spark_builtin_system}{{{{SYSTEM}}}}{EOS}'])

    agent_template: str = 'spark2_5'
    # Hybrid thinking: `<think>` opens the reasoning channel, a bare `</think>` closes it
    # immediately, and history turns keep only the bare `</think>`.
    is_thinking: bool = True
    thinking_prefix: str = '<think>'
    non_thinking_prefix: str = '</think>'
    history_thinking_prefix: str = '</think>'


register_template(Spark2_5TemplateMeta(LLMTemplateType.spark2_5, template_cls=Spark2_5Template))
