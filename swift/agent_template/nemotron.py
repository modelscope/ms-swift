# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List, Optional, Union

from .qwen3_coder import Qwen3CoderAgentTemplate


class Nemotron3_5AgentTemplate(Qwen3CoderAgentTemplate):
    """ref: https://modelscope.cn/models/nv-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (chat_template.jinja)

    NVIDIA-Nemotron-3.5 uses the Qwen3-Coder XML tool format with a few layout differences:
    no default system prompt, every call ends with '</tool_call>\\n', and a single '\\n'
    separates the (trimmed) assistant content from the first call.
    """

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tools_prompt = super()._format_tools(tools, '', user_message).lstrip('\n')
        if system:
            tools_prompt = f'{system}\n\n{tools_prompt}'
        return tools_prompt

    def _format_tool_calls(self, tool_call_messages):
        return super()._format_tool_calls(tool_call_messages) + '\n'

    def _add_tool_call_prefix(self, tool_content: str, pre_message=None) -> str:
        if pre_message and pre_message.get('role') == 'assistant':
            content = pre_message.get('content', '')
            if not isinstance(content, str):
                return tool_content
            # jinja: (content | trim) ~ '\n'
            pre_message['content'] = content.strip()
            if pre_message['content']:
                return '\n' + tool_content
        # Without assistant text, the jinja still separates its '<think></think>' from the first call.
        template_meta = getattr(self, 'template_meta', None)
        if template_meta is not None and template_meta.non_thinking_prefix:
            return '\n' + tool_content
        return tool_content
