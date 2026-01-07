# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import json

from .hermes import HermesAgentTemplate

if TYPE_CHECKING:
    from swift.llm.template import Prompt


class YoutuAgentTemplate(HermesAgentTemplate):
    """Agent template for Youtu-LLM models.

    Tool calling format:
    - Tool call: <tool_call>{"name": "function-name", "arguments": {...}}</tool_call>
    - Tool response: <tool_response>...</tool_response>
    """

    def _get_tool_responses(self, tool_messages):
        res_tool = []
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res_tool.append(f'<tool_response>{tool_content}</tool_response>')
        return '\n'.join(res_tool)

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        # For Youtu-LLM, tool responses are placed in user message
        if hasattr(self, 'template_meta'):
            prompt = self.template_meta.prompt
            chat_sep = self.template_meta.chat_sep
        else:
            prompt = ['<|User|>{{QUERY}}<|Assistant|>']
            chat_sep = ['<|end_of_text|>']
        res = chat_sep.copy()
        total_tool = self._get_tool_responses(tool_messages)
        for context in prompt:
            if isinstance(context, str):
                context = context.replace('{{QUERY}}', total_tool)
            res.append(context)
        return assistant_content, res

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_descs = [json.dumps(self.wrap_tool(tool), ensure_ascii=False) for tool in tools]
        system = system or ''
        if system:
            system = f'{system}\n\n'
        return f"""{system}<|begin_of_tool_description|>Tool calling capabilities.
You may call one or more functions to assist with the user query. You have the following functions available:
""" + '\n'.join([f'```json\n{desc}\n```' for desc in tool_descs]) + """
For tool call returns, you MUST use the following format:
<tool_call>{"name": "function-name", "arguments": {"param1": "value1", "param2": "value2"}}</tool_call>
<|end_of_tool_description|>"""

    def _format_tool_calls(self, tool_call_messages):
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            tool_calls.append(f'<tool_call>{json.dumps(tool_call, ensure_ascii=False)}</tool_call>')
        return ''.join(tool_calls)
