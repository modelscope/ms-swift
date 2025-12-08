# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import json

from .base import BaseAgentTemplate

if TYPE_CHECKING:
    from swift.llm.infer import Function
    from swift.llm.template import Prompt


class DeepSeekV31AgentTemplate(BaseAgentTemplate):

    def get_toolcall(self, response: str) -> List['Function']:
        # Parse tool calls using the DSV3.1 format:
        # <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>name<｜tool▁sep｜>args<｜tool▁call▁end｜>
        pattern = r'<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>'
        res_list = re.findall(pattern, response, re.DOTALL)
        functions = []
        for name, arguments in res_list:
            name = name.strip()
            arguments = self._parse_json(arguments.strip())
            if arguments is not None:
                functions.append(Function(name=name, arguments=arguments))

        if len(functions) == 0:
            # compat react_en
            return super().get_toolcall(response)
        return functions

    def _get_tool_responses(self, tool_messages):
        return ''.join(f'<｜tool▁output▁begin｜>{tool_message["content"]}<｜tool▁output▁end｜>'
                       for tool_message in tool_messages)

    def _get_tool_calls(self, tool_calls: List[str]):
        return f'<｜tool▁calls▁begin｜>{"".join(tool_calls)}<｜tool▁calls▁end｜>'

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        res = ['<｜end▁of▁sentence｜>', self._get_tool_responses(tool_messages)]
        return assistant_content, res

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_descs = []
        system = system or ''
        for tool in tools:
            tool = self.unwrap_tool(tool)
            tool_name = self._get_tool_name(tool)
            description = tool.get('description', '')
            parameters = tool.get('parameters', {})

            tool_desc = f"""### {tool_name}
Description: {description}

Parameters: {json.dumps(parameters, ensure_ascii=False)}"""
            tool_descs.append(tool_desc)

        tools_section = '\n\n'.join(tool_descs)

        return f"""{system}

## Tools
You have access to the following tools:

{tools_section}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>

Where:
- `tool_call_name` must be an exact match to one of the available tools
- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces"""

    def _format_tool_calls(self, tool_call_messages):
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            name = tool_call['name']
            arguments = json.dumps(tool_call['arguments'], ensure_ascii=False)
            tool_calls.append(f'<｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{arguments}<｜tool▁call▁end｜>')
        return self._get_tool_calls(tool_calls)
