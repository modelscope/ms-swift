# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.template import Prompt
from .base import BaseAgentTemplate


class KimiK25AgentTemplate(BaseAgentTemplate):
    """Agent template for Kimi K2.5/K2.6 models.

    Tool calling format:
    - Tool calls:
      <|tool_calls_section_begin|>
      <|tool_call_begin|>{function_name}<|tool_call_argument_begin|>{args_json}<|tool_call_end|>
      <|tool_calls_section_end|>
    - Tool response:
      ## Return of {tool_call_id}
      {content}
    """

    def get_toolcall(self, response: str) -> List[Function]:
        pattern = r'<\|tool_call_begin\|>(.*?)<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>'
        res_list = re.findall(pattern, response, re.DOTALL)
        functions = []
        for name, arguments in res_list:
            name = name.strip()
            arguments = arguments.strip()
            parsed_args = self._parse_json(arguments)
            if parsed_args is not None:
                functions.append(Function(name=name, arguments=parsed_args))
            else:
                functions.append(Function(name=name, arguments=arguments))
        if len(functions) == 0:
            # compat react_en
            return super().get_toolcall(response)
        return functions

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tools_json = json.dumps(tools, separators=(',', ':'), ensure_ascii=False)
        system = system or ''
        if system:
            system = f'{system}\n\n'
        return f'{system}{tools_json}'

    def _format_tool_calls(self, tool_call_messages) -> str:
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            name = tool_call['name']
            arguments = json.dumps(tool_call['arguments'], ensure_ascii=False)
            tool_calls.append(f'<|tool_call_begin|>{name}<|tool_call_argument_begin|>{arguments}<|tool_call_end|>')
        return f'<|tool_calls_section_begin|>{"".join(tool_calls)}<|tool_calls_section_end|>'

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        res = ['<|im_end|>']
        for tool_message in tool_messages:
            tool_call_id = tool_message.get('tool_call_id', '')
            tool_content = tool_message['content']
            res.append(f'<|im_system|>tool<|im_middle|>## Return of {tool_call_id}\n{tool_content}<|im_end|>')
        res.append('<|im_assistant|>assistant<|im_middle|>')
        return assistant_content, res
