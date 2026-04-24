# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.template import Prompt
from .base import BaseAgentTemplate


class Hy3AgentTemplate(BaseAgentTemplate):
    """Agent template for Tencent Hunyuan Hy3 models.

    Hy3 uses a unique tool calling format with structured arg_key/arg_value pairs:
        <tool_calls>
        <tool_call>func_name<tool_sep>
        <arg_key>key</arg_key>
        <arg_value>value</arg_value>
        </tool_call>
        </tool_calls>
    """

    def get_toolcall(self, response: str) -> List[Function]:
        # Parse tool calls from <tool_calls>...<tool_call>name<tool_sep>...<arg_key>...<arg_value>...</tool_call>...
        tool_call_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
        functions = []
        for block in tool_call_blocks:
            # Extract function name: text before <tool_sep>
            name_match = re.match(r'(.*?)<tool_sep>', block, re.DOTALL)
            if not name_match:
                continue
            name = name_match.group(1).strip()
            # Extract arg_key/arg_value pairs
            keys = re.findall(r'<arg_key>(.*?)</arg_key>', block, re.DOTALL)
            values = re.findall(r'<arg_value>(.*?)</arg_value>', block, re.DOTALL)
            arguments = {}
            for k, v in zip(keys, values):
                k = k.strip()
                v = v.strip()
                # Try to parse value as JSON for non-string types
                parsed = self._parse_json(v)
                arguments[k] = parsed if parsed is not None else v
            functions.append(Function(name=name, arguments=arguments))
        if len(functions) == 0:
            # compat react_en
            return super().get_toolcall(response)
        return functions

    def _get_tool_responses(self, tool_messages):
        res_tool = []
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res_tool.append(f'<tool_response>\n{tool_content}\n</tool_response>')
        tool_responses = '\n'.join(res_tool)
        return f'<tool_responses>\n{tool_responses}\n</tool_responses>'

    def _get_tool_calls(self, tool_calls: List[str]):
        tool_calls_str = '\n'.join(tool_calls)
        return f'<tool_calls>\n{tool_calls_str}\n</tool_calls>'

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        res = ['<｜hy_eos｜>', self._get_tool_responses(tool_messages), '<｜hy_Assistant｜>']
        return assistant_content, res

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_descs = [json.dumps(self.wrap_tool(tool), ensure_ascii=False) for tool in tools]
        system = system or ''
        if system:
            system = f'{system}\n\n'
        return f"""{system}# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + '\n'.join(tool_descs) + """
</tools>

For function call returns, you should first print <tool_calls>
For each function call, you should return object like:
<tool_call>{function-name}<tool_sep>
<arg_key>{arg-key-1}</arg_key>
<arg_value>{arg-value-1}</arg_value>
<arg_key>{arg-key-2}</arg_key>
<arg_value>{arg-value-2}</arg_value>
...
</tool_call>
At the end of function call returns, you should print </tool_calls>"""

    def _format_tool_calls(self, tool_call_messages):
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            name = tool_call['name']
            arguments = tool_call['arguments']
            arg_lines = []
            if isinstance(arguments, dict):
                for k, v in arguments.items():
                    if not isinstance(v, str):
                        v = json.dumps(v, ensure_ascii=False)
                    arg_lines.append(f'<arg_key>{k}</arg_key>\n<arg_value>{v}</arg_value>')
            arg_str = '\n'.join(arg_lines)
            tool_calls.append(f'<tool_call>{name}<tool_sep>\n{arg_str}\n</tool_call>')
        return self._get_tool_calls(tool_calls)
