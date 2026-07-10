# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.template import Prompt
from .base import BaseAgentTemplate


class HyV3PreviewAgentTemplate(BaseAgentTemplate):
    HYTK = ''

    def get_toolcall(self, response: str) -> List[Function]:
        # Parse tool calls from <tool_calls>...<tool_call>name<tool_sep>...<arg_key>...<arg_value>...</tool_call>...
        tool_call_blocks = re.findall(rf'<tool_call{self.HYTK}>(.*?)</tool_call{self.HYTK}>', response, re.DOTALL)
        functions = []
        for block in tool_call_blocks:
            # Extract function name: text before <tool_sep>
            name_match = re.match(rf'(.*?)<tool_sep{self.HYTK}>', block, re.DOTALL)
            if not name_match:
                continue
            name = name_match.group(1).strip()
            # Extract arg_key/arg_value pairs together to avoid misalignment
            pairs = re.findall(
                rf'<arg_key{self.HYTK}>(.*?)</arg_key{self.HYTK}>\s*<arg_value{self.HYTK}>(.*?)</arg_value{self.HYTK}>',
                block, re.DOTALL)
            arguments = {}
            for k, v in pairs:
                k = k.strip()
                v = v.strip()
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
            res_tool.append(f'<tool_response{self.HYTK}>\n{tool_content}\n</tool_response{self.HYTK}>')
        tool_responses = '\n'.join(res_tool)
        return f'<tool_responses{self.HYTK}>\n{tool_responses}\n</tool_responses{self.HYTK}>'

    def _get_tool_calls(self, tool_calls: List[str]):
        tool_calls_str = '\n'.join(tool_calls)
        return f'<tool_calls{self.HYTK}>\n{tool_calls_str}\n</tool_calls{self.HYTK}>'

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        res = [f'<｜hy_eos{self.HYTK}｜>', self._get_tool_responses(tool_messages), f'<｜hy_Assistant{self.HYTK}｜>']
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
""" + '\n'.join(tool_descs) + f"""
</tools>

For function call returns, you should first print <tool_calls{self.HYTK}>
For each function call, you should return object like:
<tool_call{self.HYTK}>{{function-name}}<tool_sep{self.HYTK}>
<arg_key{self.HYTK}>{{arg-key-1}}</arg_key{self.HYTK}>
<arg_value{self.HYTK}>{{arg-value-1}}</arg_value{self.HYTK}>
<arg_key{self.HYTK}>{{arg-key-2}}</arg_key{self.HYTK}>
<arg_value{self.HYTK}>{{arg-value-2}}</arg_value{self.HYTK}>
...
</tool_call{self.HYTK}>
At the end of function call returns, you should print </tool_calls{self.HYTK}>"""

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
                    arg_lines.append(f'<arg_key{self.HYTK}>{k}</arg_key{self.HYTK}>\n'
                                     f'<arg_value{self.HYTK}>{v}</arg_value{self.HYTK}>')
            arg_str = '\n'.join(arg_lines)
            tool_calls.append(f'<tool_call{self.HYTK}>{name}<tool_sep{self.HYTK}>\n{arg_str}\n</tool_call{self.HYTK}>')
        return self._get_tool_calls(tool_calls)


class HyV3AgentTemplate(HyV3PreviewAgentTemplate):
    HYTK = ':opensource'
