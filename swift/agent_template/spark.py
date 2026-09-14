# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.template import Prompt
from .base import BaseAgentTemplate

BOS = '<｜start▁of▁sentence｜>'
EOS = '<｜end▁of▁sentence｜>'


class Spark2_5AgentTemplate(BaseAgentTemplate):
    """ref: https://modelscope.cn/models/XHToken/Spark-X2.5-4B (chat_template.jinja)

    Tools are listed in the system block, and every call is an XML-ish block:
        <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value></tool_call>
    Observations live in their own `<|Tool|>` turn.
    """

    @staticmethod
    def _find_function_call(single_content: str) -> Optional[Function]:
        single_content = single_content.strip()
        func_name_match = re.match(r'^([^<]+)', single_content)
        if not func_name_match:
            return None
        func_name = func_name_match.group(1).strip()
        keys = re.findall(r'<arg_key>(.*?)</arg_key>', single_content, re.DOTALL)
        values = re.findall(r'<arg_value>(.*?)</arg_value>', single_content, re.DOTALL)
        if len(keys) != len(values):
            return None
        args = {k.strip(): v.strip() for k, v in zip(keys, values)}
        return Function(name=func_name, arguments=json.dumps(args, ensure_ascii=False))

    def get_toolcall(self, response: str) -> List[Function]:
        toolcall_list = re.findall(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
        functions = []
        for toolcall in toolcall_list:
            function = self._find_function_call(toolcall)
            if function:
                functions.append(function)
        if len(functions) == 0:
            # compat react_en
            return super().get_toolcall(response)
        return functions

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_descs = ['## Tools\nYou have access to the following functions:\n<tools>']
        for tool in tools:
            tool_descs.append(json.dumps(self.unwrap_tool(tool), ensure_ascii=False))
        tool_descs.append('</tools>')
        res = '\n'.join(tool_descs)
        if system:
            res += f'\n\n{system}'
        return res

    def _format_tool_calls(self, tool_call_messages) -> str:
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            tool_calls.append(f'<tool_call>{tool_call["name"]}')
            for arg_key, arg_value in tool_call['arguments'].items():
                if not isinstance(arg_value, str):
                    # `{{ v if v is string else v | tojson }}`
                    arg_value = json.dumps(arg_value, ensure_ascii=False)
                tool_calls.append(f'<arg_key>{arg_key}</arg_key>')
                tool_calls.append(f'<arg_value>{arg_value}</arg_value>')
            tool_calls.append('</tool_call>')
        return ''.join(tool_calls)

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        # The assistant turn is not followed by `chat_sep`, so its EOS is emitted here.
        res = [f'{EOS}{BOS}<|Tool|>']
        for tool_message in tool_messages:
            res.append(f'<tool_response>{tool_message["content"]}</tool_response>')
        res.append(f'{EOS}{BOS}<|Bot|>')
        return assistant_content, res

    def _format_standalone_tool_responses(self, tool_messages) -> 'Prompt':
        # Appended to the user query, i.e. inserted before the EOS that closes the user turn.
        res = [f'{EOS}{BOS}<|Tool|>']
        for tool_message in tool_messages:
            res.append(f'<tool_response>{tool_message["content"]}</tool_response>')
        return res
