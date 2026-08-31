# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.template import Prompt
from .base import BaseAgentTemplate

_FORMAT_EXAMPLE = ('<tool_call>{function-name}\n'
                   '<arg_key>{arg-key-1}</arg_key>\n'
                   '<arg_value>{arg-value-1}</arg_value>\n'
                   '<arg_key>{arg-key-2}</arg_key>\n'
                   '<arg_value>{arg-value-2}</arg_value>\n'
                   '...\n'
                   '</tool_call>\n')


class Ling3AgentTemplate(BaseAgentTemplate):
    """Agent template for Ling-3.0 models (Bailing V3 format).

    Tool calling uses tool_call/arg_key/arg_value XML tags,
    matching the model's chat_template.jinja.
    """

    @staticmethod
    def _find_function_call(single_content: str) -> Optional[Function]:
        single_content = single_content.strip()
        func_name_match = re.match(r'^([^\n<]+)', single_content)
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
        tool_descs = [
            '# Tools\n\n'
            'You may call one or more functions to assist with the user query.\n\n'
            'You are provided with function signatures within '
            '<tools></tools> XML tags:\n<tools>'
        ]
        for tool in tools:
            tool = self.wrap_tool(tool)
            tool_descs.append(json.dumps(tool, ensure_ascii=False))
        tool_descs.append('</tools>\n\n'
                          'If none of the functions can be used, point it out. '
                          'If the given question lacks the parameters required by the function, '
                          'also point it out.\n'
                          'If you need to use a function, for each function call, '
                          'output the function name and arguments within the following '
                          'XML format:\n' + _FORMAT_EXAMPLE)
        tool_section = '\n'.join(tool_descs)
        if system is not None and system.strip():
            return system.strip() + '\n' + tool_section
        return tool_section

    def _format_tool_calls(self, tool_call_messages) -> str:
        # Jinja: \n before each tool_call when (loop.first and content) or not loop.first.
        # Since _format_tool_calls returns ONLY tool calls (content is separate),
        # the first call gets no prefix, subsequent calls get \n.
        parts = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            tc = '<tool_call>' + tool_call['name']
            for arg_key, arg_value in tool_call['arguments'].items():
                tc += '<arg_key>' + arg_key + '</arg_key>'
                tc += '\n<arg_value>'
                if isinstance(arg_value, str):
                    tc += arg_value
                else:
                    tc += json.dumps(arg_value, ensure_ascii=False)
                tc += '</arg_value>'
            tc += '\n</tool_call>'
            parts.append(tc)
        return '\n'.join(parts)

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = (self.keyword.action in assistant_content and self.keyword.action_input in assistant_content)
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        # Close assistant turn, open OBSERVATION, then re-open ASSISTANT.
        # chat_sep is NOT added when next query is tool, so we must include <|role_end|>.
        res = ['<|role_end|><role>OBSERVATION</role>']
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res.append('\n<tool_response>\n' + tool_content + '\n</tool_response>')
        res.append('<|role_end|><role>ASSISTANT</role>\n')
        return assistant_content, res

    def _format_standalone_tool_responses(self, tool_messages) -> 'Prompt':
        # Standalone tool messages follow a user message and are folded into the query.
        # Close HUMAN, open OBSERVATION; template prompt will re-open ASSISTANT.
        res = ['<|role_end|><role>OBSERVATION</role>']
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res.append('\n<tool_response>\n' + tool_content + '\n</tool_response>')
        return res
