# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.template import Prompt
from .base import BaseAgentTemplate

EOS = '<_end>'


class Xing4_0AgentTemplate(BaseAgentTemplate):
    """ref: https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B (chat_template.jinja)

    Tools are listed in the system block, and every call is an XML-ish block:
        <tool_call>{name}<param_key>{k}</param_key><param_value>{v}</param_value></tool_call>
    Observations live in an `<_observation>` turn opened by `<tool_response>`.
    """

    supports_tool_schema = True

    @staticmethod
    def _find_function_call(single_content: str, tool_schemas: Optional[dict] = None) -> Optional[Function]:
        single_content = single_content.strip()
        func_name_match = re.match(r'^([^<]+)', single_content)
        if not func_name_match:
            return None
        func_name = func_name_match.group(1).strip()
        keys = re.findall(r'<param_key>(.*?)</param_key>', single_content, re.DOTALL)
        values = re.findall(r'<param_value>(.*?)</param_value>', single_content, re.DOTALL)
        if len(keys) != len(values):
            return None
        properties = (tool_schemas or {}).get(func_name, {}).get('properties', {})
        if not isinstance(properties, dict):
            properties = {}
        args = {}
        for key, value in zip(keys, values):
            key = key.strip()
            param_schema = properties.get(key, {})
            param_type = param_schema.get('type') if isinstance(param_schema, dict) else None
            # The wire format leaves strings unquoted, including JSON-looking strings and whitespace.
            type_list = param_type if isinstance(param_type, list) else [param_type]
            is_string = 'string' in type_list
            if is_string and 'null' in type_list:
                # Bare null is ambiguous; keep decoding it as JSON null.
                is_string = value.strip() != 'null'
            if not is_string:
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    pass
            args[key] = value
        return Function(name=func_name, arguments=json.dumps(args, ensure_ascii=False))

    def get_toolcall(self, response: str, tools: Optional[List[Union[str, dict]]] = None) -> List[Function]:
        tool_schemas = self._get_tool_schemas(tools)
        toolcall_list = re.findall(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
        functions = []
        for toolcall in toolcall_list:
            function = self._find_function_call(toolcall, tool_schemas)
            if function:
                functions.append(function)
        if len(functions) == 0:
            # compat react_en
            return super().get_toolcall(response)
        return functions

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        # The jinja template dumps the tool as-is, so keep the `{'type': ..., 'function': ...}` wrapper.
        tool_descs = ['\n# Tools\n\nYou may call one or more functions to assist with the user query.']
        tool_descs.append('\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>')
        for tool in tools:
            tool_descs.append(json.dumps(tool, ensure_ascii=False))
        tool_descs.append('</tools>')
        tool_descs.append('\nFor each function call, output the function name and arguments '
                          'within the following XML format:')
        tool_descs.append('<tool_call>{function-name}<param_key>{param-key-1}</param_key>'
                          '<param_value>{param-value-1}</param_value><param_key>{param-key-2}</param_key>'
                          '<param_value>{param-value-2}</param_value>...</tool_call>')
        return (system or '') + '\n'.join(tool_descs)

    def _format_tool_calls(self, tool_call_messages) -> str:
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            tool_calls.append(f'<tool_call>{tool_call["name"]}')
            for arg_key, arg_value in tool_call['arguments'].items():
                if not isinstance(arg_value, str):
                    # `{{ v if v is string else v | tojson }}`
                    arg_value = json.dumps(arg_value, ensure_ascii=False)
                tool_calls.append(f'<param_key>{arg_key}</param_key>')
                tool_calls.append(f'<param_value>{arg_value}</param_value>')
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
        res = [f'{EOS}\n<_observation>']
        for tool_message in tool_messages:
            res.append(f'<tool_response>{tool_message["content"]}</tool_response>')
        res.append('<_bot>')
        return assistant_content, res

    def _format_standalone_tool_responses(self, tool_messages) -> 'Prompt':
        # Appended to the user query, i.e. inserted before the `<_bot>` that opens the assistant turn.
        res = ['<_observation>']
        for tool_message in tool_messages:
            res.append(f'<tool_response>{tool_message["content"]}</tool_response>')
        return res
