# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.template import Prompt, Tool
from .base import BaseAgentTemplate


class TeleChat3AgentTemplate(BaseAgentTemplate):

    @staticmethod
    def _dump_tool(tool) -> str:
        return json.dumps(tool, ensure_ascii=False)

    @staticmethod
    def _dump_arg_value(value) -> str:
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    def get_toolcall(self, response: str) -> List[Function]:
        functions = []
        for tool_call in re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', response, re.DOTALL):
            tool_call = self._parse_json(tool_call.strip())
            if isinstance(tool_call, dict) and 'name' in tool_call:
                functions.append(Function(name=tool_call['name'], arguments=tool_call.get('arguments') or {}))
        return functions

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_descs = '\n'.join(self._dump_tool(tool) for tool in tools)
        tool_instruction = ('\n\n# 可用工具\n'
                            '你可以调用<tools></tools>标签中包含的一个或多个工具来辅助你回答问题,以下是可用工具详情：\n'
                            '<tools>\n'
                            f'{tool_descs}\n'
                            '</tools>\n\n'
                            '# 调用方法\n'
                            '你需要遵循工具的要求，使用json格式返回工具名称及参数，并用<tool_call></tool_call>包含。下方是一个调用模板：\n'
                            '<tool_call>\n'
                            '{"name": <function-name>, "arguments": <args-json-object>}\n'
                            '</tool_call>\n')
        return (system or '') + tool_instruction

    def _format_tool_calls(self, tool_call_messages) -> str:
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            tool_calls.append(f'<tool_call>\n{json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>')
        return '\n'.join(tool_calls)

    @staticmethod
    def _to_prompt(content: Union[str, Prompt]) -> Prompt:
        return content if isinstance(content, list) else [content]

    def _format_tool_responses(self, assistant_content: Union[str, Prompt], tool_messages) -> Tuple[Prompt, Prompt]:
        res: Prompt = []
        for i, tool_message in enumerate(tool_messages):
            tool_content = tool_message['content']
            if i == 0:
                res.append('<_user><tool_response>\n')
            else:
                res.append('\n<tool_response>\n')
            res.append(tool_content)
            res.append('\n</tool_response>')
        res.append('<_bot>')
        return self._to_prompt(assistant_content) + ['<_end>\n'], res


class TeleChat3CoderAgentTemplate(TeleChat3AgentTemplate):

    def _add_tool_call_prefix(self, tool_content: str, pre_message=None) -> str:
        if pre_message is None or pre_message['role'] != 'assistant':
            return '</think>' + tool_content
        return tool_content

    def get_toolcall(self, response: str) -> List[Function]:
        return self.get_toolcall_with_tools(response)

    def get_toolcall_with_tools(self, response: str, tools: Optional[List[Tool]] = None) -> List[Function]:
        functions = []
        for block in re.findall(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL):
            name, args = self._parse_coder_tool_call(block, tools)
            if name:
                functions.append(Function(name=name, arguments=args))
        return functions

    def _parse_coder_tool_call(self, block: str, tools: Optional[List[Tool]] = None):
        name = re.split(r'<param_key>', block, maxsplit=1)[0].strip()
        args = {}
        pattern = re.compile(r'<param_key>(.*?)</param_key>\s*<param_value>(.*?)</param_value>', re.DOTALL)
        for key, value in pattern.findall(block):
            key = key.strip()
            value = value.strip()
            if self._is_string_arg(name, key, tools):
                args[key] = value
            else:
                parsed_value = self._parse_json(value)
                args[key] = parsed_value if parsed_value is not None or value in {'null', 'None'} else value
        return name, args

    def _is_string_arg(self, tool_name, arg_name, tools):
        # Match the model's official parser: only an exact schema type of "string" keeps the raw value.
        for tool in tools or []:
            if isinstance(tool, str):
                tool = self._parse_json(tool)
            if not isinstance(tool, dict):
                continue
            tool = self.unwrap_tool(tool)
            if self._get_tool_name(tool) != tool_name:
                continue
            parameters = self._parse_json(tool.get('parameters') or {})
            if not isinstance(parameters, dict):
                return False
            properties = parameters.get('properties') or {}
            if not isinstance(properties, dict):
                return False
            arg_schema = properties.get(arg_name) or {}
            return isinstance(arg_schema, dict) and arg_schema.get('type') == 'string'
        return False

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_descs = '\n'.join(self._dump_tool(tool) for tool in tools)
        tool_instruction = (
            '\n# Tools\n\n'
            'You may call one or more functions to assist with the user query.\n\n'
            'You are provided with function signatures within <tools></tools> XML tags:\n'
            '<tools>\n'
            f'{tool_descs}\n'
            '</tools>\n\n'
            'For each function call, output the function name and arguments within the following XML format:\n'
            '<tool_call>{function-name}<param_key>{param-key-1}</param_key><param_value>{param-value-1}</param_value>'
            '<param_key>{param-key-2}</param_key><param_value>{param-value-2}</param_value>...</tool_call>')
        return (system or '') + tool_instruction

    def _format_tool_calls(self, tool_call_messages) -> str:
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            arguments = tool_call.get('arguments') or {}
            parts = [f'<tool_call>{tool_call["name"]}']
            for key, value in arguments.items():
                parts.append(f'<param_key>{key}</param_key><param_value>{self._dump_arg_value(value)}</param_value>')
            parts.append('</tool_call>')
            tool_calls.append(''.join(parts))
        return ''.join(tool_calls)

    def _format_tool_responses(self, assistant_content: Union[str, Prompt], tool_messages) -> Tuple[Prompt, Prompt]:
        res: Prompt = ['<_observation>']
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res += ['<tool_response>', tool_content, '</tool_response>']
        res.append('<_bot>')
        return self._to_prompt(assistant_content) + ['<_end>'], res
