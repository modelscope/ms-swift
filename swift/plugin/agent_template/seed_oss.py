import re
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import json

from .base import BaseAgentTemplate

if TYPE_CHECKING:
    from swift.llm.infer import Function
    from swift.llm.template import Prompt


class SeedAgentTemplate(BaseAgentTemplate):
    TOOL_CALL_START = '<seed:tool_call>'
    TOOL_CALL_END = '</seed:tool_call>'
    FUNCTION_TAG = 'function'
    PARAMETER_TAG = 'parameter'

    _PY_TYPE_MAPPING = {
        'string': 'str',
        'number': 'int',
        'integer': 'int',
        'boolean': 'bool',
        'array': 'list',
    }

    @staticmethod
    def _py_type(t: str) -> str:
        return SeedAgentTemplate._PY_TYPE_MAPPING.get(t, 'Any')

    def get_toolcall(self, response: str) -> List['Function']:
        from swift.llm.infer import Function

        res_list = re.findall(rf'{self.TOOL_CALL_START}(.+?){self.TOOL_CALL_END}', response, re.DOTALL)
        if not res_list:
            return super().get_toolcall(response)

        functions = []
        for res in res_list:
            func_name_match = re.search(rf'<{self.FUNCTION_TAG}=([^>]+)>', res)
            if not func_name_match:
                continue

            func_name = func_name_match.group(1)
            param_matches = re.findall(rf'<{self.PARAMETER_TAG}=([^>]+)>(.*?)</{self.PARAMETER_TAG}>', res, re.DOTALL)
            arguments = {name: value for name, value in param_matches}
            functions.append(Function(name=func_name, arguments=arguments))

        return functions

    def _get_tool_responses(self, tool_messages: List[dict]) -> str:
        responses = [f"<seed:bos>tool\n{tool_message['content']}<seed:eos>" for tool_message in tool_messages]
        return ''.join(responses) + '<seed:bos>assistant\n'

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages: List[dict],
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)

        formatted_tool_responses = self._get_tool_responses(tool_messages)
        return assistant_content, ['<seed:eos>', formatted_tool_responses]

    def _build_tool_def_string(self, tool: dict) -> str:
        """Helper to build a single tool definition string."""
        func = tool.get('function', {})
        func_name = func.get('name')

        if not func_name:
            return ''

        parameters = func.get('parameters', {})
        properties = parameters.get('properties', {})
        params = [
            f"{name}: {self._py_type(spec.get('type', 'any'))}" for name, spec in properties.items()
            if isinstance(spec, dict)
        ]
        param_str = ','.join(params)

        docstring_parts = ['    """', f'    {func.get("description", "").strip()}']

        if properties:
            docstring_parts.append('\n    Args:')
            required_params = parameters.get('required', [])
            for name, spec in properties.items():
                if isinstance(spec, dict):
                    req_tag = '[必填]' if name in required_params else '[选填]'
                    desc = spec.get('description', '')
                    type_str = self._py_type(spec.get('type', 'any'))
                    docstring_parts.append(f'    - {name} ({type_str}) {req_tag}: {desc}')

        returns_props = func.get('returns', {}).get('properties', {})
        if returns_props:
            docstring_parts.append('\n    Returns:')
            for name, spec in returns_props.items():
                desc = spec.get('description', '')
                type_str = self._py_type(spec.get('type', 'any'))
                docstring_parts.append(f'    - {name} ({type_str}): {desc}')

        docstring_parts.append('\n    """')
        docstring = '\n'.join(docstring_parts)

        return f'Function:\ndef {func_name}({param_str}):\n{docstring}'

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        if not tools:
            return system or ''

        tool_defs = [
            tool_def for tool in tools if (wrapped_tool := self.wrap_tool(tool)).get('type') == 'function' and
            (tool_def := self._build_tool_def_string(wrapped_tool)) != ''
        ]
        tool_defs_joined = '\n\n'.join(tool_defs)

        tool_call_format_instruction = (
            '工具调用请遵循如下格式:\n'
            f'{self.TOOL_CALL_START}\n'
            f'<{self.FUNCTION_TAG}=example_function_name>\n'
            f'<{self.PARAMETER_TAG}=example_parameter_1>value_1</{self.PARAMETER_TAG}>\n'
            f'<{self.PARAMETER_TAG}=example_parameter_2>This is the value for the second parameter\n'
            'that can span\n'
            f'multiple lines</{self.PARAMETER_TAG}>\n'
            f'</{self.FUNCTION_TAG}>\n'
            f'{self.TOOL_CALL_END}')

        split_token = '<seed:eos><seed:bos>system'

        if system and split_token in system:
            parts = system.split(split_token, 1)
            return f'{parts[0]}\n\n{tool_defs_joined}\n{tool_call_format_instruction}\n{split_token}{parts[1]}'
        else:
            doubao_prompt = ('You are Doubao, a helpful AI assistant. '
                             'You may call one or more functions to assist with the user query.')
            return (f'{doubao_prompt}\n\n{tool_defs_joined}\n{tool_call_format_instruction}\n'
                    f'{split_token}\n{system or ""}')

    def _format_tool_calls(self, tool_call_messages: List[dict]) -> str:
        formatted_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            func_name = tool_call['name']
            arguments = tool_call.get('arguments', {})

            call_parts = [f'<{self.FUNCTION_TAG}={func_name}>']
            for arg_name, arg_value in arguments.items():
                arg_value_str = arg_value if isinstance(arg_value, str) else json.dumps(arg_value, ensure_ascii=False)
                call_parts.append(f'<{self.PARAMETER_TAG}={arg_name}>{arg_value_str}</{self.PARAMETER_TAG}>')

            call_parts.append(f'</{self.FUNCTION_TAG}>')
            call_parts_joined = '\n'.join(call_parts)

            full_call = f'{self.TOOL_CALL_START}\n{call_parts_joined}\n{self.TOOL_CALL_END}'
            formatted_calls.append(full_call)
        return '\n'.join(formatted_calls)
