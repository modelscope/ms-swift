# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.infer_engine.protocol import NamespacedFunction
from swift.template import Prompt
from .base import BaseAgentTemplate

DSML_TOKEN = '｜DSML｜'

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. \
You can invoke tools by writing a "<{dsml_token}{calls_tag}>" block like the following:

<{dsml_token}{calls_tag}>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}{calls_tag}>

String parameters should be specified as is and set `string="true"`. \
For all other types (numbers, booleans, arrays, objects), \
pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), \
you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def _to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps(value, ensure_ascii=True)


def _encode_arguments_to_dsml(arguments: Dict[str, Any], dsml_token: str = DSML_TOKEN) -> str:
    """Encode tool call arguments dict into DSML parameter lines."""
    lines = []
    for k, v in arguments.items():
        is_str = 'true' if isinstance(v, str) else 'false'
        val = v if isinstance(v, str) else _to_json(v)
        lines.append(f'<{dsml_token}parameter name="{k}" string="{is_str}">{val}</{dsml_token}parameter>')
    return '\n'.join(lines)


class DeepSeekV4AgentTemplate(BaseAgentTemplate):

    dsml_token = DSML_TOKEN
    calls_tag = 'tool_calls'

    def get_toolcall(self, response: str) -> List[Function]:
        # Parse DSML tool calls from model output
        dsml_token = re.escape(self.dsml_token)
        invoke_pattern = re.compile(rf'<{dsml_token}invoke\s+name="([^"]+)">\s*(.*?)\s*</{dsml_token}invoke>',
                                    re.DOTALL)
        param_pattern = re.compile(
            rf'<{dsml_token}parameter\s+name="([^"]+)"\s+string="(true|false)">'
            rf'(.*?)</{dsml_token}parameter>', re.DOTALL)

        functions = []
        for match in invoke_pattern.finditer(response):
            tool_name = match.group(1)
            params_block = match.group(2)
            arguments = {}
            for pm in param_pattern.finditer(params_block):
                param_name = pm.group(1)
                is_string = pm.group(2)
                param_value = pm.group(3)
                if is_string == 'false':
                    try:
                        param_value = json.loads(param_value)
                    except json.JSONDecodeError:
                        pass
                arguments[param_name] = param_value
            functions.append(Function(name=tool_name, arguments=json.dumps(arguments, ensure_ascii=False)))

        if len(functions) == 0:
            # Fallback to ReAct format
            return super().get_toolcall(response)
        return functions

    def _get_tool_responses(self, tool_messages):
        # The official encoding merges tool results into one user turn, joining the
        # `<tool_result>` blocks with a blank line.
        return '\n\n'.join(f'<tool_result>{tool_message["content"]}</tool_result>' for tool_message in tool_messages)

    def _add_tool_call_prefix(self, tool_content: str, pre_message=None) -> str:
        # The official encoding always renders the tool_calls block as `\n\n` + block,
        # right after the assistant's (possibly empty) textual content.
        return '\n\n' + tool_content

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        res = [
            '<｜end▁of▁sentence｜><｜User｜>',
            self._get_tool_responses(tool_messages),
            '<｜Assistant｜>',
        ]
        return assistant_content, res

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_schemas = []
        for tool in tools:
            tool = self.unwrap_tool(tool)
            tool_schemas.append(_to_json(tool))

        tools_section = TOOLS_TEMPLATE.format(
            tool_schemas='\n'.join(tool_schemas),
            dsml_token=self.dsml_token,
            calls_tag=self.calls_tag,
        )

        system = system or ''
        return f'{system}\n\n{tools_section}' if system else tools_section

    def _format_tool_calls(self, tool_call_messages) -> str:
        invocations = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            name = tool_call['name']
            arguments = tool_call['arguments']
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            dsml_args = _encode_arguments_to_dsml(arguments, self.dsml_token)
            invocations.append(f'<{self.dsml_token}invoke name="{name}">\n{dsml_args}\n</{self.dsml_token}invoke>')

        tool_calls_str = '\n'.join(invocations)
        return f'<{self.dsml_token}{self.calls_tag}>\n{tool_calls_str}\n</{self.dsml_token}{self.calls_tag}>'


class DeepSeekV41AgentTemplate(DeepSeekV4AgentTemplate):
    # V4.1 uses leading-space tag names, including ` calls` instead of `tool_calls`.
    dsml_token = f'{DSML_TOKEN} '
    calls_tag = 'calls'

    @staticmethod
    def _split_tool_name(name, namespace=None):
        if isinstance(namespace, dict):
            namespace = namespace['name']
        prefix, separator, bare_name = name.partition('::')
        if separator:
            if namespace is not None and namespace != prefix:
                raise ValueError(f'Conflicting tool namespaces: {namespace} != {prefix}')
            namespace, name = prefix, bare_name
        if '::' in name or namespace is not None and '::' in namespace:
            raise ValueError('Tool names support a single namespace::name qualifier.')
        return namespace, name

    @classmethod
    def _qualified_tool_name(cls, tool):
        namespace, name = cls._split_tool_name(tool['name'], tool.get('namespace'))
        return name if namespace is None else f'{namespace}::{name}'

    @classmethod
    def unwrap_tool(cls, tool):
        function = dict(super().unwrap_tool(tool))
        if tool.get('namespace') is not None:
            function['namespace'] = tool['namespace']
        function['name'] = cls._qualified_tool_name(function)
        namespace = function.pop('namespace', None)
        if isinstance(namespace, dict) and namespace.get('description'):
            function['description'] = namespace['description'] + '\n' + (function.get('description') or '')
        return function

    @classmethod
    def _parse_tool_call(cls, content):
        content = dict(cls._parse_json(content))
        content['name'] = cls._qualified_tool_name(content)
        return super()._parse_tool_call(content)

    def _format_tools(self, tools, system=None, user_message=None):
        result = super()._format_tools(tools, system, user_message)
        return result if system else '\n\n' + result

    def get_toolcall(self, response: str) -> List[Function]:
        functions = []
        for function in super().get_toolcall(response):
            namespace, name = self._split_tool_name(function.name)
            if namespace is not None:
                function = NamespacedFunction(name=name, arguments=function.arguments, namespace=namespace)
            functions.append(function)
        return functions
