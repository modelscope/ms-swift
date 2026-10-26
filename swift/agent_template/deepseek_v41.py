# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.template import Prompt
from .base import BaseAgentTemplate

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. \
You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name="$TOOL_NAME">
<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name="$TOOL_NAME2">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

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


def _encode_arguments_to_dsml(arguments: Dict[str, Any]) -> str:
    lines = []
    for name, value in arguments.items():
        is_string = 'true' if isinstance(value, str) else 'false'
        value = value if isinstance(value, str) else _to_json(value)
        lines.append(f'<｜DSML｜ parameter name="{name}" string="{is_string}">{value}</｜DSML｜ parameter>')
    return '\n'.join(lines)


class DeepSeekV41AgentTemplate(BaseAgentTemplate):

    def get_toolcall(self, response: str) -> List[Function]:
        # V4.1 has a space between the DSML token and each tag name.
        invoke_pattern = re.compile(r'<｜DSML｜ invoke\s+name="([^"]+)">\s*(.*?)\s*</｜DSML｜ invoke>', re.DOTALL)
        param_pattern = re.compile(
            r'<｜DSML｜ parameter\s+name="([^"]+)"\s+string="(true|false)">'
            r'(.*?)</｜DSML｜ parameter>', re.DOTALL)

        functions = []
        for match in invoke_pattern.finditer(response):
            tool_name, params_block = match.groups()
            arguments = {}
            for param_match in param_pattern.finditer(params_block):
                param_name, is_string, param_value = param_match.groups()
                if is_string == 'false':
                    try:
                        param_value = json.loads(param_value)
                    except json.JSONDecodeError:
                        pass
                arguments[param_name] = param_value
            functions.append(Function(name=tool_name, arguments=json.dumps(arguments, ensure_ascii=False)))

        if not functions:
            return super().get_toolcall(response)
        return functions

    def _get_tool_responses(self, tool_messages):
        return '\n\n'.join(f'<tool_result>{message["content"]}</tool_result>' for message in tool_messages)

    def _add_tool_call_prefix(self, tool_content: str, pre_message=None) -> str:
        return '\n\n' + tool_content

    def _format_tool_responses(self, assistant_content: str, tool_messages) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        prompt = [
            '<｜end▁of▁sentence｜><｜User｜>',
            self._get_tool_responses(tool_messages),
            '<｜Assistant｜>',
        ]
        return assistant_content, prompt

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_schemas = [_to_json(self.unwrap_tool(tool)) for tool in tools]
        tools_section = TOOLS_TEMPLATE.format(tool_schemas='\n'.join(tool_schemas))
        # V4.1 always separates tool schemas from system text, even when it is empty.
        return f'{system or ""}\n\n{tools_section}'

    def _format_tool_calls(self, tool_call_messages) -> str:
        invocations = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            name = tool_call['name']
            arguments = tool_call['arguments']
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            dsml_args = _encode_arguments_to_dsml(arguments)
            invocations.append(f'<｜DSML｜ invoke name="{name}">\n{dsml_args}\n</｜DSML｜ invoke>')

        tool_calls = '\n'.join(invocations)
        return f'<｜DSML｜ calls>\n{tool_calls}\n</｜DSML｜ calls>'
