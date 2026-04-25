# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.template import Prompt
from .base import BaseAgentTemplate

DSML_TOKEN = '｜DSML｜'

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. \
You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. \
For all other types (numbers, booleans, arrays, objects), \
pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), \
you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls."""


def _to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps(value, ensure_ascii=True)


def _encode_arguments_to_dsml(arguments: Dict[str, Any]) -> str:
    """Encode tool call arguments dict into DSML parameter lines."""
    lines = []
    for k, v in arguments.items():
        is_str = 'true' if isinstance(v, str) else 'false'
        val = v if isinstance(v, str) else _to_json(v)
        lines.append(f'<{DSML_TOKEN}parameter name="{k}" string="{is_str}">{val}</{DSML_TOKEN}parameter>')
    return '\n'.join(lines)


class DeepSeekV4AgentTemplate(BaseAgentTemplate):

    def get_toolcall(self, response: str) -> List[Function]:
        # Parse DSML tool calls from model output
        # Pattern: <｜DSML｜invoke name="tool_name">...params...</｜DSML｜invoke>
        invoke_pattern = re.compile(
            rf'<{re.escape(DSML_TOKEN)}invoke\s+name="([^"]+)">\s*(.*?)\s*</{re.escape(DSML_TOKEN)}invoke>',
            re.DOTALL)
        param_pattern = re.compile(
            rf'<{re.escape(DSML_TOKEN)}parameter\s+name="([^"]+)"\s+string="(true|false)">'
            rf'(.*?)</{re.escape(DSML_TOKEN)}parameter>', re.DOTALL)

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
        return ''.join(f'<tool_result>{tool_message["content"]}</tool_result>' for tool_message in tool_messages)

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        res = ['<｜end▁of▁sentence｜>', self._get_tool_responses(tool_messages)]
        return assistant_content, res

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_schemas = []
        for tool in tools:
            tool = self.unwrap_tool(tool)
            tool_schemas.append(_to_json(tool))

        tools_section = TOOLS_TEMPLATE.format(
            tool_schemas='\n'.join(tool_schemas),
            dsml_token=DSML_TOKEN,
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
            dsml_args = _encode_arguments_to_dsml(arguments)
            invocations.append(f'<{DSML_TOKEN}invoke name="{name}">\n{dsml_args}\n</{DSML_TOKEN}invoke>')

        tool_calls_str = '\n'.join(invocations)
        return f'<{DSML_TOKEN}tool_calls>\n{tool_calls_str}\n</{DSML_TOKEN}tool_calls>'
