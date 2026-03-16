# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.template import Prompt
from .base import BaseAgentTemplate


class MinimaxM2AgentTemplate(BaseAgentTemplate):
    """
    Agent template for MiniMax-M2 series models.

    This template handles tool calling in MiniMax's XML-based format:
    <minimax:tool_call>
    <invoke name="tool-name">
    <parameter name="param-key">param-value</parameter>
    </invoke>
    </minimax:tool_call>
    """

    def get_toolcall(self, response: str) -> List[Function]:
        """
        Extract tool calls from MiniMax response format.

        Format:
        <minimax:tool_call>
        <invoke name="tool-name">
        <parameter name="param-key">param-value</parameter>
        </invoke>
        </minimax:tool_call>
        """
        functions = []

        # Find all tool_call blocks
        tool_call_blocks = re.findall(r'<minimax:tool_call>(.*?)</minimax:tool_call>', response, re.DOTALL)

        for block in tool_call_blocks:
            # Find all invoke blocks within the tool_call
            invoke_blocks = re.findall(r'<invoke name="([^"]+)">(.*?)</invoke>', block, re.DOTALL)

            for tool_name, params_block in invoke_blocks:
                # Extract parameters
                params = {}
                param_matches = re.findall(r'<parameter name="([^"]+)">(.*?)</parameter>', params_block, re.DOTALL)

                for param_name, param_value in param_matches:
                    param_value = param_value.strip()
                    # Try to parse as JSON if it looks like a JSON structure
                    parsed_value = self._parse_json(param_value)
                    params[param_name] = parsed_value if parsed_value is not None else param_value

                functions.append(Function(name=tool_name, arguments=params))

        # Fallback to react format if no functions found
        if len(functions) == 0:
            return super().get_toolcall(response)

        return functions

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        """
        Format tool execution results in MiniMax format.

        Tool responses are wrapped in <response></response> tags.
        """
        # Check if using react format
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)

        # Use template meta if available
        if hasattr(self, 'template_meta'):
            prompt = self.template_meta.prompt.copy()
            chat_sep = self.template_meta.chat_sep
            for i in range(len(prompt)):
                if isinstance(prompt[i], str):
                    prompt[i] = prompt[i].replace('user', 'tool')
        else:
            # Default format based on the Jinja2 template
            prompt = [']~b]tool\n{{QUERY}}[e~[\n']
            chat_sep = ['[e~[\n']

        res = chat_sep.copy()

        # Format tool responses
        tool_responses = []
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            tool_responses.append(f'<response>{tool_content}</response>')

        total_tool = '\n'.join(tool_responses)

        for context in prompt:
            if isinstance(context, str):
                context = context.replace('{{QUERY}}', total_tool)
            res.append(context)

        return assistant_content, res

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        """
        Format tools in MiniMax format with JSONSchema and XML invocation examples.
        """
        # Parse tools to JSONSchema format
        tool_schemas = []
        for tool in tools:
            tool = self.unwrap_tool(tool)
            tool_schemas.append(json.dumps(tool, ensure_ascii=False))

        system = system or ''

        return f"""{system}

# Tools
You may call one or more tools to assist with the user query.
Here are the tools available in JSONSchema format:

<tools>
""" + '\n'.join(f'<tool>{schema}</tool>' for schema in tool_schemas) + """
</tools>

When making tool calls, use XML format to invoke tools and pass parameters:

<minimax:tool_call>
<invoke name="tool-name-1">
<parameter name="param-key-1">param-value-1</parameter>
<parameter name="param-key-2">param-value-2</parameter>
...
</invoke>
</minimax:tool_call>"""

    def _format_tool_calls(self, tool_call_messages):
        """
        Format tool call messages into MiniMax XML format.

        Args:
            tool_call_messages: List of messages containing tool call information.

        Returns:
            Formatted string with tool calls in MiniMax XML format.
        """
        tool_calls = []

        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            name = tool_call['name']
            arguments = tool_call['arguments']

            # Build parameter list
            params = []
            for key, value in arguments.items():
                # Convert value to JSON string if it's not a string
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                params.append(f'<parameter name="{key}">{value}</parameter>')

            # Build invoke block
            invoke_block = f'<invoke name="{name}">\n' + '\n'.join(params) + '\n</invoke>'
            tool_calls.append(invoke_block)

        # Wrap all invocations in tool_call tags
        if tool_calls:
            return '<minimax:tool_call>\n' + '\n'.join(tool_calls) + '\n</minimax:tool_call>'

        return ''
