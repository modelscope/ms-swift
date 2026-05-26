# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.template import Prompt
from .base import BaseAgentTemplate


class MiniCPM5AgentTemplate(BaseAgentTemplate):
    """Agent template for MiniCPM5 models using XML-based function calling format.

    Tool call format:
        <function name="function-name"><param name="param-name">param-value</param></function>

    Tool response format:
        <tool_response>
        response_content
        </tool_response>
    """

    def get_toolcall(self, response: str) -> List[Function]:
        # Match <function name="...">...</function> blocks
        func_pattern = re.compile(r'<function\s+name="([^"]+)">(.*?)</function>', re.DOTALL)
        param_pattern = re.compile(r'<param\s+name="([^"]+)">'
                                   r'(?:<!\[CDATA\[(.*?)\]\]>|([^<]*))'
                                   r'</param>', re.DOTALL)

        functions = []
        for func_match in func_pattern.finditer(response):
            func_name = func_match.group(1)
            func_body = func_match.group(2)
            arguments = {}
            for param_match in param_pattern.finditer(func_body):
                param_name = param_match.group(1)
                # CDATA value or plain value
                param_value = param_match.group(2) if param_match.group(2) is not None else param_match.group(3)
                # Try to parse as JSON value (number, bool, etc.)
                try:
                    param_value = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    pass
                arguments[param_name] = param_value
            functions.append(Function(name=func_name, arguments=arguments))

        if len(functions) == 0:
            # Fallback to ReAct-style parsing
            return super().get_toolcall(response)
        return functions

    def _get_tool_responses(self, tool_messages):
        res_tool = []
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res_tool.append(f'<tool_response>\n{tool_content}\n</tool_response>')
        return '\n'.join(res_tool)

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        if hasattr(self, 'template_meta'):
            prompt = self.template_meta.prompt
            chat_sep = self.template_meta.chat_sep
        else:
            prompt = ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n']
            chat_sep = ['<|im_end|>\n']
        res = chat_sep.copy()
        total_tool = self._get_tool_responses(tool_messages)
        for context in prompt:
            if isinstance(context, str):
                context = context.replace('{{QUERY}}', total_tool)
            res.append(context)
        return assistant_content, res

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_descs = [json.dumps(self.unwrap_tool(tool), ensure_ascii=False) for tool in tools]
        system = system or ''
        if system:
            system = f'{system}\n\n'
        return (f'{system}# Tools\n\n'
                'You are provided with function signatures within <tools></tools> XML tags:\n'
                '<tools>\n' + '\n'.join(tool_descs) + '\n</tools>\n\n'
                'Tool usage guidelines:\n'
                '- You may call zero or more functions. If no function calls are needed, '
                'just answer normally and do not include any <function ... </function>.\n'
                '- When calling a function, return an XML object within <function ... </function> using:\n'
                '<function name="function-name"><param name="param-name">param-value</param></function>\n'
                '- param-value may be multi-line. If it contains <, & or newline characters, '
                'wrap it in a CDATA block: <param name="param-name"><![CDATA[...multi-line value...]]></param>')

    def _format_tool_calls(self, tool_call_messages) -> str:
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            name = tool_call['name']
            arguments = tool_call['arguments']
            params_xml = ''
            if isinstance(arguments, dict):
                for param_name, param_value in arguments.items():
                    value_str = param_value if isinstance(param_value, str) else json.dumps(
                        param_value, ensure_ascii=False)
                    if isinstance(param_value, str) and ('<' in param_value or '&' in param_value
                                                         or '\n' in param_value):
                        params_xml += f'<param name="{param_name}"><![CDATA[{value_str}]]></param>'
                    else:
                        params_xml += f'<param name="{param_name}">{value_str}</param>'
            tool_calls.append(f'<function name="{name}">{params_xml}</function>')
        return '\n'.join(tool_calls)
