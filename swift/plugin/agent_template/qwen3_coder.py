# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from typing import TYPE_CHECKING, List, Optional, Union

import json

from .hermes import HermesAgentTemplate

if TYPE_CHECKING:
    from swift.llm.infer import Function


def render_extra_keys(obj, handled_keys):
    """Helper function to render extra keys not explicitly handled"""
    result = ''
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key not in handled_keys:
                result += f'\n<{key}>{json.dumps(value, ensure_ascii=False)}</{key}>'
    return result


class Qwen3CoderAgentTemplate(HermesAgentTemplate):

    @staticmethod
    def _find_function_call(single_content: str) -> Optional['Function']:
        from swift.llm.infer import Function
        single_content = single_content.strip()
        # Check whether the complete function tag is included
        if not single_content.startswith('<function=') or not single_content.endswith('</function>'):
            return None

        # Extract function name
        func_name_match = re.search(r'<function=([^>]+)>', single_content)
        if not func_name_match:
            return None

        func_name = func_name_match.group(1).strip()
        parameters = {}

        # Use regular expressions to match parameters
        # Match any content of <parameter=name>content</parameter>
        param_pattern = r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>'
        param_matches = re.findall(param_pattern, single_content, re.DOTALL)

        for param_name, param_value in param_matches:
            # Clear the parameter values and remove any possible additional whitespace
            clean_value = param_value.strip()
            parameters[param_name.strip()] = clean_value

        return Function(name=func_name, arguments=json.dumps(parameters, ensure_ascii=False))

    def get_toolcall(self, response: str) -> List['Function']:
        # Extract the tool call parameters from the model's response
        toolcall_list = re.findall(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
        functions = []
        for toolcall in toolcall_list:
            function = self._find_function_call(toolcall)
            if function:
                functions.append(function)
        if len(functions) == 0:
            # Compat react_en
            return super(HermesAgentTemplate, self).get_toolcall(response)
        return functions

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        if system is None:
            system = 'You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.'
        tool_descs = [f'{system}\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>']
        for tool in tools:
            tool_desc = ''

            # Check function key
            if isinstance(tool, dict) and 'function' in tool:
                tool = tool['function']

            # Add function name
            tool_desc += f"<function>\n<name>{tool['name']}</name>"

            # Add description if available
            if 'description' in tool:
                tool_desc += f"\n<description>{tool['description'].strip()}</description>"

            # Add parameters section
            tool_desc += '\n<parameters>'

            # Process parameters if they exist in the expected structure
            if ('parameters' in tool and isinstance(tool['parameters'], dict) and 'properties' in tool['parameters']
                    and isinstance(tool['parameters']['properties'], dict)):

                for param_name, param_fields in tool['parameters']['properties'].items():
                    tool_desc += '\n<parameter>'
                    tool_desc += f'\n<name>{param_name}</name>'

                    if 'type' in param_fields:
                        tool_desc += f"\n<type>{str(param_fields['type'])}</type>"

                    if 'description' in param_fields:
                        tool_desc += f"\n<description>{param_fields['description'].strip()}</description>"

                    # Add any extra parameter fields
                    handled_param_keys = ['name', 'type', 'description']
                    tool_desc += render_extra_keys(param_fields, handled_param_keys)

                    tool_desc += '\n</parameter>'
            # Add any extra parameter section fields
            handled_keys = ['type', 'properties']
            if 'parameters' in tool:
                tool_desc += render_extra_keys(tool['parameters'], handled_keys)

            tool_desc += '\n</parameters>'

            # Add any extra function fields
            handled_keys = ['type', 'name', 'description', 'parameters']
            tool_desc += render_extra_keys(tool, handled_keys)

            tool_desc += '\n</function>'

            tool_descs.append(tool_desc)

        tool_descs.append(
            '</tools>\n\nIf you choose to call a function ONLY reply in the following format with '
            'NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n'
            'value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n'
            'that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n'
            'Reminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> '
            'block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n'
            '- You may provide optional reasoning for your function call in natural language BEFORE the function call, '
            'but NOT after\n- If there is no function call available, '
            'answer the question like normal with your current '
            'knowledge and do not tell the user about function calls\n</IMPORTANT>')
        tool_descs = '\n'.join(tool_descs)
        return tool_descs

    def _format_tool_calls(self, tool_call_messages):
        result_parts = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            result_parts.append(f"<tool_call>\n<function={tool_call['name']}>\n")
            # Processing parameters (if present)
            if 'arguments' in tool_call and tool_call['arguments']:
                for args_name, args_value in tool_call['arguments'].items():
                    result_parts.append(f'<parameter={args_name}>\n')
                    # Handle different types of parameter values
                    if isinstance(args_value, (dict, list)):
                        # For dictionaries or lists, use json formatting
                        args_value = json.dumps(args_value)
                    else:
                        # For other types, convert to strings
                        args_value = str(args_value)
                    result_parts.append(f'{args_value}\n</parameter>\n')
            # Close tags
            result_parts.append('</function>\n</tool_call>')
        return ''.join(result_parts)

    def _get_tool_responses(self, tool_messages):
        res_tool = []
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res_tool.append(f'<tool_response>\n{tool_content}\n</tool_response>\n')
        return ''.join(res_tool)
