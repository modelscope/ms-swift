# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import json

from .base import BaseAgentTemplate

if TYPE_CHECKING:
    from swift.llm.infer import Function
    from swift.llm.template import Prompt


def render_extra_keys(obj, handled_keys):
    """Helper function to render extra keys not explicitly handled"""
    result = ""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key not in handled_keys:
                result += f"\n<{key}>{value}</{key}>"
    return result


class QWEN3CODER_AgentTemplate(BaseAgentTemplate):
    @staticmethod
    def _find_function_call(single_content: str) -> Optional['Function']:
        from swift.llm.infer import Function
        single_content = single_content.strip()
        # 检查是否包含完整的function标签
        if not single_content.startswith('<function=') or not single_content.endswith('</function>'):
            return None

        # 提取函数名
        func_name_match = re.search(r'<function=([^>]+)>', single_content)
        if not func_name_match:
            return None

        func_name = func_name_match.group(1).strip()
        parameters = {}

        # 使用更精确的正则表达式匹配参数
        # 匹配 <parameter=name>任意内容</parameter>
        param_pattern = r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>'
        param_matches = re.findall(param_pattern, single_content, re.DOTALL)

        for param_name, param_value in param_matches:
            # 清理参数值，移除可能的额外空白
            clean_value = param_value.strip()
            parameters[param_name.strip()] = clean_value

        return Function(name=func_name, arguments=json.dumps(parameters, ensure_ascii=False))

    def get_toolcall(self, response: str) -> List['Function']:
        # 如何从模型的回答中提取工具的调用（入参）
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

    def _format_tools(self, tools: List[Union[str, dict]], system: str, user_message=None) -> str:
        tool_descs = [
            'You have access to the following functions:\n\n'
            '<tools>'
        ]
        for tool in tools:
            tool_desc = ""

            # 判断是否有function字段
            if isinstance(tool, dict) and 'function' in tool:
                tool = tool['function']

            # Add function name
            tool_desc += f"<function>\n<name>{tool['name']}</name>"

            # Add description if available
            if 'description' in tool:
                tool_desc += f"\n<description>{tool['description'].strip()}</description>"

            # Add parameters section
            tool_desc += "\n<parameters>"

            # Process parameters if they exist in the expected structure
            if ('parameters' in tool and isinstance(tool['parameters'], dict) and
                    'properties' in tool['parameters'] and isinstance(tool['parameters']['properties'], dict)):

                for param_name, param_fields in tool['parameters']['properties'].items():
                    tool_desc += "\n<parameter>"
                    tool_desc += f"\n<name>{param_name}</name>"

                    if 'type' in param_fields:
                        tool_desc += f"\n<type>{str(param_fields['type'])}</type>"

                    if 'description' in param_fields:
                        tool_desc += f"\n<description>{param_fields['description'].strip()}</description>"

                    # Add any extra parameter fields
                    handled_param_keys = ['name', 'type', 'description']
                    tool_desc += render_extra_keys(param_fields, handled_param_keys)

                    tool_desc += "\n</parameter>"
            # Add any extra parameter section fields
            handled_keys = ['type', 'properties']
            if 'parameters' in tool:
                tool_desc += render_extra_keys(tool['parameters'], handled_keys)

            tool_desc += "\n</parameters>"

            # Add any extra function fields
            handled_keys = ['type', 'name', 'description', 'parameters']
            tool_desc += render_extra_keys(tool, handled_keys)

            tool_desc += "\n</function>"

            tool_descs.append(tool_desc)

        tool_descs.append('</tools>\n\nIf you choose to call a function ONLY reply in the following format with '
                          'NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n'
                          'value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n'
                          'that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n'
                          'Reminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> '
                          'block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n'
                          '- You may provide optional reasoning for your function call in natural language BEFORE the function call, '
                          'but NOT after\n- If there is no function call available, answer the question like normal with your current '
                          'knowledge and do not tell the user about function calls\n</IMPORTANT>')
        tool_descs = '\n'.join(tool_descs)
        if system.strip():
            tool_descs = '<|system|>\n' + system.strip() + '\n\n' + tool_descs
        return tool_descs

    def _format_tool_calls(self, tool_call_messages):
        result = ''
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            result += f"<tool_call>\n<function={tool_call['name']}>\n"
            # 处理参数（如果存在）
            if 'arguments' in tool_call and tool_call['arguments']:
                for args_name, args_value in tool_call['arguments'].items():
                    result += f"<parameter={args_name}>\n"
                    # 处理不同类型的参数值
                    if isinstance(args_value, (dict, list)):
                        # 对于字典或列表，使用json格式化
                        args_value = json.dumps(args_value)
                    else:
                        # 对于其他类型，转换为字符串
                        args_value = str(args_value)
                    result += f"{args_value}\n</parameter>\n"
            # 关闭标签
            result += "</function>\n</tool_call>"
        return result

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