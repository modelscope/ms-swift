# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.template import Prompt
from .glm4 import GLM4_5AgentTemplate


class FlintAgentTemplate(GLM4_5AgentTemplate):
    """Flint's tool format, taken from the checkpoint's `chat_template.jinja`.

    Parsing is identical to GLM-4.5/5 (`<tool_call>name<arg_key>..</arg_key><arg_value>..</arg_value>
    </tool_call>`); only the rendering differs: no newline between consecutive tool calls, and the
    tool preamble is a `<|system|>` turn that precedes any user-provided system prompt.
    """

    model_type = 'flint'

    @staticmethod
    def _find_function_call(single_content: str) -> Optional[Function]:
        single_content = single_content.strip()
        func_name_match = re.match(r'^([^\n<]+)', single_content)
        if not func_name_match:
            return None
        keys = re.findall(r'<arg_key>(.*?)</arg_key>', single_content, re.DOTALL)
        values = re.findall(r'<arg_value>(.*?)</arg_value>', single_content, re.DOTALL)
        if len(keys) != len(values):
            return None
        args = {k.strip(): v.strip() for k, v in zip(keys, values, strict=True)}
        return Function(name=func_name_match.group(1).strip(), arguments=json.dumps(args, ensure_ascii=False))

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_descs = [
            '# Tools\n\nYou may call one or more functions to assist with the user query.\n\n'
            'You are provided with function signatures within <tools></tools> XML tags:\n<tools>'
        ]
        # The template renders `{{ fn | tojson(ensure_ascii=False) }}` back to back, with no separator.
        tool_descs.extend(json.dumps(self.unwrap_tool(tool), ensure_ascii=False) for tool in tools)
        tool_descs.append('</tools>\n\nFor each function call, output the function name and arguments within '
                          'the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>'
                          '<arg_value>{arg-value-1}</arg_value>...</tool_call>')
        # One leading newline after `<|system|>`, one trailing newline before the next message.
        preamble = '\n' + ''.join(tool_descs) + '\n'
        if system is not None and system.strip():
            # System messages come first, then the tool preamble as its own `<|system|>` turn.
            return system.strip() + '<|system|>' + preamble
        return preamble

    def _format_tool_calls(self, tool_call_messages) -> str:
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            tool_calls.append(f"<tool_call>{tool_call['name']}")
            for arg_key, arg_value in tool_call['arguments'].items():
                if not isinstance(arg_value, str):
                    # `{{ v | tojson(ensure_ascii=False) if v is not string else v }}`
                    arg_value = json.dumps(arg_value, ensure_ascii=False)
                tool_calls.append(f'<arg_key>{arg_key}</arg_key>')
                tool_calls.append(f'<arg_value>{arg_value}</arg_value>')
            tool_calls.append('</tool_call>')
        return ''.join(tool_calls) + '<|observation|>'

    def _format_tool_responses(self, assistant_content: str, tool_messages) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super(GLM4_5AgentTemplate, self)._format_tool_responses(assistant_content, tool_messages)
        res = []
        for tool_message in tool_messages:
            res.append(f"<tool_response>{tool_message['content']}</tool_response>")
        res.append('<|assistant|>')
        return assistant_content, res

    def _format_standalone_tool_responses(self, tool_messages) -> 'Prompt':
        res = ['<|observation|>']
        for tool_message in tool_messages:
            res.append(f"<tool_response>{tool_message['content']}</tool_response>")
        return res
