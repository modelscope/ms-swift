# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.template import Prompt
from .base import BaseAgentTemplate


class KimiK3AgentTemplate(BaseAgentTemplate):
    """Agent template for Kimi K3 models (XTML format).

    Aligned with `encoding_k3.py` in the model repo:
    - Tools are declared in a separate system message with `type="tool-declare"`,
      carrying the deep-sorted JSONSchema in a compact ```json block.
    - Tool calls follow the closed response channel:
      <|close|>response<|sep|><|open|>tools<|sep|>
      <|open|>call tool="{name}" index="{i}"<|sep|>
      <|open|>argument key="{key}" type="{type}"<|sep|>{value}<|close|>argument<|sep|>
      <|close|>call<|sep|><|close|>tools<|sep|>
    - Tool response:
      <|open|>message role="tool" tool="{name}" index="{i}"<|sep|>{content}<|close|>message<|sep|><|end_of_msg|>
    """

    @staticmethod
    def _deep_sort(obj):
        if isinstance(obj, dict):
            return {k: KimiK3AgentTemplate._deep_sort(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            return [KimiK3AgentTemplate._deep_sort(item) for item in obj]
        return obj

    @staticmethod
    def _escape_attr(value) -> str:
        return str(value).replace('&', '&amp;').replace('"', '&quot;')

    @staticmethod
    def _unescape_attr(value: str) -> str:
        return value.replace('&quot;', '"').replace('&amp;', '&')

    @staticmethod
    def _xtml_type(value) -> str:
        if isinstance(value, bool):
            return 'boolean'
        if value is None:
            return 'null'
        if isinstance(value, (int, float)):
            return 'number'
        if isinstance(value, str):
            return 'string'
        if isinstance(value, dict):
            return 'object'
        return 'array'

    @staticmethod
    def _xtml_value(value) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def get_toolcall(self, response: str) -> List[Function]:
        call_pattern = r'<\|open\|>call tool="(.*?)"(?: index="\d+")?<\|sep\|>(.*?)<\|close\|>call<\|sep\|>'
        arg_pattern = r'<\|open\|>argument key="(.*?)" type="(.*?)"<\|sep\|>(.*?)<\|close\|>argument<\|sep\|>'
        json_pattern = r'<\|open\|>json type="object"<\|sep\|>(.*?)<\|close\|>json<\|sep\|>'
        functions = []
        for name, body in re.findall(call_pattern, response, re.DOTALL):
            name = self._unescape_attr(name)
            json_match = re.search(json_pattern, body, re.DOTALL)
            if json_match:
                arguments = self._parse_json(json_match.group(1))
                if arguments is None:
                    arguments = json_match.group(1)
            else:
                arguments = {}
                for key, arg_type, value in re.findall(arg_pattern, body, re.DOTALL):
                    key = self._unescape_attr(key)
                    if arg_type == 'string':
                        arguments[key] = value
                    elif arg_type == 'null':
                        arguments[key] = None
                    else:
                        parsed = self._parse_json(value)
                        arguments[key] = value if parsed is None else parsed
            functions.append(Function(name=name, arguments=arguments))
        if len(functions) == 0:
            # compat react_en
            return super().get_toolcall(response)
        return functions

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tools = self._deep_sort([self.wrap_tool(tool) for tool in tools])
        tools_json = json.dumps(tools, ensure_ascii=False, separators=(',', ':'))
        tool_content = ('# Tools\nHere are the available tools, described in JSONSchema.\n\n'
                        f'```json\n{tools_json}\n```')
        # Hijack the `<|open|>message {{SYSTEM}}<|close|>message<|sep|><|end_of_msg|>` system
        # prefix so the tool-declare message is rendered before the plain system message.
        res = f'role="system" type="tool-declare"<|sep|>{tool_content}'
        if system:
            res += f'<|close|>message<|sep|><|end_of_msg|><|open|>message role="system"<|sep|>{system}'
        return res

    def _format_tool_calls(self, tool_call_messages) -> str:
        calls = []
        for index, message in enumerate(tool_call_messages, start=1):
            tool_call = self._parse_tool_call(message['content'])
            name, arguments = tool_call['name'], tool_call['arguments']
            call = [f'<|open|>call tool="{self._escape_attr(name)}" index="{index}"<|sep|>']
            if isinstance(arguments, dict):
                for key, value in arguments.items():
                    call.append(f'<|open|>argument key="{self._escape_attr(key)}" '
                                f'type="{self._xtml_type(value)}"<|sep|>'
                                f'{self._xtml_value(value)}<|close|>argument<|sep|>')
            else:
                # non-object arguments fall back to a raw json block
                json_block = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
                call.append(f'<|open|>json type="object"<|sep|>{json_block}<|close|>json<|sep|>')
            call.append('<|close|>call<|sep|>')
            calls.append(''.join(call))
        return f'<|open|>tools<|sep|>{"".join(calls)}<|close|>tools<|sep|>'

    def _add_tool_call_prefix(self, tool_content: str, pre_message=None) -> str:
        # Without a preceding assistant text message (whose converted content already ends
        # with `<|close|>response<|sep|>`), the empty response channel must be closed here.
        if pre_message is None or pre_message['role'] != 'assistant':
            tool_content = '<|close|>response<|sep|>' + tool_content
        return tool_content

    def _format_tool_responses(
        self,
        assistant_content,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        contents = assistant_content if isinstance(assistant_content, list) else [assistant_content]
        joined = ''.join(c for c in contents if isinstance(c, str))
        with_action = self.keyword.action in joined and self.keyword.action_input in joined
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        # `tool`/`index` attributes are resolved from the preceding assistant tool calls,
        # in call order (aligned with the `tool_index` enumeration in encoding_k3.py).
        names = re.findall(r'<\|open\|>call tool="(.*?)" index="\d+"<\|sep\|>', joined)
        res = ['<|close|>message<|sep|><|end_of_msg|>']
        for index, tool_message in enumerate(tool_messages, start=1):
            if index <= len(names):
                name = self._unescape_attr(names[index - 1])
            else:
                name = tool_message.get('tool') or tool_message.get('name') or ''
            res.append(f'<|open|>message role="tool" tool="{self._escape_attr(name)}" index="{index}"<|sep|>'
                       f'{tool_message["content"]}<|close|>message<|sep|><|end_of_msg|>')
        res.append('<|open|>message role="assistant"<|sep|>')
        return assistant_content, res
