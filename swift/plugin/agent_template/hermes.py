# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import json

from .base import BaseAgentTemplate

if TYPE_CHECKING:
    from swift.llm.infer import Function
    from swift.llm.template import Prompt


class HermesAgentTemplate(BaseAgentTemplate):

    def get_toolcall(self, response: str) -> List['Function']:
        from swift.llm.infer import Function
        res_list = re.findall(r'<tool_call>(.+?)</tool_call>', response, re.DOTALL)
        functions = []
        for res in res_list:
            res = self._parse_json(res)
            if isinstance(res, dict) and 'name' in res and 'arguments' in res:
                functions.append(Function(name=res['name'], arguments=res['arguments']))
        if len(functions) == 0:
            # compat react_en
            return super().get_toolcall(response)
        return functions

    def _get_tool_responses(self, tool_messages):
        res_tool = []
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res_tool.append(f'<tool_response>\n{tool_content}\n</tool_response>')
        return '\n'.join(res_tool)

    def _get_tool_calls(self, tool_calls: List[str]):
        return '\n'.join(tool_calls)

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
        tool_descs = [json.dumps(self.wrap_tool(tool), ensure_ascii=False) for tool in tools]
        system = system or ''
        return f"""{system}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + '\n'.join(tool_descs) + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

    def _format_tool_calls(self, tool_call_messages):
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            tool_calls.append(f'<tool_call>\n{json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>')
        return self._get_tool_calls(tool_calls)


class HunyuanHermesAgentTemplate(HermesAgentTemplate):

    def get_toolcall(self, response: str) -> List['Function']:
        from swift.llm.infer import Function
        res_list = re.findall(r'<tool_call>(.+?)\n```json(.+?)```</tool_call>', response, re.DOTALL)
        functions = []
        for name, arguments in res_list:
            arguments = self._parse_json(arguments)
            functions.append(Function(name=name, arguments=arguments))
        if len(functions) == 0:
            # compat react_en
            return super().get_toolcall(response)
        return functions

    def _get_tool_responses(self, tool_messages):
        res_tool = []
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res_tool.append(f'<tool_response>{tool_content}</tool_response>')
        tool_responses = '\n'.join(res_tool)
        return f'<tool_responses>{tool_responses}</tool_responses>'

    def _get_tool_calls(self, tool_calls: List[str]):
        tool_calls = '\n'.join(tool_calls)
        return f'<tool_calls>\n{tool_calls}\n</tool_calls>'

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_descs = [json.dumps(self.wrap_tool(tool), ensure_ascii=False) for tool in tools]
        system = system or ''
        if system:
            system = f'{system}\n\n'
        return f"""{system}# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + '\n'.join(tool_descs) + """
</tools>

For function call returns, you should first print <tool_calls>For each function call, you should return object like:
<tool_call>function_name
```json
function_arguments_in_json_format
```</tool_call>At the end of function call returns, you should print </tool_calls>"""
