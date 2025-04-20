# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import re
from typing import Any, Dict, List, Optional, Union

import json

from .base import BaseAgentTemplate


class HermesAgentTemplate(BaseAgentTemplate):

    def get_toolcall(self, response: str) -> List['Function']:
        from swift.llm.infer.protocol import Function
        res_list = re.findall(r'<tool_call>(.+?)</tool_call>', response, re.DOTALL)
        functions = []
        for res in res_list:
            try:
                res = json.loads(res)
            except json.JSONDecodeError:
                try:
                    res = ast.literal_eval(res)
                except Exception:
                    continue
            functions.append(Function(name=res['name'], arguments=res['arguments']))
        if len(functions) == 0:
            # compat react_en
            return super().get_toolcall(response)
        return functions

    def _format_tool_messages(
        self,
        assistant_content: str,
        tool_messages: List[str],
    ) -> str:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_messages(assistant_content, tool_messages)
        res = ['<|im_end|>\n<|im_start|>user']
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res.append(f'\n<tool_response>\n{tool_content}\n</tool_response>')
        res.append('<|im_end|>\n<|im_start|>assistant\n')
        return assistant_content, ''.join(res)

    def _format_system(self, tools: List[Union[str, dict]], system: str) -> str:
        tool_descs = [json.dumps({'type': 'function', 'function': tool}, ensure_ascii=False) for tool in tools]
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
