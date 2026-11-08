# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import List, Optional, Tuple, Union

import json

from ..utils import Prompt
from .base import BaseAgentTemplate
from .protocol import Function


class HermesAgentTemplate(BaseAgentTemplate):
    tool_call_start = "<tool_call>"

    def get_toolcall(self, response: str) -> List[Function]:
        res_list = re.findall(r"<tool_call>(.+?)</tool_call>", response, re.DOTALL)
        functions = []
        for res in res_list:
            res = self._parse_json(res)
            if isinstance(res, dict) and "name" in res and "arguments" in res:
                functions.append(Function(name=res["name"], arguments=res["arguments"]))
        if len(functions) == 0:
            # compat react_en
            return super().get_toolcall(response)
        return functions

    def _get_tool_responses(self, tool_messages):
        res_tool = []
        for tool_message in tool_messages:
            tool_content = tool_message["content"]
            res_tool.append(f"<tool_response>\n{tool_content}\n</tool_response>")
        return "\n".join(res_tool)

    def _get_tool_calls(self, tool_calls: List[str]):
        return "\n".join(tool_calls)

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, "Prompt"]:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        if hasattr(self, "template_meta"):
            prompt = self.template_meta.prompt
        else:
            prompt = ["<|im_start|>user\n{{QUERY}}<|im_end|>\n"]
        # no separator prefix: the encode loop emits chat_sep after the previous response, and
        # the opener (split out of prompt) right before this round's response
        res = []
        total_tool = self._get_tool_responses(tool_messages)
        for context in prompt:
            if isinstance(context, str):
                context = context.replace("{{QUERY}}", total_tool)
            res.append(context)
        return assistant_content, res

    def _format_tool_user_followup(self, tool_messages, user_messages) -> 'Prompt':
        prompt = self.template_meta.prompt
        chat_sep = self.template_meta.chat_sep or []
        contents = [self._get_tool_responses(tool_messages)] + [message['content'] for message in user_messages]
        # the official jinja prints every content of this run inside ONE user turn: the separator
        # goes between rounds, never between the tool response and the follow-up user content
        res = []
        for i, content in enumerate(contents):
            for context in prompt:
                if isinstance(context, str) and '{{QUERY}}' in context:
                    prefix, suffix = context.split('{{QUERY}}', 1)
                    res.append(prefix)
                    res.append(content)
                    if i < len(contents) - 1:
                        res.extend(chat_sep)
                else:
                    res.append(context)
        if contents:
            res.append(suffix)
        return res

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_descs = [json.dumps(self.wrap_tool(tool), ensure_ascii=False) for tool in tools]
        system = system or ""
        return (
            f"""{system}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
"""
            + "\n".join(tool_descs)
            + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""
        )

    def _format_tool_calls(self, tool_call_messages):
        tool_calls = []
        for message in tool_call_messages:
            # The raw payload is kept so json.dumps reproduces the model jinja's tojson for both
            # argument forms: a spec-form str renders as an escaped JSON string, a dict inline.
            tool_call = self._raw_tool_call(message["content"])
            tool_calls.append(f"<tool_call>\n{json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>")
        return self._get_tool_calls(tool_calls)


class Qwen3AgentTemplate(HermesAgentTemplate):
    """Qwen3 hermes variant: string arguments render verbatim.

    The Qwen2.5 jinjas pass ``arguments | tojson`` (a spec-form string renders escaped); the
    Qwen3 jinjas branch on the type and print a string as-is. A dict is inline in both.
    """

    def _format_tool_calls(self, tool_call_messages):
        tool_calls = []
        for message in tool_call_messages:
            payload = self._raw_tool_call(message["content"])
            arguments = payload["arguments"]
            if isinstance(arguments, str):
                rendered = '{"name": ' + json.dumps(payload["name"], ensure_ascii=False) + ', "arguments": ' + arguments + '}'
            else:
                rendered = json.dumps(payload, ensure_ascii=False)
            tool_calls.append(f"<tool_call>\n{rendered}\n</tool_call>")
        return self._get_tool_calls(tool_calls)


class HunyuanHermesAgentTemplate(HermesAgentTemplate):
    tool_call_start = "<tool_calls>"

    def _format_tool_user_followup(self, tool_messages, user_messages) -> 'Prompt':
        _, tool_content = self._format_tool_responses('', tool_messages)
        prefix, suffix = ''.join(self.template_meta.prompt).split('{{QUERY}}', 1)
        # the follow-up user content continues inside the tool_responses block: this jinja branch
        # drops its closing marker before the next user turn
        tool_content = [
            item.replace('</tool_responses>', '') if isinstance(item, str) else item for item in tool_content
        ]
        last = tool_content[-1]
        if suffix and isinstance(last, str) and last.endswith(suffix):
            tool_content[-1] = last[:-len(suffix)]
        # this jinja closes every user turn with the assistant opener, not just the last one
        res = list(tool_content)
        for i, message in enumerate(user_messages):
            res.append(prefix + message['content'])
            if i < len(user_messages) - 1:
                res.append(suffix)
        res.append(suffix)
        return res

    def get_toolcall(self, response: str) -> List[Function]:
        res_list = re.findall(r"<tool_call>(.+?)\n```json(.+?)```</tool_call>", response, re.DOTALL)
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
            tool_content = tool_message["content"]
            res_tool.append(f"<tool_response>{tool_content}</tool_response>")
        tool_responses = "\n".join(res_tool)
        return f"<tool_responses>{tool_responses}</tool_responses>"

    def _get_tool_calls(self, tool_calls: List[str]):
        tool_calls = "\n".join(tool_calls)
        return f"<tool_calls>{tool_calls}</tool_calls>"

    def _format_tool_calls(self, tool_call_messages):
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            arguments = json.dumps(tool_call['arguments'], ensure_ascii=False)
            tool_calls.append(f'<tool_call>{tool_call["name"]}\n```json\n{arguments}\n```</tool_call>')
        return self._get_tool_calls(tool_calls)

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_descs = [json.dumps(self.wrap_tool(tool), ensure_ascii=False) for tool in tools]
        system = system or ""
        if system:
            system = f"{system}\n\n"
        return (
            f"""{system}# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
"""
            + "\n".join(tool_descs)
            + """
</tools>

For function call returns, you should first print <tool_calls>For each function call, you should return object like:
<tool_call>function_name
```json
function_arguments_in_json_format
```</tool_call>At the end of function call returns, you should print </tool_calls>"""
        )
