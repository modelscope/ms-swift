# Copyright (c) ModelScope Contributors. All rights reserved.
# ruff: noqa: E501
import re
from typing import List, Optional, Tuple, Union

import json

from ..utils import Prompt
from .base import BaseAgentTemplate
from .protocol import Function


class Llama3AgentTemplate(BaseAgentTemplate):
    tool_call_start = "{"
    eom_token = "<|eom_id|>"
    start_token = "<|start_header_id|>"
    end_token = "<|end_header_id|>"
    eot_token = "<|eot_id|>"

    @staticmethod
    def _tool_call_objects(response: str):
        decoder = json.JSONDecoder()
        for index, char in enumerate(response):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(response[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "name" in value and "parameters" in value:
                yield index, value

    def get_toolcall(self, response: str) -> List[Function]:
        functions = [
            Function(name=value["name"], arguments=value["parameters"])
            for _, value in self._tool_call_objects(response)
        ]
        if functions:
            return functions
        # Legacy tolerance: single-quoted dicts still parse via _parse_json.
        for raw in re.findall(r'{[^{]*?"name":.*?"parameters":\s*?{.*?}\s*?}', response, re.DOTALL):
            value = self._parse_json(raw)
            if isinstance(value, dict) and "name" in value and "parameters" in value:
                functions.append(Function(name=value["name"], arguments=value["parameters"]))
        return functions or super().get_toolcall(response)

    def find_tool_call_start(self, text: str, calls) -> Optional[int]:
        match = next(self._tool_call_objects(text), None)
        return match[0] if match else super().find_tool_call_start(text, calls)

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, "Prompt"]:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        # the engine owns the round separator (<|eot_id|>) and the response opener
        # (<|start_header_id|>assistant...): the tool block only carries the tool turns
        res = []
        for tool_message in tool_messages:
            tool_content = tool_message["content"]
            # the jinja renders every tool turn under the 'ipython' header and passes the content
            # through tojson (a str counts as iterable there, so it is quoted and escaped)
            if not isinstance(tool_content, (str, dict, list)):
                raise TypeError(f"unsupported tool content type: {type(tool_content).__name__}")
            rendered = json.dumps(tool_content, ensure_ascii=False)
            res.append(f"{self.start_token}ipython{self.end_token}\n\n{rendered}{self.eot_token}")
        return assistant_content, res

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        assert user_message is not None
        user_content = user_message["content"]
        tool_descs = [json.dumps(tool, ensure_ascii=False, indent=4) for tool in tools]
        new_user_content = (
            """Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

"""
            + "\n\n".join(tool_descs)
            + f"""

{user_content}"""
        )  # noqa
        user_message["content"] = new_user_content
        return system or ""

    def _format_tool_calls(self, tool_call_messages) -> str:
        tool_calls = []
        for message in tool_call_messages:
            # the jinja keeps the arguments form: a str renders through tojson as an escaped string
            tool_call = self._raw_tool_call(message["content"])
            tool_call["parameters"] = tool_call.pop("arguments")
            tool_calls.append(json.dumps(tool_call, ensure_ascii=False))
        return "\n".join(tool_calls)


class Llama4AgentTemplate(Llama3AgentTemplate):
    eom_token = "<|eom|>"
    start_token = "<|header_start|>"
    end_token = "<|header_end|>"
    eot_token = "<|eot|>"
    toolcall_pattern = r"(.+?)<\|eom\|>"

    def _add_tool_call_prefix(self, tool_content: str, pre_message=None) -> str:
        # the jinja wraps the assistant content of a tool-call turn in <|python_start|>...<|python_end|>
        # and puts the call JSON after it; dev renders the JSON as that turn's content, so the
        # (empty) content sits between the two markers
        return f"<|python_start|><|python_end|>{tool_content}"
