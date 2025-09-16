# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import json

from .base import BaseAgentTemplate

if TYPE_CHECKING:
    from swift.llm.infer import Function
    from swift.llm.template import Prompt


class Llama3AgentTemplate(BaseAgentTemplate):
    eom_token = '<|eom_id|>'
    start_token = '<|start_header_id|>'
    end_token = '<|end_header_id|>'
    eot_token = '<|eot_id|>'

    def get_toolcall(self, response: str) -> List['Function']:
        from swift.llm.infer import Function
        if response.endswith(self.eom_token):
            response = response[:-len(self.eom_token)]
        functions = []
        res_list = re.findall(r'{[^{]*?"name":.*?"parameters":\s*?{.*?}\s*?}', response, re.DOTALL)
        for res in res_list:
            res = self._parse_json(res)
            if isinstance(res, dict) and 'name' in res and 'parameters' in res:
                functions.append(Function(name=res['name'], arguments=res['parameters']))
        if len(functions) == 0:
            # compat react_en
            return super().get_toolcall(response)
        return functions

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        res = [self.eot_token]
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res.append(f'{self.start_token}tool{self.end_token}\n\n{tool_content}{self.eot_token}')
        res.append(f'{self.start_token}assistant{self.end_token}\n\n')
        return assistant_content, res

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        assert user_message is not None
        user_content = user_message['content']
        tool_descs = [json.dumps(tool, ensure_ascii=False, indent=4) for tool in tools]
        new_user_content = """Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

""" + '\n\n'.join(tool_descs) + f"""

{user_content}"""  # noqa
        user_message['content'] = new_user_content
        return system or ''

    def _format_tool_calls(self, tool_call_messages) -> str:
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            tool_call['parameters'] = tool_call.pop('arguments')
            tool_calls.append(json.dumps(tool_call, ensure_ascii=False))
        return '\n'.join(tool_calls)


class Llama4AgentTemplate(Llama3AgentTemplate):
    eom_token = '<|eom|>'
    start_token = '<|header_start|>'
    end_token = '<|header_end|>'
    eot_token = '<|eot|>'
    toolcall_pattern = r'(.+?)<\|eom\|>'
