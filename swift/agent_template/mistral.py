# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.template import Prompt
from .base import BaseAgentTemplate


class MistralAgentTemplate(BaseAgentTemplate):

    def get_toolcall(self, response: str) -> List[Function]:
        res_list = re.findall(r'\[TOOL_CALLS\]\[(.*?)\]</s>', response, re.DOTALL)
        if not res_list:
            return []
        res_list = res_list[0].strip().split('\n')
        functions = []
        for res_str in res_list:
            parsed_res = self._parse_json(res_str)
            if isinstance(parsed_res, dict):
                parsed_res = [parsed_res]  # Handle single tool call
            if isinstance(parsed_res, list):
                for tool_call in parsed_res:
                    if isinstance(tool_call, dict) and 'name' in tool_call and 'arguments' in tool_call:
                        functions.append(Function(name=tool_call['name'], arguments=tool_call['arguments']))
        if len(functions) == 0:
            # compat react_en
            return super().get_toolcall(response)
        return functions

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        if not hasattr(self, 'template_meta'):
            raise ValueError('MistralAgentTemplate requires template_meta to be registered')
        prompt = self.template_meta.prompt
        chat_sep = self.template_meta.chat_sep

        res = chat_sep.copy()
        res_tool = []
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            # append `[TOOL_RESULTS]{"content": {{ .Content }}}[/TOOL_RESULTS]` to res_tool
            res_tool.append(f'[TOOL_RESULTS]{json.dumps({"content": tool_content}, ensure_ascii=False)}[/TOOL_RESULTS]')
        total_tool = '\n'.join(res_tool)
        for context in prompt:
            if isinstance(context, str):
                context = context.replace('{{QUERY}}', total_tool)
            res.append(context)
        return assistant_content, res

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_descs = [json.dumps(self.wrap_tool(tool), ensure_ascii=False) for tool in tools]
        system = system or ''
        return f"""{system}[AVAILABLE_TOOLS]{' '.join(tool_descs)}[/AVAILABLE_TOOLS]"""

    def _format_tool_calls(self, tool_call_messages):
        tool_calls = []
        for message in tool_call_messages:
            # needs `{'name': name, 'arguments': arguments}`, which self._parse_tool_call
            # satisfies
            tool_call = self._parse_tool_call(message['content'])
            tool_calls.append(json.dumps(tool_call, ensure_ascii=False))
        return f'[TOOL_CALLS][\n{chr(10).join(tool_calls)}\n]</s>'  # check if need `</s>` at end


class Mistral2512AgentTemplate(MistralAgentTemplate):
    """ref: https://modelscope.cn/models/mistralai/Ministral-3-8B-Instruct-2512 (chat_template.jinja)

    Ministral-3 2512 declares the tools as one JSON list after the system block, renders every call as
    `[TOOL_CALLS]name[ARGS]arguments` and every tool result as a bare `[TOOL_RESULTS]content[/TOOL_RESULTS]`.
    """

    def get_toolcall(self, response: str) -> List[Function]:
        res_list = re.findall(r'\[TOOL_CALLS\](.+?)\[ARGS\](.*?)(?=\[TOOL_CALLS\]|</s>|$)', response, re.DOTALL)
        if not res_list:
            # compat mistral
            return super().get_toolcall(response)
        return [Function(name=name, arguments=arguments) for name, arguments in res_list]

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        # chat_sep ends the tool-call turn; the results are not wrapped in a user turn.
        res = self.template_meta.chat_sep.copy()
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res.append(f'[TOOL_RESULTS]{tool_content}[/TOOL_RESULTS]')
        return assistant_content, res

    def _format_tool_user_followup(self, tool_messages, user_messages) -> 'Prompt':
        # The next user turn directly follows the results; the jinja joins consecutive user messages.
        _, tool_content = self._format_tool_responses('', tool_messages)
        prefix, suffix = ''.join(self.template_meta.prompt).split('{{QUERY}}', 1)
        return tool_content + [prefix + '\n\n'.join(message['content'] for message in user_messages) + suffix]

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tools = json.dumps([self.wrap_tool(tool) for tool in tools], ensure_ascii=False)
        system = system or ''
        # The system prefix is `[SYSTEM_PROMPT]{{SYSTEM}}[/SYSTEM_PROMPT]`: close the system block here,
        # because the tools are declared after it (see Mistral2512Template._swift_encode).
        return f'{system}[/SYSTEM_PROMPT][AVAILABLE_TOOLS]{tools}[/AVAILABLE_TOOLS]'

    def _format_tool_calls(self, tool_call_messages):
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            name = tool_call['name']
            arguments = json.dumps(tool_call['arguments'], ensure_ascii=False)
            tool_calls.append(f'[TOOL_CALLS]{name}[ARGS]{arguments}')
        # No trailing `</s>`: chat_sep / suffix end the turn.
        return ''.join(tool_calls)
