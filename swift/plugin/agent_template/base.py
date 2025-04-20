# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import json


@dataclass
class AgentKeyword:
    action: str = 'Action:'
    action_input: str = 'Action Input:'
    observation: str = 'Observation:'


@dataclass
class ToolDesc:
    name_for_model: str
    name_for_human: str
    description_for_model: str
    parameters: str
    args_format: str


class ReactCompatMixin:
    keyword = AgentKeyword()

    def _split_action_action_input(self, response: str, keyword: AgentKeyword) -> List['Function']:
        from swift.llm.template import split_str_parts_by
        from swift.llm.infer.protocol import Function
        agent_parts = split_str_parts_by(response, list(asdict(keyword).values()))
        functions = []
        action_content = None

        for part in agent_parts:
            key, content = part['key'].lower(), part['content']
            if action_content is None and key == keyword.action.lower():
                action_content = content
            elif action_content is not None and key == keyword.action_input.lower():
                functions.append(Function(name=action_content, arguments=content))
                action_content = None

        return functions

    def get_toolcall(self, response: str) -> List['Function']:
        from swift.llm.infer.protocol import Function
        functions = self._split_action_action_input(response, self.keyword)
        if len(functions) == 0 and self.keyword != ReactCompatMixin.keyword:
            # compat react
            functions = self._split_action_action_input(response, ReactCompatMixin.keyword)
        return functions

    def _format_tool_messages(
        self,
        assistant_content: str,
        tool_messages: List[str],
    ) -> str:
        assert len(tool_messages) > 0
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            if not assistant_content.endswith(self.keyword.observation):
                if not assistant_content.endswith('\n'):
                    assistant_content += '\n'
                assistant_content += self.keyword.observation
            res = []
            for i, tool_message in enumerate(tool_messages):
                if i > 0:
                    res.append(self.keyword.observation)
                tool_content = tool_message['content']
                res.append(tool_content)
                if not tool_content.endswith('\n'):
                    res.append('\n')
        else:
            res = []
            for tool_message in tool_messages:
                res.append(tool_message['content'])
        return assistant_content, ''.join(res)


class BaseAgentTemplate(ReactCompatMixin, ABC):

    @staticmethod
    def _get_tool_name(tool):
        return tool.get('name_for_model') or tool.get('name')

    @staticmethod
    def _parse_tool(tool, lang: Literal['zh', 'en']) -> ToolDesc:
        name_for_model = BaseAgentTemplate._get_tool_name(tool)
        name_for_human = tool.get('name_for_human') or name_for_model

        description = tool.get('description') or tool.get('description_for_model')
        parameters = tool.get('parameters') or {}
        parameters = parameters if isinstance(parameters, str) else json.dumps(parameters, ensure_ascii=False)
        args_format = '此工具的输入应为JSON对象。' if lang == 'zh' else 'Format the arguments as a JSON object.'
        tool_desc = ToolDesc(
            name_for_model=name_for_model,
            name_for_human=name_for_human,
            description_for_model=description,
            parameters=parameters,
            args_format=args_format)
        assert name_for_model is not None and description is not None, f'tool_desc: {tool_desc}'
        return tool_desc

    def format_tools(self, tools, system: Optional[str] = None, user_message=None):
        # user_message: first user message
        system = system or ''
        if isinstance(tools, str):
            tools = json.loads(tools)
        new_tools = []
        for tool in tools:  # info: Dict[str, Union[str, dict]]
            if isinstance(tool, dict) and 'function' in tool:
                tool = tool['function']
            new_tools.append(tool)
        return self._format_tools(new_tools, system, user_message)

    @staticmethod
    def _parse_json(json_str: str) -> Optional[Any]:
        try:
            res = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                res = ast.literal_eval(json_str)
            except Exception:
                return
        return res

    def format_messages(self, messages: List[Dict[str, str]]) -> None:
        if len(messages) < 2:
            return
        i = 1
        while i < len(messages):
            pre_message, message = messages[i - 1], messages[i]
            pre_role, pre_content = pre_message['role'], pre_message['content']
            role, content = message['role'], message['content']
            if pre_role == 'assistant' and role == 'tool':
                i_start = i
                while i + 1 < len(messages) and messages[i + 1]['role'] == 'tool':
                    i += 1
                pre_message['content'], tool_content = self._format_tool_messages(pre_content, messages[i_start:i + 1])
                messages[i_start:i + 1] = [{'role': 'tool', 'content': tool_content}]
                i = i_start + 1
            elif pre_role == 'assistant' and role == 'assistant':
                # Consecutive messages from the assistant role need to be merged to prevent errors.
                pre_message['content'] = pre_content + content
                messages.pop(i)
            else:
                i += 1

    @abstractmethod
    def _format_tools(self, tools: List[Union[str, dict]], system: str, user_message=None) -> str:
        pass
