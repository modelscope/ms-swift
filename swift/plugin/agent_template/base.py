# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import json


@dataclass
class Function:
    name: str
    arguments: Optional[str]


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

    def _split_action_action_input(response: str) -> Tuple[Optional[str], Optional[str]]:
        from swift.llm.template import split_str_parts_by
        keyword = self.keyword
        agent_keyword = [
            'action:', 'Action:', 'ACTION:', 'action input:', 'Action Input:', 'Action input:', 'ACTION INPUT:',
            'Thought:', 'Final Answer:', 'Observation:'
        ]
        for key in asdict(keyword).values():
            if key not in agent_keyword:
                agent_keyword.append(key)
        agent_parts = split_str_parts_by(response, agent_keyword)
        action = None
        action_input = None
        for c in agent_parts:
            if c['key'].lower() == keyword.action.lower():
                action = c['content']
            elif c['key'].lower() == keyword.action_input.lower():
                action_input = c['content']
        if action:
            action = action.strip().replace('\n', '')
        if action_input:
            action_input.strip().replace('\n', '')
        return action, action_input

    def get_toolcall(self, response: str) -> List[Function]:
        action, action_input = split_action_action_input(response, keyword=self.keyword)
        if action is None:
            return []

        return [Function(name=action, arguments=action_input)]

    def _format_tool_messages(
        self,
        assistant_content: str,
        tool_messages: List[str],
    ) -> str:
        res = [assistant_content]
        with_observation = assistant_content.endswith(self.keyword.observation)
        for tool_message in tool_messages:
            res.append(tool_message['content'])
            if with_observation:
                res.append('\n')
        return ''.join(res)


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

    def format_system(self, tools, system: Optional[str] = None):
        system = system or ''
        if isinstance(tools, str):
            tools = json.loads(tools)
        new_tools = []
        for tool in tools:  # info: Dict[str, Union[str, dict]]
            if isinstance(tool, dict) and 'function' in tool:
                tool = tool['function']
            new_tools.append(tool)
        return self._format_system(new_tools, system)

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
                pre_message['content'] = self._format_tool_messages(pre_content, messages[i_start:i + 1])
                messages[i_start:i + 1] = []
            elif pre_role == 'assistant' and role == 'assistant':
                # Consecutive messages from the assistant role need to be merged to prevent errors.
                pre_message['content'] = pre_content + content
                messages.pop(i)
            else:
                i += 1

    @abstractmethod
    def _format_system(self, tools: List[Union[str, dict]], system: str) -> str:
        pass
