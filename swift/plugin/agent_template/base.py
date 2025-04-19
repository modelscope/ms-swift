# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from swift.llm.template import split_str_parts_by
from swift.llm.utils import Messages


@dataclass
class Function:
    name: str
    arguments: Optional[str]


@dataclass
class AgentKeyword:
    action: str = 'Action:'
    action_input: str = 'Action Input:'
    observation: str = 'Observation:'


def split_action_action_input(response: str,
                              keyword: Optional[AgentKeyword] = None) -> Tuple[Optional[str], Optional[str]]:
    keyword = keyword or AgentKeyword()
    agent_keyword = [
        'action:', 'Action:', 'ACTION:', 'action input:', 'Action Input:', 'Action input:', 'ACTION INPUT:', 'Thought:',
        'Final Answer:', 'Observation:'
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


class ReactCompatMixin:
    keyword = AgentKeyword()

    def get_toolcall(self, response: str) -> List[Function]:
        action, action_input = split_action_action_input(response, keyword=self.keyword)
        if action is None:
            return []

        return [Function(name=action, arguments=action_input)]


class BaseAgentTemplate(ReactCompatMixin, ABC):

    def format_system(self, tools, system: Optional[str] = None):
        system = system or ''
        if isinstance(tools, str):
            tools = json.loads(tools)
        tool_names = []
        for tool in tools:  # info: Dict[str, Union[str, dict]]
            if isinstance(tool, dict) and 'function' in tool and 'name' in tool['function']:
                tool_names.append(tool['function']['name'])
            else:
                tool_names.append(tool['name'])
        return self._format_system(tool_names, tools, system)

    def format_observations(self, messages: Messages):
        if len(messages) < 2:
            return
        i = 1
        from swift.plugin import get_tools_keyword
        keyword = get_tools_keyword(tools_prompt)
        while i < len(messages):
            pre_message, message = messages[i - 1], messages[i]
            pre_role, pre_content = pre_message['role'], pre_message['content']
            role, content = message['role'], message['content']
            if (pre_role == 'assistant' and role == 'tool' and isinstance(pre_content, str)
                    and pre_content.endswith(keyword.observation)):
                assert isinstance(pre_content, str)
                pre_message['content'] = pre_content + content + '\n'  # assistant
                messages.pop(i)  # remove tool
            elif (pre_role == 'assistant' and role == 'assistant' and isinstance(pre_content, str)
                  and isinstance(content, str)):
                # Consecutive messages from the assistant role need to be merged to prevent errors.
                pre_message['content'] = pre_content + content
                messages.pop(i)
            else:
                i += 1

    @abstractmethod
    def _format_system(self, tool_names: List[str], tools: List[Union[str, Dict[str, Any]]], system: str) -> str:
        pass

    @abstractmethod
    def _format_observations(self, observations: List[str]) -> str:
        pass
