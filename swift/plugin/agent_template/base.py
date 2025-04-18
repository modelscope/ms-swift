# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from swift.llm.template import split_str_parts_by


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

    @abstractmethod
    def format_system(self, tool_names: List[str], tools: List[Union[str, Dict[str, Any]]],
                      system: Optional[str]) -> str:
        pass

    @abstractmethod
    def format_observations(self, observations: List[str]) -> str:
        pass
