# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import json

if TYPE_CHECKING:
    from swift.llm.infer import Function
    from swift.llm.template import Prompt


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

    @staticmethod
    def _split_action_action_input(response: str, keyword: AgentKeyword) -> List['Function']:
        from swift.llm.template import split_str_parts_by
        from swift.llm.infer import Function
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
        functions = self._split_action_action_input(response, self.keyword)
        if len(functions) == 0 and self.keyword != ReactCompatMixin.keyword:
            # compat react
            functions = self._split_action_action_input(response, ReactCompatMixin.keyword)
        return functions

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
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
        return assistant_content, res

    @staticmethod
    def _parse_tool_call(content) -> Dict[str, Any]:
        obj = BaseAgentTemplate._parse_json(content)
        name = obj['name']
        arguments = obj.get('arguments')
        if arguments is None:
            arguments = obj.get('parameters')
        arguments = BaseAgentTemplate._parse_json(arguments)
        assert arguments is not None, f'content: {content}'
        return {'name': name, 'arguments': arguments}

    def _format_tool_calls(self, tool_call_messages) -> str:
        # -> assistant_content
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            tool_calls.append(f'{self.keyword.action} {tool_call["name"]}\n'
                              f'{self.keyword.action_input} {tool_call["arguments"]}\n')
        tool_calls.append(self.keyword.observation)
        return ''.join(tool_calls)


class BaseAgentTemplate(ReactCompatMixin, ABC):

    @staticmethod
    def _get_tool_name(tool):
        return tool.get('name_for_model') or tool.get('name')

    @staticmethod
    def unwrap_tool(tool):
        assert isinstance(tool, dict), f'tool: {tool}'
        if 'type' in tool and 'function' in tool:
            tool = tool['function']
        return tool

    @staticmethod
    def wrap_tool(tool):
        assert isinstance(tool, dict), f'tool: {tool}'
        if 'type' not in tool and 'function' not in tool:
            tool = {'type': 'function', 'function': tool}
        return tool

    @staticmethod
    def _parse_tool(tool, lang: Literal['zh', 'en']) -> ToolDesc:
        tool = BaseAgentTemplate.unwrap_tool(tool)
        name_for_model = BaseAgentTemplate._get_tool_name(tool)
        name_for_human = tool.get('name_for_human') or name_for_model

        description = tool.get('description')
        if description is None:
            description = tool.get('description_for_model')
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

    @staticmethod
    def _parse_json(json_str: str) -> Optional[Any]:
        if not isinstance(json_str, str):
            return json_str
        try:
            res = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                res = ast.literal_eval(json_str)
            except Exception:
                return
        return res

    @abstractmethod
    def _format_tools(self, tools: List[Union[str, dict]], system: str, user_message=None) -> str:
        pass
