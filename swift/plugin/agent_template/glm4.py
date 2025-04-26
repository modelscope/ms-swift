# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import json

from .base import BaseAgentTemplate

if TYPE_CHECKING:
    from swift.llm.infer import Function
    from swift.llm.template import Prompt


class GLM4AgentTemplate(BaseAgentTemplate):
    is_glm4_0414 = False

    @staticmethod
    def _find_function_call(single_content: str) -> Optional['Function']:
        from swift.llm.infer import Function
        single_content = single_content.replace('<|observation|>', '')
        pattern = re.compile(r'([^\n`]*?)\n({.*?})(?=\w*\n|$)', re.DOTALL)
        matches = pattern.findall(single_content)
        if not matches:
            return

        name, arguments = matches[0]
        return Function(name=name, arguments=arguments)

    def get_toolcall(self, response: str) -> List['Function']:
        toolcall_list = response.split('<|assistant|>')
        functions = []
        for toolcall in toolcall_list:
            function = self._find_function_call(toolcall)
            if function:
                functions.append(function)
        if len(functions) == 0:
            # compat react_en
            return super().get_toolcall(response)
        return functions

    def _format_tools(self, tools: List[Union[str, dict]], system: str, user_message=None) -> str:
        tool_descs = []
        for tool in tools:
            tool = self.unwrap_tool(tool)
            name = self._get_tool_name(tool)
            tool_descs.append(f'## {name}\n\n{json.dumps(tool, ensure_ascii=False, indent=4)}\n'
                              '在调用上述函数时，请使用 Json 格式表示调用的参数。')
        glm4_system = '你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n'  # noqa
        return ('' if self.is_glm4_0414 else glm4_system) + """# 可用工具

""" + '\n'.join(tool_descs)

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        res = ['\n']
        for i, tool_message in enumerate(tool_messages):
            tool_content = tool_message['content']
            if i > 0:
                res.append('<|observation|>\n')
            res.append(tool_content)
        res.append('<|assistant|>\n')
        return assistant_content, res

    def _format_tool_calls(self, tool_call_messages) -> str:
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            tool_calls.append(f'{tool_call["name"]}\n{tool_call["arguments"]}')
        return '<|assistant|>'.join(tool_calls) + '<|observation|>'


class GLM4_0414AgentTemplate(GLM4AgentTemplate):
    is_glm4_0414 = True
