# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Optional, Union

from .base import BaseAgentTemplate


class ReactEnAgentTemplate(BaseAgentTemplate):

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_names = []
        tool_descs = []
        for tool in tools:
            tool_desc = self._parse_tool(tool, 'en')
            tool_names.append(tool_desc.name_for_model)
            tool_descs.append(
                f'{tool_desc.name_for_model}: Call this tool to interact with the {tool_desc.name_for_human} API. '
                f'What is the {tool_desc.name_for_human} API useful for? {tool_desc.description_for_model} '
                f'Parameters: {tool_desc.parameters} {tool_desc.args_format}')

        return """Answer the following questions as best you can. You have access to the following tools:

""" + '\n\n'.join(tool_descs) + f"""

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{','.join(tool_names)}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""


class ReactZnAgentTemplate(BaseAgentTemplate):

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_names = []
        tool_descs = []
        for tool in tools:
            tool_desc = self._parse_tool(tool, 'zh')
            tool_names.append(tool_desc.name_for_model)
            tool_descs.append(f'{tool_desc.name_for_model}: 调用此工具与 {tool_desc.name_for_human} API 进行交互。'
                              f'{tool_desc.name_for_human} 有什么用？{tool_desc.description_for_model} '
                              f'输入参数：{tool_desc.parameters} {tool_desc.args_format}')
        return """尽可能地回答以下问题。你可以使用以下工具:

""" + '\n\n'.join(tool_descs) + f"""

请按照以下格式进行:

Question: 需要你回答的输入问题
Thought: 你应该总是思考该做什么
Action: 需要使用的工具，应该是[{','.join(tool_names)}]中的一个
Action Input: 传入工具的内容
Observation: 行动的结果
... (这个Thought/Action/Action Input/Observation可以重复N次)
Thought: 我现在知道最后的答案
Final Answer: 对原始输入问题的最终答案

现在开始！
"""
