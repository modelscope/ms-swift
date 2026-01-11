# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Optional, Union

from .base import BaseAgentTemplate


class ReactGRPOAgentTemplate(BaseAgentTemplate):

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

        return """A conversation for tool calling between User and Assistant. The user asks a question which may be solved by calling tools, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process should be enclosed within <think> </think>tags and answer should follow the ReACT format(Action:xxx\nAction Input:xxx), i.e., <think> reasoning process here </think> Action: action here\nAction Input: parameters here

Answer the following questions as best as you can. You have access to the following tools:

""" + '\n\n'.join(tool_descs) + f"""

Use the following format:

<think>you should always think about what to do</think>
Action: the action to take, should be one of [{','.join(tool_names)}]
Action Input: the input to the action
Observation: the result of the action, given by the actual calling
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Final Answer: the final answer to the original input question

Begin!
"""  # noqa
