# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Optional, Union

from .base import AgentKeyword, BaseAgentTemplate

keyword = AgentKeyword(
    action='✿FUNCTION✿:',
    action_input='✿ARGS✿:',
    observation='✿RESULT✿:',
)


class QwenEnAgentTemplate(BaseAgentTemplate):
    keyword = keyword

    def _get_tool_names_descs(self, tools):
        tool_names = []
        tool_descs = []
        for tool in tools:
            tool_desc = self._parse_tool(tool, 'en')
            tool_names.append(tool_desc.name_for_model)
            tool_descs.append(f'### {tool_desc.name_for_human}\n\n'
                              f'{tool_desc.name_for_model}: {tool_desc.description_for_model} '
                              f'Parameters: {tool_desc.parameters} {tool_desc.args_format}')
        return tool_names, tool_descs

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_names, tool_descs = self._get_tool_names_descs(tools)
        system = system or ''
        return f"""{system}

# Tools

## You have access to the following tools:

""" + '\n\n'.join(tool_descs) + f"""

## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:

✿FUNCTION✿: The tool to use, should be one of [{','.join(tool_names)}]
✿ARGS✿: The input of the tool
✿RESULT✿: Tool results
✿RETURN✿: Reply based on tool results. Images need to be rendered as ![](url)"""  # noqa


class QwenZhAgentTemplate(BaseAgentTemplate):
    keyword = keyword

    def _get_tool_names_descs(self, tools):
        tool_names = []
        tool_descs = []
        for tool in tools:
            tool_desc = self._parse_tool(tool, 'zh')
            tool_names.append(tool_desc.name_for_model)
            tool_descs.append(f'### {tool_desc.name_for_human}\n\n'
                              f'{tool_desc.name_for_model}: {tool_desc.description_for_model} '
                              f'输入参数：{tool_desc.parameters} {tool_desc.args_format}')
        return tool_names, tool_descs

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_names, tool_descs = self._get_tool_names_descs(tools)
        system = system or ''
        return f"""{system}

# 工具

## 你拥有如下工具：

""" + '\n\n'.join(tool_descs) + f"""

## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

✿FUNCTION✿: 工具名称，必须是[{','.join(tool_names)}]之一。
✿ARGS✿: 工具输入
✿RESULT✿: 工具结果
✿RETURN✿: 根据工具结果进行回复，需将图片用![](url)渲染出来"""  # noqa


class QwenEnParallelAgentTemplate(QwenEnAgentTemplate):

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_names, tool_descs = self._get_tool_names_descs(tools)
        system = system or ''
        return f"""{system}

# Tools

## You have access to the following tools:

""" + '\n\n'.join(tool_descs) + f"""

## Insert the following command in your reply when you need to call N tools in parallel:

✿FUNCTION✿: The name of tool 1, should be one of [{','.join(tool_names)}]
✿ARGS✿: The input of tool 1
✿FUNCTION✿: The name of tool 2
✿ARGS✿: The input of tool 2
...
✿FUNCTION✿: The name of tool N
✿ARGS✿: The input of tool N
✿RESULT✿: The result of tool 1
✿RESULT✿: The result of tool 2
...
✿RESULT✿: he result of tool N
✿RETURN✿: Reply based on tool results. Images need to be rendered as ![](url)"""  # noqa


class QwenZhParallelAgentTemplate(QwenZhAgentTemplate):

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_names, tool_descs = self._get_tool_names_descs(tools)
        system = system or ''
        return f"""{system}

# 工具

## 你拥有如下工具：

""" + '\n\n'.join(tool_descs) + f"""

## 你可以在回复中插入以下命令以并行调用N个工具：

✿FUNCTION✿: 工具1的名称，必须是[{','.join(tool_names)}]之一
✿ARGS✿: 工具1的输入
✿FUNCTION✿: 工具2的名称
✿ARGS✿: 工具2的输入
...
✿FUNCTION✿: 工具N的名称
✿ARGS✿: 工具N的输入
✿RESULT✿: 工具1的结果
✿RESULT✿: 工具2的结果
...
✿RESULT✿: 工具N的结果
✿RETURN✿: 根据工具结果进行回复，需将图片用![](url)渲染出来"""  # noqa
