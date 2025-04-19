# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Optional, Union

import json

from .base import BaseAgentTemplate


class QwenEnAgentTemplate(BaseAgentTemplate):

    def _format_system(self, tool_names: List[str], tools: List[Union[str, dict]], system: str) -> str:
        tools = [t if isinstance(t, str) else json.dumps(t, ensure_ascii=False) for t in tools]

        tool_descs = []
        for tool_name, tool in zip(tool_names, tools):
            tool_descs.append(f'### {tool_name}\n\n{tool}\n\n')
        return f"""{system}

# Tools

## You have access to the following tools:

""" + ''.join(tool_descs) + f"""## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:

✿FUNCTION✿: The tool to use, should be one of [{','.join(tool_names)}]
✿ARGS✿: The input of the tool
✿RESULT✿: Tool results
✿RETURN✿: Reply based on tool results. Images need to be rendered as ![](url)"""  # noqa


class QwenZhAgentTemplate(BaseAgentTemplate):

    def _format_system(self, tool_names: List[str], tools: List[Union[str, dict]], system: str) -> str:
        tools = [t if isinstance(t, str) else json.dumps(t, ensure_ascii=False) for t in tools]

        tool_descs = []
        for tool_name, tool in zip(tool_names, tools):
            tool_descs.append(f'### {tool_name}\n\n{tool}\n\n')
        return f"""{system}

# 工具

## 你拥有如下工具：

""" + ''.join(tool_descs) + f"""## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

✿FUNCTION✿: 工具名称，必须是[{','.join(tool_names)}]之一。
✿ARGS✿: 工具输入
✿RESULT✿: 工具结果
✿RETURN✿: 根据工具结果进行回复，需将图片用![](url)渲染出来"""  # noqa
