# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional, Union

import json

from .base import BaseAgentTemplate


class GLM4AgentTemplate(BaseAgentTemplate):
    is_glm4_0414 = False

    def _format_system(self, tools: List[Union[str, dict]], system: str) -> str:
        tool_descs = []
        for tool in tools:
            name = self._get_tool_name(tool)
            tool_descs.append(f'## {name}\n\n{json.dumps(tool, ensure_ascii=False, indent=4)}\n'
                              '在调用上述函数时，请使用 Json 格式表示调用的参数。')
        glm4_system = '你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n'  # noqa
        return ('' if self.is_glm4_0414 else glm4_system) + """# 可用工具

""" + '\n'.join(tool_descs)


class GLM4_0414AgentTemplate(GLM4AgentTemplate):
    is_glm4_0414 = True
