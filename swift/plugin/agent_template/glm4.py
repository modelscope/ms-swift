# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional, Union

import json

from .base import BaseAgentTemplate


class GLM4AgentTemplate(BaseAgentTemplate):

    def _format_system(self, tools: List[Union[str, dict]], system: str) -> str:
        tool_descs = []
        for tool in tools:
            name = self._get_tool_name(tool)
            tool_descs.append(f'## {name}\n\n{json.dumps(tool, ensure_ascii=False, indent=4)}\n'
                              '在调用上述函数时，请使用 Json 格式表示调用的参数。')
        return """# 可用工具

""" + '\n'.join(tool_descs)
