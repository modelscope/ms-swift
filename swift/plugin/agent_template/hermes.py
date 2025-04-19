# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional, Union

import json

from .base import BaseAgentTemplate


class HermesAgentTemplate(BaseAgentTemplate):

    def _format_system(self, tools: List[Union[str, dict]], system: str) -> str:
        tool_descs = [json.dumps({'type': 'function', 'function': tool}, ensure_ascii=False) for tool in tools]
        return f"""{system}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + '\n'.join(tool_descs) + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""
