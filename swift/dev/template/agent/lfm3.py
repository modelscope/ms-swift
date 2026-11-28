# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from typing import List, Optional, Union

from .qwen3_coder import Qwen3CoderAgentTemplate


class LFM3AgentTemplate(Qwen3CoderAgentTemplate):
    """LFM3 JSON tool declarations and Qwen3-Coder XML calls."""

    tool_call_separator = "\n\n"

    def _add_tool_call_prefix(self, tool_content: str, pre_message=None) -> str:
        return tool_content

    def _format_tool_responses(self, assistant_content: str, tool_messages):
        return assistant_content, [
            "<|im_start|>tool\n",
            self._get_tool_responses(tool_messages),
            "<|im_end|>\n",
        ]

    def _format_standalone_tool_responses(self, tool_messages):
        return ["<|im_start|>tool\n", self._get_tool_responses(tool_messages), "<|im_end|>\n"]

    def _get_tool_responses(self, tool_messages):
        return "\n".join(f'<tool_response>\n{message["content"]}\n</tool_response>' for message in tool_messages)

    def _format_tool_user_followup(self, tool_messages, user_messages):
        result = ["<|im_start|>tool\n", self._get_tool_responses(tool_messages), "<|im_end|>\n"]
        for message in user_messages:
            result += ["<|im_start|>user\n", message["content"], "<|im_end|>\n"]
        return result

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        lines = []
        for tool in tools:
            if isinstance(tool, str):
                lines.append(tool)
                continue
            function = tool.get("function", tool)
            if isinstance(function, str):
                lines.append(function)
                continue
            fields = [
                f"{json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}"
                for key, value in function.items() if key != "defer_loading"
            ]
            lines.append("{" + ", ".join(fields) + "}")
        block = "<tools>\n" + "\n".join(lines) + "\n</tools>"
        return f"{system}\n\n{block}" if system else block
