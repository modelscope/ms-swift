# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from typing import List, Optional, Union

from .qwen3_coder import Qwen3CoderAgentTemplate


def _to_json(value) -> str:
    # Mirrors the official jinja: insertion order, ", " / ": " separators, non-ASCII kept raw.
    return json.dumps(value, ensure_ascii=False)


class LFM3AgentTemplate(Qwen3CoderAgentTemplate):
    """LFM3 declares tools as JSON objects inside `<tools>...</tools>` and calls them with
    the `<function=...><parameter=...>` XML form, so only the declaration differs from
    Qwen3-Coder; parsing is inherited."""

    tool_call_separator = '\n\n'

    def _add_tool_call_prefix(self, tool_content: str, pre_message=None) -> str:
        # The official jinja concatenates the assistant content and the tool_call block
        # with no separator, unlike Qwen3-Coder's '\n\n'.
        return tool_content

    def _format_tool_responses(self, assistant_content: str, tool_messages):
        # Official LFM3 renders tool results as a `tool` turn, not as a user turn.
        chat_sep = (self.template_meta.chat_sep or []) if hasattr(self, 'template_meta') else ['<|im_end|>\n']
        # The tool turn is the query of the next round, so close it with the assistant header.
        return assistant_content, chat_sep + [
            '<|im_start|>tool\n',
            self._get_tool_responses(tool_messages),
            '<|im_end|>\n',
            '<|im_start|>assistant\n',
        ]

    def _format_standalone_tool_responses(self, tool_messages) -> 'Prompt':
        # Official role for tool results is `tool`:
        # `<|im_start|>tool\n<tool_response>...</tool_response><|im_end|>\n`.
        return [
            '<|im_start|>tool\n',
            self._get_tool_responses(tool_messages),
            '<|im_end|>\n',
        ]

    def _get_tool_responses(self, tool_messages):
        # Official jinja: `<tool_response>\n{content}\n</tool_response>`, no trailing newline.
        return '\n'.join(f'<tool_response>\n{m["content"]}\n</tool_response>' for m in tool_messages)

    def _format_tool_user_followup(self, tool_messages, user_messages) -> 'Prompt':
        # Tool results are a `tool` turn; the following user turns keep the user role.
        res = list(self.template_meta.chat_sep or [])
        res += ['<|im_start|>tool\n', self._get_tool_responses(tool_messages), '<|im_end|>\n']
        for message in user_messages:
            res += ['<|im_start|>user\n', message['content'], '<|im_end|>\n']
        res.append('<|im_start|>assistant\n')
        return res

    def _format_tools(self,
                      tools: List[Union[str, dict]],
                      system: Optional[str] = None,
                      user_message: Optional[dict] = None) -> str:
        lines = []
        for tool in tools:
            if isinstance(tool, str):
                lines.append(tool)
                continue
            fn = tool.get('function', tool)
            if isinstance(fn, str):
                lines.append(fn)
                continue
            fields = [f'{_to_json(k)}: {_to_json(v)}' for k, v in fn.items() if k != 'defer_loading']
            lines.append('{' + ', '.join(fields) + '}')
        block = '<tools>\n' + '\n'.join(lines) + '\n</tools>'
        return f'{system}\n\n{block}' if system else block
