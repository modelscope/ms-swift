# Copyright (c) ModelScope Contributors. All rights reserved.
"""Message helpers shared by the AgentArk Env and scheduler."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Iterable, Mapping


def copy_messages(value: Any, *, field_name: str, require_nonempty: bool = False) -> list[dict[str, Any]]:
    """Validate and copy an OpenAI-style message list without touching media."""

    if not isinstance(value, list):
        raise ValueError(f'{field_name} must be a list of messages')
    if require_nonempty and not value:
        raise ValueError(f'{field_name} must not be empty')
    result: list[dict[str, Any]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise ValueError(f'{field_name}[{index}] must be an object')
        role = message.get('role')
        if role not in {'system', 'user', 'assistant', 'tool'}:
            raise ValueError(f'{field_name}[{index}].role is invalid: {role!r}')
        if 'content' not in message:
            raise ValueError(f'{field_name}[{index}] is missing content')
        result.append(deepcopy(dict(message)))
    return result


def content_to_text(content: Any) -> str:
    """Convert an assistant content value to the text sent to AgentArk."""

    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get('text', item.get('content', ''))
                if text is not None and str(text):
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return '\n'.join(parts)
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)


def latest_assistant_text(messages: Iterable[Mapping[str, Any]]) -> str:
    for message in reversed(list(messages)):
        if message.get('role') == 'assistant':
            return content_to_text(message.get('content'))
    raise ValueError('AgentArk step requires a generated assistant message')


def _extract_tag_content(text: str, tag: str) -> list[str]:
    pattern = rf'<{re.escape(tag)}\b[^>]*>(.*?)</{re.escape(tag)}>'
    return re.findall(pattern, text, re.IGNORECASE | re.DOTALL)


def extract_action(assistant_raw: str) -> str | None:
    """Match AgentArk's current API-agent action extraction semantics.

    The raw assistant response is preserved separately for conversation history;
    only this extracted value is submitted as executable environment action.
    """

    if not assistant_raw:
        return None

    tool_calls = _extract_tag_content(assistant_raw, 'tool_call')
    if tool_calls:
        if len(tool_calls) == 1:
            return f'<tool_call>{tool_calls[0].strip()}</tool_call>'
        joined = '\n'.join(block.strip() for block in tool_calls if block.strip())
        return f'<tool_call>{joined}</tool_call>' if joined else None

    params = _extract_tag_content(assistant_raw, 'params')
    if params:
        if len(params) == 1:
            return f'<params>{params[0].strip()}</params>'
        joined = '\n'.join(block.strip() for block in params if block.strip())
        return f'<params>{joined}</params>' if joined else None

    code = _extract_tag_content(assistant_raw, 'code')
    if code:
        if len(code) == 1:
            return code[0].strip('\n')
        return '\n\n'.join(block.strip('\n') for block in code if block.strip()) or None

    return assistant_raw


def new_environment_messages(
    conversation: list[dict[str, Any]],
    returned_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select the environment delta and remove AgentArk's assistant echo.

    AgentArk is normally configured for append-only delta messages and returns
    ``[assistant echo, user observation]``. The longest common-prefix removal
    also tolerates a server configured to return the full transcript.
    """

    messages = deepcopy(returned_messages)
    common = 0
    for existing, returned in zip(conversation, messages):  # noqa: B905
        if existing != returned:
            break
        common += 1
    if common:
        messages = messages[common:]
    return [message for message in messages if message.get('role') != 'assistant']
