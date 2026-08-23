# Copyright (c) ModelScope Contributors. All rights reserved.
"""The Anthropic messages dialect: a turn whose ``content`` is a list of typed blocks.

Anthropic's API does not put a turn's text directly in ``content``; it wraps everything in typed
blocks::

    {'role': 'user', 'content': [
        {'type': 'text', 'text': 'what is this'},
        {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': '...'}},
    ]}
    {'role': 'assistant', 'content': [
        {'type': 'tool_use', 'name': 'search', 'input': {'q': 1}},
    ]}
    {'role': 'user', 'content': [
        {'type': 'tool_result', 'content': 'result'},
    ]}

So this is a *dialect of the messages format*, not a top-level format of its own: it shares the same
``messages`` column and the same role handling, and differs only in how one turn's content is encoded.
That is why the parsing lives here as static methods that :class:`OpenAIConverter`'s auto path calls,
while the class itself is a thin subclass that pins ``message_format='anthropic'`` for a dataset known
to be Anthropic -- mirroring legacy main's ``AnthropicMessagesPreprocessor``.

What the parser does with each block type:

- ``text``     -> appended to the turn's text.
- ``image``    -> emits an ``<image>`` placeholder in the text and lifts the source (a base64 data
                  URI or a plain URL) into ``media['images']``.
- ``tool_use`` -> flushed into its own ``tool_call`` message ``{'name', 'arguments'}``.
- ``tool_result`` -> its own ``tool_response`` message.

This is emphatically **not** the Anthropic HH-RLHF dataset (``chosen``/``rejected`` transcript
strings): that is one dataset's quirk and belongs in a dataset-specific preprocessor, not in the
format layer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Messages, register_format
from .openai import OpenAIConverter

__all__ = ['AnthropicConverter']


@register_format
class AnthropicConverter(OpenAIConverter):
    """Pin the Anthropic dialect. Auto-detection is handled by :class:`OpenAIConverter`."""

    format_name = 'anthropic'
    # Never auto-selected by column name -- it shares the `messages` column with the openai dialect,
    # so the two are only separable per row (done inside OpenAIConverter). This class exists to be
    # pinned explicitly (`format_name='anthropic'`) and to hold the block parser below.
    priority = 12
    is_fallback = False

    def __init__(self, **kwargs):
        kwargs['message_format'] = 'anthropic'
        super().__init__(**kwargs)

    @classmethod
    def detect(cls, columns) -> bool:
        return False

    @staticmethod
    def anthropic_to_messages(messages: List[Dict[str, Any]], media: Optional[Dict[str, List]] = None) -> Messages:
        """Expand Anthropic content blocks into plain role/content messages.

        A turn whose ``content`` is not a block list is already plain and passes through. Within a
        block list, consecutive text/image blocks accumulate into one message; a tool_use or
        tool_result block first flushes that accumulated text, then emits its own message -- so the
        relative order of text and tool calls within a turn is preserved.
        """
        media = media if media is not None else {'images': []}
        new_messages: Messages = []
        for message in messages:
            content = message.get('content')
            if not isinstance(content, list):
                new_messages.append(message)
                continue

            metadata = {key: message[key] for key in ('loss', 'loss_scale') if key in message}
            pending: List[str] = []

            def flush() -> None:
                if pending:
                    new_messages.append({'role': message['role'], 'content': ''.join(pending), **metadata})
                    pending.clear()

            for block in content:
                block_type = block.get('type')
                if block_type == 'text':
                    pending.append(block.get('text', ''))
                elif block_type == 'image':
                    pending.append('<image>')
                    media['images'].append(AnthropicConverter._image_source(block))
                elif block_type == 'tool_use':
                    flush()
                    new_messages.append({
                        'role': 'tool_call',
                        'content': {
                            'name': block['name'],
                            'arguments': block.get('input', {}),
                        },
                        **metadata,
                    })
                elif block_type == 'tool_result':
                    flush()
                    new_messages.append({
                        'role': 'tool_response',
                        'content': AnthropicConverter._block_content(block.get('content', ''), media),
                        **metadata,
                    })
                else:
                    raise ValueError(f'Unsupported Anthropic content block type: {block_type}')
            flush()
        return new_messages

    @staticmethod
    def _image_source(block: Dict[str, Any]) -> str:
        """Pull a usable image reference out of an Anthropic image block.

        Base64 blocks become a ``data:`` URI (so a single string carries both media type and bytes);
        URL blocks yield the plain URL. Anything else is a malformed block and raises.
        """
        source = block.get('source', {})
        source_type = source.get('type')
        if source_type == 'base64':
            media_type = source.get('media_type')
            data = source.get('data')
            if not media_type or not data:
                raise ValueError(f'Invalid Anthropic base64 image block: {block}')
            return f'data:{media_type};base64,{data}'
        if source_type == 'url':
            url = source.get('url')
            if not url:
                raise ValueError(f'Invalid Anthropic URL image block: {block}')
            return url
        raise ValueError(f'Unsupported Anthropic image source type: {source_type}')

    @classmethod
    def _block_content(cls, content: Any, media: Dict[str, List]) -> Any:
        """Flatten a tool_result's content, which may itself be a block list, into text.

        A plain string is returned unchanged; a block list is flattened the same way a turn's content
        is (text appended, images turned into ``<image>`` placeholders plus a lifted source).
        """
        if not isinstance(content, list):
            return content
        parts: List[str] = []
        for block in content:
            block_type = block.get('type')
            if block_type == 'text':
                parts.append(block.get('text', ''))
            elif block_type == 'image':
                parts.append('<image>')
                media['images'].append(cls._image_source(block))
            else:
                raise ValueError(f'Unsupported Anthropic content block type: {block_type}')
        return ''.join(parts)
