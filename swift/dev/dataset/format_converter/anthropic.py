# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import Any, Dict, List, Optional

from .base import Messages, register_format
from .openai import OpenAIConverter

__all__ = ['AnthropicConverter']


@register_format
class AnthropicConverter(OpenAIConverter):
    """Pin the Anthropic dialect. Auto-detection is handled by :class:`OpenAIConverter`."""

    format_name = 'anthropic'
    priority = 12
    is_fallback = False

    def __init__(self, **kwargs):
        kwargs['message_format'] = 'anthropic'
        super().__init__(**kwargs)

    @classmethod
    def detect(cls, columns) -> bool:
        ## 这里为什么是False？什么时候anthropic的converter才能生效？
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
            flush = AnthropicConverter._flush_text
            ## swift的标准dataset格式，改为content是list是否合适？这样converter的逻辑
            ## 就不是合并为一条再拆解，而是在这里把特殊字符的string拆开为list
            for block in content:
                block_type = block.get('type')
                if block_type == 'text':
                    pending.append(block.get('text', ''))
                elif block_type == 'image':
                    pending.append('<image>')
                    media['images'].append(AnthropicConverter._image_source(block))
                elif block_type == 'tool_use':
                    flush(new_messages, pending, message['role'], metadata)
                    new_messages.append({
                        'role': 'tool_call',
                        'content': {
                            'name': block['name'],
                            'arguments': block.get('input', {}),
                        },
                        **metadata,
                    })
                elif block_type == 'tool_result':
                    flush(new_messages, pending, message['role'], metadata)
                    new_messages.append({
                        'role': 'tool_response',
                        'content': AnthropicConverter._block_content(block.get('content', ''), media),
                        **metadata,
                    })
                else:
                    raise ValueError(f'Unsupported Anthropic content block type: {block_type}')
            flush(new_messages, pending, message['role'], metadata)
        return new_messages

    @staticmethod
    def _flush_text(new_messages: Messages, pending: List[str], role: str, metadata: Dict[str, Any]) -> None:
        """Emit the accumulated text as one message and clear the buffer; a no-op when empty.

        Every argument is explicit rather than captured from the enclosing loop: a closure over a
        loop variable is the kind of thing that works until someone defers the call.
        """
        if pending:
            new_messages.append({'role': role, 'content': ''.join(pending), **metadata})
            pending.clear()

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
            ## video和audio块如何处理？
            elif block_type == 'image':
                parts.append('<image>')
                media['images'].append(cls._image_source(block))
            else:
                raise ValueError(f'Unsupported Anthropic content block type: {block_type}')
        return ''.join(parts)
