# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import Any, Collection, Dict, List, Optional, Tuple

import json

from .base import FormatConverter, Messages, register_format

__all__ = ['OpenAIConverter']


@register_format
class OpenAIConverter(FormatConverter):
    """The messages/dialogue row format (a ``messages`` / ``conversations`` column).

    The column holds plain role/content messages, OpenAI ``tool_calls``, OpenAI/v5 inline
    multimodal ``content`` parts, or Anthropic content blocks -- one row shape, so a single
    messages format; ``normalize_content`` rewrites each message per its own content.
    """

    format_name = 'openai'
    priority = 10
    aliases = {
        'conversation': 'messages',
        'conversations': 'messages',
        'system_prompt': 'system',
    }

    ROLE_KEYS = ('role', 'from')
    CONTENT_KEYS = ('content', 'value')
    ROLE_ALIASES = {
        'user': 'user',
        'human': 'user',
        'assistant': 'assistant',
        'gpt': 'assistant',
        'bot': 'assistant',
        'chatgpt': 'assistant',
        'function_call': 'tool_call',
        'tool_call': 'tool_call',
        'function_response': 'tool_response',
        'observation': 'tool_response',
        'observations': 'tool_response',
        'tool_response': 'tool_response',
        'tool': 'tool_response',
        'system': 'system',
    }

    def __init__(self,
                 *,
                 role_key: Optional[str] = None,
                 content_key: Optional[str] = None,
                 user_role: Optional[str] = None,
                 assistant_role: Optional[str] = None,
                 system_role: str = 'system',
                 inner_key: Optional[str] = None,
                 **kwargs):
        """Args: ``role_key``/``content_key`` pin the message's role/content keys; ``user_role``/
        ``assistant_role`` pin the role values meaning user/assistant (a pin replaces the whole
        alias family); ``system_role`` names the system role value; ``inner_key`` reaches through
        when the dialogue nests one level deeper (``{'inner': [...]}``)."""
        super().__init__(**kwargs)
        self.role_keys = (role_key, ) if role_key else self.ROLE_KEYS
        self.content_keys = (content_key, ) if content_key else self.CONTENT_KEYS
        self.system_role = system_role
        self.inner_key = inner_key

        role_map = dict(self.ROLE_ALIASES)
        if user_role:
            role_map = {k: v for k, v in role_map.items() if v != 'user'}
            role_map[user_role] = 'user'
        if assistant_role:
            role_map = {k: v for k, v in role_map.items() if v != 'assistant'}
            role_map[assistant_role] = 'assistant'
        role_map[system_role] = 'system'
        self.role_map = role_map

    @classmethod
    def detect(cls, columns: Collection[str]) -> bool:
        return any(column in columns for column in ('messages', 'conversation', 'conversations'))

    def convert(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.apply_aliases(row)
        system = row.pop('system', None)

        built = self.build_messages(row.get('messages'), system)
        if not built:
            return None
        messages, media = built
        row['messages'] = messages
        # Anthropic image blocks are lifted out of the dialogue into the top-level media columns,
        # like every other multimodal dataset; mixing both representations is rejected.
        for key, values in media.items():
            if values:
                assert not row.get(key), f'Cannot mix Anthropic content blocks with the top-level `{key}` field.'
                row[key] = values

        if row.get('rejected_messages') is not None:
            rejected = self.build_messages(row['rejected_messages'], system)
            if not rejected:
                return None
            row['rejected_messages'] = rejected[0]
        return row

    def build_messages(self, raw: Any, system: Optional[str] = None) -> Optional[Tuple[Messages, Dict[str, List]]]:
        """Normalise one dialogue value into ``(messages, media)``, or ``None`` if unusable.

        ``media`` collects anything ``normalize_content`` lifts out of the dialogue (Anthropic
        images); it stays empty for plain and OpenAI content.
        """
        if self.inner_key is not None and isinstance(raw, dict):
            raw = raw.get(self.inner_key)
        raw = self.parse_literal(raw)
        if not isinstance(raw, (list, tuple)) or not raw:
            return None
        if not all(isinstance(turn, dict) for turn in raw):
            return None

        media: Dict[str, List] = {'images': []}
        raw = self.normalize_content(list(raw), media)
        if not raw:
            return None

        if self.is_turn_pairs(raw):
            messages = self.expand_turn_pairs(raw)
        else:
            messages = self.normalise_messages(raw)
        if not messages:
            return None
        return self.prepend_system(messages, system), media

    def normalize_content(self, messages: Messages, media: Dict[str, List]) -> Messages:
        """Rewrite each message's provider-specific content into plain role/content messages.

        Per message, not per dataset -- turns are not uniformly shaped, so classifying from a
        sample row would miss the multimodal turns. Anthropic content blocks expand with images
        lifted into ``media``; an assistant ``tool_calls`` list expands into standalone
        ``tool_call`` messages; everything else passes through (the template handles inline parts).
        """
        normalized: Messages = []
        for message in messages:
            content = message.get('content')
            if isinstance(content, list) and self._has_anthropic_blocks(content):
                self._expand_anthropic_message(message, normalized, media)
                continue
            # Resolve the role through ``role_map`` (not the literal ``role`` key): a sharegpt turn
            # keeps its role under ``from`` with an alias value, and its tool_calls must expand too.
            role = self.first_present(message, self.role_keys)
            is_assistant = (role is not None
                            and self.role_map.get(str(role).replace('-', '_')) == 'assistant')
            tool_calls = message.get('tool_calls') if is_assistant else None
            if not tool_calls:
                normalized.append(message)
                continue
            if content:
                normalized.append(
                    {key: value
                     for key, value in message.items() if key in {'role', 'content', 'loss', 'loss_scale'}})
            for tool_call in tool_calls:
                function = tool_call.get('function', tool_call)
                arguments = function.get('arguments', {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                tool_message: Dict[str, Any] = {'role': 'tool_call', 'content': {
                    'name': function['name'],
                    'arguments': arguments,
                }}
                for key in ('loss', 'loss_scale'):
                    if key in message:
                        tool_message[key] = message[key]
                normalized.append(tool_message)
        return normalized

    def is_turn_pairs(self, raw: List[Dict[str, Any]]) -> bool:
        """Whether each element holds a whole turn (``{'user': q, 'assistant': a}``) rather than one
        message -- recognised by the keys present, not by legacy's "has neither ``role`` nor
        ``content``", which also claimed a malformed row that simply lost its keys.
        """
        first = raw[0]
        if any(key in first for key in self.role_keys):
            return False
        return 'user' in first or 'assistant' in first

    @staticmethod
    def expand_turn_pairs(raw: List[Dict[str, Any]]) -> Messages:
        """Expand ``[{'user': q, 'assistant': a}, ...]`` into one message per party."""
        messages: Messages = []
        for turn in raw:
            for role in ('user', 'assistant'):
                content = turn.get(role)
                if content is not None:
                    messages.append({'role': role, 'content': content})
        return messages

    def normalise_messages(self, raw: List[Dict[str, Any]]) -> Messages:
        """Rewrite each message's keys and role value into the standard ones.

        Unknown roles pass through untouched: the message check downstream is the right place to
        reject them, with the offending row in the error.
        """
        messages: Messages = []
        for turn in raw:
            role = self.first_present(turn, self.role_keys)
            content = self.first_present(turn, self.content_keys)
            if role is None:
                continue
            role = self.role_map.get(str(role).replace('-', '_'), role)
            message = {'role': role, 'content': content}
            for key in ('loss', 'loss_scale'):
                if key in turn:
                    message[key] = turn[key]
            messages.append(message)
        return messages

    @staticmethod
    def first_present(turn: Dict[str, Any], keys: Collection[str]) -> Any:
        """Value of the first key that is present; ``None`` when none are.

        First-wins, not last-wins: a message carrying both ``content`` and ``value`` means the one
        named by the earlier (more standard) key.
        """
        for key in keys:
            if key in turn:
                return turn[key]
        return None

    # Anthropic spells content as typed blocks the template layer cannot read: images carry bytes
    # under ``source`` and tools are ``tool_use``/``tool_result`` blocks. Static stateless helpers.

    @staticmethod
    def _has_anthropic_blocks(content: Any) -> bool:
        """Whether a content list carries Anthropic-only blocks.

        The v5 canonical inline parts (``text`` / ``image_url`` / a ``source``-less ``image``) are
        NOT Anthropic and must fall through to the template, so only ``tool_use`` / ``tool_result``
        / an ``image`` carrying a ``source`` count.
        """
        return any(
            isinstance(block, dict) and (block.get('type') in {'tool_use', 'tool_result'}
                                         or (block.get('type') == 'image' and 'source' in block))
            for block in content)

    @staticmethod
    def _expand_anthropic_message(message: Dict[str, Any], new_messages: Messages, media: Dict[str, List]) -> None:
        """Expand one message's Anthropic content blocks into ``new_messages``.

        Consecutive text/image blocks accumulate into one message; a tool_use or tool_result block
        first flushes that accumulated text, then emits its own message, preserving order. Images
        are lifted into ``media`` for ``convert`` to place in the top-level media columns.
        """
        metadata = {key: message[key] for key in ('loss', 'loss_scale') if key in message}
        role = message['role']
        pending: List[str] = []
        for block in message['content']:
            block_type = block.get('type')
            if block_type == 'text':
                pending.append(block.get('text', ''))
            elif block_type == 'image':
                pending.append('<image>')
                media['images'].append(OpenAIConverter._image_source(block))
            elif block_type == 'tool_use':
                OpenAIConverter._flush_text(new_messages, pending, role, metadata)
                new_messages.append({
                    'role': 'tool_call',
                    'content': {
                        'name': block['name'],
                        'arguments': block.get('input', {}),
                    },
                    **metadata,
                })
            elif block_type == 'tool_result':
                OpenAIConverter._flush_text(new_messages, pending, role, metadata)
                new_messages.append({
                    'role': 'tool_response',
                    'content': OpenAIConverter._block_content(block.get('content', ''), media),
                    **metadata,
                })
            else:
                raise ValueError(f'Unsupported Anthropic content block type: {block_type}')
        OpenAIConverter._flush_text(new_messages, pending, role, metadata)

    @staticmethod
    def _flush_text(new_messages: Messages, pending: List[str], role: str, metadata: Dict[str, Any]) -> None:
        """Emit the accumulated text as one message and clear the buffer; a no-op when empty."""
        if pending:
            new_messages.append({'role': role, 'content': ''.join(pending), **metadata})
            pending.clear()

    @staticmethod
    def _image_source(block: Dict[str, Any]) -> str:
        """Pull a usable image reference out of an Anthropic image block.

        Base64 blocks become a ``data:`` URI (one string carries both media type and bytes); URL
        blocks yield the plain URL. Anything else is malformed and raises.
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

    @staticmethod
    def _block_content(content: Any, media: Dict[str, List]) -> Any:
        """Flatten a tool_result's content, which may itself be a block list, into text.

        A plain string is returned unchanged; a block list is flattened the same way a turn's
        content is (text appended, images turned into ``<image>`` placeholders plus a lifted
        source).
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
                media['images'].append(OpenAIConverter._image_source(block))
            else:
                raise ValueError(f'Unsupported Anthropic content block type: {block_type}')
        return ''.join(parts)
