# Copyright (c) ModelScope Contributors. All rights reserved.
"""Datasets that ship a dialogue in a ``messages`` (or ``conversations``) column.

This is the canonical chat shape -- a list of ``{'role', 'content'}`` turns -- so it is also the base
for the two provider *dialects* that only differ in how a single turn is encoded:

- **swift** (the plain form): ``{'role': 'user', 'content': 'q'}``, plus the shorthands the wild
  uses -- ``{'from', 'value'}`` key names, and one-element-per-turn ``{'user': q, 'assistant': a}``.
- **openai**: an assistant turn carries a ``tool_calls`` list; each call becomes its own ``tool_call``
  message. See :meth:`openai_to_messages`.
- **anthropic**: ``content`` is a list of typed blocks (text / image / tool_use / tool_result); the
  block parsing lives in ``anthropic.py`` and is delegated to lazily, keeping that provider's bulk
  out of this file while still letting the auto path reach it.

``message_format`` selects the dialect; ``'auto'`` (the default) detects it per row, exactly as
legacy main's ``MessagesPreprocessor`` does. The two dialects are *not* separate top-level formats
picked by column name -- all three arrive in the same ``messages`` column and are only distinguishable
per row -- which is why the factory routes every messages dataset here and the dialect is resolved
inside :meth:`normalize_provider_messages`. ``anthropic.py`` exposes a thin pin subclass for a dataset
that is known to be Anthropic and wants to skip auto-detection.
"""
from __future__ import annotations
from typing import Any, Collection, Dict, List, Optional, Tuple

import json

from .base import FormatConverter, Messages, register_format

__all__ = ['OpenAIConverter']


@register_format
class OpenAIConverter(FormatConverter):
    """The messages/dialogue format, covering the swift / openai / anthropic dialects."""

    format_name = 'openai'
    # Asked first: a dataset with both a dialogue column and an `instruction` column is a dialogue
    # dataset that happens to carry an instruction, not an Alpaca dataset.
    priority = 10
    aliases = {
        'conversation': 'messages',
        'conversations': 'messages',
        'system_prompt': 'system',
    }

    # Keys a single message may use for its role and its text.
    ROLE_KEYS = ('role', 'from')
    CONTENT_KEYS = ('content', 'value')
    # Role values seen in the wild, mapped to the standard set. `-` is normalised to `_` before
    # lookup, so `function-call` and `function_call` both land on `tool_call`.
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
                 message_format: str = 'auto',
                 **kwargs):
        """
        Args:
            role_key: Pin the key holding a message's role, instead of trying :attr:`ROLE_KEYS`.
            content_key: Pin the key holding a message's text.
            user_role: Pin the role value meaning "user"; other user-ish values stop being recognised.
            assistant_role: Pin the role value meaning "assistant".
            system_role: Role value this dataset uses for the system prompt. Only needed when it is
                not literally ``'system'``.
            inner_key: Key to reach through when the dialogue is nested one level deeper, i.e. the
                column holds ``{'inner': [...]}`` rather than ``[...]``.
            message_format: Which dialect to apply -- ``'auto'`` / ``'swift'`` / ``'openai'`` /
                ``'anthropic'``. ``'auto'`` detects per row.
        """
        super().__init__(**kwargs)
        self.role_keys = (role_key, ) if role_key else self.ROLE_KEYS
        self.content_keys = (content_key, ) if content_key else self.CONTENT_KEYS
        self.system_role = system_role
        self.inner_key = inner_key
        self.message_format = message_format

        role_map = dict(self.ROLE_ALIASES)
        # A pinned role replaces the whole family rather than joining it: a dataset that says its
        # user role is `human` is telling us `user` may mean something else there.
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
        # Anthropic image blocks are lifted out of the dialogue into a top-level `images` column,
        # matching how every other multimodal dataset carries media. A dataset that both used content
        # blocks AND set `images` itself would be ambiguous, so that is rejected rather than merged.
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

        ``media`` collects anything the provider layer pulls out of the dialogue (Anthropic images);
        it is empty for the swift and openai dialects.
        """
        if self.inner_key is not None and isinstance(raw, dict):
            raw = raw.get(self.inner_key)
        raw = self.parse_literal(raw)
        if not isinstance(raw, (list, tuple)) or not raw:
            return None
        if not all(isinstance(turn, dict) for turn in raw):
            return None

        media: Dict[str, List] = {'images': []}
        raw = self.normalize_provider_messages(list(raw), media)
        if not raw:
            return None

        if self.is_turn_pairs(raw):
            messages = self.expand_turn_pairs(raw)
        else:
            messages = self.normalise_messages(raw)
        if not messages:
            return None
        return self.prepend_system(messages, system), media

    # -- provider dialects -----------------------------------------------------------------------

    def normalize_provider_messages(self, messages: List[Dict[str, Any]], media: Dict[str, List]) -> Messages:
        """Resolve the dialect and expand provider-specific turns into plain role/content messages.

        Auto-detection matches legacy main: an assistant ``tool_calls`` list means OpenAI; a
        ``content`` that is a list of typed blocks (tool_use / tool_result / sourced image) means
        Anthropic; otherwise the messages are already plain and pass through untouched.
        """
        message_format = self.message_format
        if message_format == 'auto':
            if any(message.get('tool_calls') for message in messages):
                message_format = 'openai'
            elif any(
                    isinstance(message.get('content'), list) and any(
                        block.get('type') in {'tool_use', 'tool_result'}
                        or (block.get('type') == 'image' and 'source' in block)
                        for block in message['content'] if isinstance(block, dict)) for message in messages):
                message_format = 'anthropic'
            else:
                message_format = 'swift'

        if message_format == 'openai':
            return self.openai_to_messages(messages)
        if message_format == 'anthropic':
            # Lazy import breaks the import cycle: anthropic.py subclasses this class, so it cannot be
            # imported at module load, but its parser is only needed here at call time.
            from .anthropic import AnthropicConverter
            return AnthropicConverter.anthropic_to_messages(messages, media)
        return messages

    @staticmethod
    def openai_to_messages(messages: List[Dict[str, Any]]) -> Messages:
        """Expand OpenAI assistant ``tool_calls`` into standalone ``tool_call`` messages.

        A ``tool_calls`` entry may wrap its payload under a ``function`` key (the OpenAI shape) or
        carry ``name``/``arguments`` directly; ``arguments`` is a JSON string in OpenAI's schema and
        is parsed to a dict, but left as-is if it does not parse. A tool-calling assistant turn may
        also carry ordinary content, which is preserved as its own message before the calls.
        """
        normalized: Messages = []
        for message in messages:
            tool_calls = message.get('tool_calls') if message.get('role') == 'assistant' else None
            if not tool_calls:
                normalized.append(message)
                continue
            if message.get('content'):
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

    # -- swift dialect (plain role/content, from/value, turn-pairs) -------------------------------

    def is_turn_pairs(self, raw: List[Dict[str, Any]]) -> bool:
        """Whether each element holds a whole turn (``{'user': q, 'assistant': a}``) rather than one
        message.

        Recognised by the keys present -- a ``user``/``assistant`` pair and no role key -- instead of
        legacy's "has neither ``role`` nor ``content``, so it must be the other shape", which would
        also claim a malformed row that simply lost its keys.
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

        Unknown roles are passed through untouched rather than dropped or guessed: a dataset may use a
        role this converter has not seen, and the message check downstream is the right place to
        reject it, with the offending row in the error.
        """
        messages: Messages = []
        for turn in raw:
            role = self.first_present(turn, self.role_keys)
            content = self.first_present(turn, self.content_keys)
            if role is None:
                continue
            role = self.role_map.get(str(role).replace('-', '_'), role)
            message = {'role': role, 'content': content}
            # Carry through the two per-message fields the standard shape allows, so a dataset that
            # ships token-level loss weights does not lose them here.
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
