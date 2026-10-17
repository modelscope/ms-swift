# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import ast
from typing import Any, Collection, Dict, List, Optional, Sequence, Type

__all__ = [
    'FORMAT_MAPPING', 'FormatConverter', 'Message', 'Messages', 'get_converter', 'list_formats', 'register_format'
]

Message = Dict[str, Any]
Messages = List[Message]

# format_name -> converter class.
FORMAT_MAPPING: Dict[str, Type['FormatConverter']] = {}


def register_format(converter_cls: Type['FormatConverter'] = None, *, exist_ok: bool = False):
    """Register a format, keyed by its ``format_name``. Usable bare or with keywords."""

    def _register(cls: Type['FormatConverter']) -> Type['FormatConverter']:
        format_name = cls.format_name
        assert format_name, f'{cls.__name__} must set `format_name`.'
        if not exist_ok and format_name in FORMAT_MAPPING:
            raise ValueError(f'format `{format_name}` is already registered '
                             f'by {FORMAT_MAPPING[format_name].__name__}.')
        FORMAT_MAPPING[format_name] = cls
        return cls

    return _register if converter_cls is None else _register(converter_cls)


def get_converter(columns: Collection[str],
                  *,
                  format_name: Optional[str] = None,
                  aliases: Optional[Dict[str, str]] = None,
                  **kwargs) -> 'FormatConverter':
    """Pick the converter for a dataset with these column names, and instantiate it.

    An explicit ``format_name`` wins (a dataset whose columns lie can be pinned); else every
    format is asked ``detect``, lowest priority first; else the highest-priority fallback.
    ``aliases`` are caller renames applied before detection (legacy's ``--columns``). Only names
    are needed, so this works on an IterableDataset, a dict of rows, or a literal list.
    """
    columns = set(columns)
    if aliases:
        columns = {aliases.get(column, column) for column in columns}

    if format_name is not None:
        if format_name not in FORMAT_MAPPING:
            raise ValueError(f'format `{format_name}` is not registered. Available: {sorted(FORMAT_MAPPING)}')
        return FORMAT_MAPPING[format_name](aliases=aliases, **kwargs)

    candidates = sorted(FORMAT_MAPPING.values(), key=lambda cls: cls.priority)
    for cls in candidates:
        if cls.detect(columns):
            return cls(aliases=aliases, **kwargs)
    for cls in candidates:
        if cls.is_fallback:
            return cls(aliases=aliases, **kwargs)
    raise ValueError(f'No format matched columns {sorted(columns)} and no fallback is registered.')


def list_formats() -> List[str]:
    """Registered format names, in detection order."""
    return [cls.format_name for cls in sorted(FORMAT_MAPPING.values(), key=lambda cls: cls.priority)]


class FormatConverter:
    """One raw row shape, and how to rewrite it as standard ``messages``.

    Subclasses declare ``format_name`` / ``aliases`` / ``priority``, then implement ``detect`` and
    ``convert``.
    """

    format_name: Optional[str] = None
    # Detection order, lowest first: formats are not mutually exclusive (a `conversations` dataset
    # also carrying `instruction` is a conversation dataset), so the asking order matters.
    priority: int = 100
    # Column renames this format applies before reading a row: alias -> standard name.
    aliases: Dict[str, str] = {}
    # True for the format used when nothing else claims the columns; exactly one format sets it.
    is_fallback: bool = False

    # `tool` is a synonym of `tool_response` -- legacy's message check accepted both.
    STANDARD_ROLES = ('system', 'user', 'assistant', 'tool_call', 'tool_response', 'tool')

    # Media columns are a property of the row, not the dialogue shape, so every format answers to
    # the singular spelling too; kept apart from `aliases` so a subclass cannot drop them.
    MEDIA_ALIASES = {'image': 'images', 'audio': 'audios', 'video': 'videos'}

    def __init__(self, *, aliases: Optional[Dict[str, str]] = None, **kwargs):
        # Caller aliases win on conflict: the caller knows their dataset, the class the common cases.
        self.aliases = {**self.MEDIA_ALIASES, **type(self).aliases, **(aliases or {})}
        self.random_state = kwargs.pop('random_state', None)
        self.kwargs = kwargs

    @classmethod
    def detect(cls, columns: Collection[str]) -> bool:
        """Whether this format recognises a dataset with these columns.

        Decided on column names alone -- a dataset's row shape is fixed. Content sharing one shape
        across providers (a ``messages`` column holding plain, OpenAI or Anthropic content) is not
        a format distinction: it is handled per message inside the owning format's ``convert``.
        """
        raise NotImplementedError

    def convert(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Rewrite one raw row as a standard row, or return ``None`` to drop it.

        Dropping (rather than raising) is how a structurally unusable row is skipped: a public
        dataset with a few bad rows out of a million should not fail the run.
        """
        raise NotImplementedError

    def apply_aliases(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """A copy of ``row`` with aliased keys renamed to their standard names."""
        renames = self.resolve_aliases(row.keys())
        return {renames.get(key, key): value for key, value in row.items()}

    def resolve_aliases(self, keys: Collection[str]) -> Dict[str, str]:
        """Of these column names, which get renamed to which standard name: ``alias -> standard``.

        A standard name the row spells out itself beats every alias of it; among competing
        aliases, the first declared in ``aliases`` wins. A loser keeps its own name (downstream
        may still want it). Both rules read only ``keys``, so the outcome never depends on the raw
        column order -- unlike legacy's last-alias-wins loop, which dropped *both* names.
        """
        keys = set(keys)
        ranks = {alias: rank for rank, alias in enumerate(self.aliases)}
        winners: Dict[str, str] = {}
        for key in keys:
            standard = self.aliases.get(key, key)
            if standard == key or standard in keys:
                continue
            incumbent = winners.get(standard)
            if incumbent is None or ranks[key] < ranks[incumbent]:
                winners[standard] = key
        return {key: standard for standard, key in winners.items()}

    @staticmethod
    def parse_literal(value: Any) -> Any:
        """Parse a Python literal that arrived as a string, else pass the value through.

        DataFrame dumps store a list column as its ``repr`` (``history`` comes back as
        ``"[['q', 'a']]"``). A string that is not a literal yields ``None`` -- safe only for
        fields that are always lists; see PreferenceConverter._recover_message_list for the
        list-or-string case.
        """
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return None
        return value

    @staticmethod
    def history_to_messages(history: Sequence[Sequence[Optional[str]]], system: Optional[str] = None) -> Messages:
        """Flatten ``[[query, response], ...]`` into messages, skipping ``None`` turns.

        A ``None`` response is a legitimate unanswered final turn (inference input), so it is
        skipped rather than emitted as an empty assistant message.
        """
        messages: Messages = []
        if system is not None:
            messages.append({'role': 'system', 'content': system})
        for turn in history:
            query, response = turn[0], turn[1]
            if query is not None:
                messages.append({'role': 'user', 'content': query})
            if response is not None:
                messages.append({'role': 'assistant', 'content': response})
        return messages

    @staticmethod
    def prepend_system(messages: Messages, system: Optional[str]) -> Messages:
        """Put ``system`` at the front, unless the dialogue already opens with one."""
        if system is None:
            return messages
        if messages and messages[0].get('role') == 'system':
            return messages
        return [{'role': 'system', 'content': system}, *messages]
