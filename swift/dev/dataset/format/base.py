# Copyright (c) ModelScope Contributors. All rights reserved.
"""Turning whatever shape a dataset ships in into one standard row shape.

Every dataset in the wild names its fields differently -- ``query``/``prompt``/``question``,
``response``/``output``/``answer``, ``conversations``/``messages`` -- and some nest a whole dialogue
in a single column. A :class:`FormatConverter` knows one such shape and rewrites it into the standard
one: a ``messages`` list of ``{'role', 'content'}``.

How this differs from legacy's ``preprocessor/core.py``, which did the same job:

- **Field aliases are data, not constructor code.** Legacy's ``ResponsePreprocessor.__init__`` wrote
  23 aliases into ``self.columns`` imperatively, mixed in the same dict as the caller's own column
  overrides, so neither could be read without running the other. Here :attr:`FormatConverter.aliases`
  is a plain class attribute.
- **Each format states how to recognise itself.** Legacy centralised detection in
  ``AutoPreprocessor._get_preprocessor``, an if-chain that every new format had to be threaded into.
  Here :meth:`FormatConverter.detect` lives on the format, and the factory only sorts candidates.
- **Detection order is declared, not implied.** Legacy's order was whatever its if-chain happened to
  be. Here it is :attr:`FormatConverter.priority`, so it survives someone reordering the code.
- **One base class.** Legacy's ``AutoPreprocessor`` was *not* a ``RowPreprocessor`` subclass, so the
  auto path and the explicit path had different interfaces.
- **Conversion returns a new row.** Legacy mixed in-place mutation (``to_std_messages``, whose
  docstring literally says ``# inplace``) with returning fresh objects (``sharegpt_to_messages``) in
  the same class.
"""
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

    This is the factory. Order of resolution:

    1. ``format_name`` -- an explicit choice always wins, so a dataset whose columns lie about its
       shape can be pinned.
    2. Otherwise every registered format is asked :meth:`FormatConverter.detect`, lowest
       :attr:`FormatConverter.priority` first, and the first to claim the columns wins.
    3. Otherwise the highest-priority format that declares itself a fallback.

    Args:
        columns: The dataset's column names. Only names are needed, not types -- so this is callable
            on an ``IterableDataset`` whose features are resolved, on a plain dict of rows, or in a
            test with a literal list.
        format_name: Pin a format instead of detecting one.
        aliases: Extra column renames applied *before* detection, for a dataset whose fields are
            named so unusually that no format recognises it (legacy's ``--columns``). These are the
            caller's business and are kept apart from :attr:`FormatConverter.aliases`, which is the
            format's own built-in knowledge.
        kwargs: Passed to the converter's constructor.
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

    Subclasses declare :attr:`format_name`, :attr:`aliases` and :attr:`priority`, then implement
    :meth:`detect` and :meth:`convert`.
    """

    format_name: Optional[str] = None
    # Detection order, lowest first. Explicit because the formats are not mutually exclusive: a
    # dataset carrying both `conversations` and `instruction` is a conversation dataset that happens
    # to have an instruction column, so the conversation format must be asked first. Legacy encoded
    # exactly this in the order of an if-chain, where it was invisible and easy to break.
    priority: int = 100
    # Column renames this format applies before reading a row: alias -> standard name. Pure data, so
    # a reader can see every name a format answers to without executing anything.
    aliases: Dict[str, str] = {}
    # True for the format used when nothing else claims the columns. Exactly one format should set it.
    is_fallback: bool = False

    # Roles the standard shape allows. `tool` is accepted as a synonym of `tool_response`, which is
    # what legacy's message check did.
    STANDARD_ROLES = ('system', 'user', 'assistant', 'tool_call', 'tool_response', 'tool')

    def __init__(self, *, aliases: Optional[Dict[str, str]] = None, **kwargs):
        # Caller-supplied aliases are applied on top of the class's own, and win on conflict: the
        # caller knows their dataset, the class only knows the common cases.
        self.aliases = {**type(self).aliases, **(aliases or {})}
        self.kwargs = kwargs

    @classmethod
    def detect(cls, columns: Collection[str]) -> bool:
        """Whether this format recognises a dataset with these column names."""
        raise NotImplementedError

    def convert(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Rewrite one raw row as a standard row, or return ``None`` to drop it.

        Returning ``None`` is how a structurally unusable row (empty dialogue, unparseable nesting)
        is dropped without raising -- a public dataset with a few thousand bad rows out of a million
        should not fail the run.
        """
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------------------------------

    def apply_aliases(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """A copy of ``row`` with aliased keys renamed to their standard names.

        A standard name already present is never overwritten: if a row has both ``response`` and
        ``output``, ``response`` is the one meant. Legacy's ``_to_std_key`` looped over the alias list
        assigning as it went, so the *last* alias in the list won and the outcome depended on the
        order of a literal.
        """
        new_row = {}
        for key, value in row.items():
            new_key = self.aliases.get(key, key)
            if new_key in new_row and key != new_key:
                continue
            new_row[new_key] = value
        return new_row

    @staticmethod
    def parse_literal(value: Any) -> Any:
        """Parse a Python literal that arrived as a string, else pass the value through.

        Datasets built by dumping a DataFrame routinely store a list column as its ``repr``, so
        ``history`` comes back as the string ``"[['q', 'a']]"``.
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

        A ``None`` response is legitimate and means an unanswered final turn (inference input), so it
        is skipped rather than emitted as an empty assistant message.
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
