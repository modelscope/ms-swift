# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Any, Collection, Dict, Optional

from .base import register_format
from .openai import OpenAIConverter
from .response import ResponseConverter

__all__ = ['PreferenceConverter']


@register_format
class PreferenceConverter(ResponseConverter):
    """Top-level ``chosen`` / ``rejected`` -- the RLHF row shape behind DPO, reward modelling, KTO.

    The chosen side becomes the standard ``messages``, the rejected side ``rejected_messages``
    (full turns) or ``rejected_response`` (a single reply). Each side is normalised by the
    converter that owns its shape: a message list via a composed messages converter, a bare
    string via this class's own response handling.
    """

    format_name = 'preference'
    # Between openai (10) and alpaca (20): a dataset with BOTH `conversations` and `chosen` is a
    # conversation dataset, but a top-level `chosen` is a stronger signal than `instruction`.
    priority = 15
    is_fallback = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The message-list side is normalised by a composed OpenAIConverter, built lazily on the
        # first message-list row and cached; see `_messages_side`.
        self._messages_converter: Optional[OpenAIConverter] = None

    @classmethod
    def detect(cls, columns: Collection[str]) -> bool:
        # Exact top-level names, not the `rejected_*` standard keys: a dataset already carrying
        # `rejected_messages`/`rejected_response` is standard-shaped and belongs to openai/response.
        return 'chosen' in columns or 'rejected' in columns

    def convert(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.apply_aliases(row)
        chosen = self._recover_message_list(row.pop('chosen', None))
        rejected = self._recover_message_list(row.pop('rejected', None))

        # The rejected side is decided independently of the chosen side's shape: a message list
        # becomes `rejected_messages` verbatim; anything else folds to a single reply first.
        if self._is_message_list(rejected):
            row['rejected_messages'] = rejected
        elif rejected is not None:
            rejected = self.pick_response(rejected)
            if rejected is not None:
                row['rejected_response'] = rejected

        if self._is_message_list(chosen):
            # chosen is a full conversation, so it IS the messages: the messages converter
            # normalises roles/content, lifts inline media and prepends the system turn.
            row['messages'] = chosen
            return self._messages_side().convert(row)
        # chosen is a bare reply: the response converter (this class's own base) assembles it with
        # the shared query turn and folds a rejected_response string into place.
        row['response'] = chosen
        return super().convert(row)

    def _messages_side(self) -> OpenAIConverter:
        """The converter for a message-list ``chosen``, built lazily and cached.

        It is configured from this converter's own ``aliases`` and ``kwargs``, so a dataset pinning
        ``role_key``/``content_key`` applies to the dialogue too.
        """
        if self._messages_converter is None:
            self._messages_converter = OpenAIConverter(aliases=self.aliases, **self.kwargs)
        return self._messages_converter

    @classmethod
    def _recover_message_list(cls, value: Any) -> Any:
        """Recover a message list a DataFrame dump stored as its repr string; anything else as-is.

        ``parse_literal`` returns ``None`` for a non-literal string -- right for ``history``
        (always a list) but fatal here, where a plain ``chosen``/``rejected`` reply such as ``"the
        answer"`` is valid and must not be dropped. Only a string that parses to a list is replaced.
        """
        if isinstance(value, str):
            parsed = cls.parse_literal(value)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        return value

    @classmethod
    def _is_message_list(cls, value: Any) -> bool:
        """Whether ``value`` is a list of role/from-bearing message dicts rather than a bare reply."""
        return (isinstance(value, (list, tuple)) and len(value) > 0 and all(isinstance(m, dict) for m in value)
                and any(key in value[0] for key in OpenAIConverter.ROLE_KEYS))
