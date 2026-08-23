# Copyright (c) ModelScope Contributors. All rights reserved.
"""Datasets with one example per row spread across flat columns.

``system`` / ``query`` / ``response`` (and their many aliases), optionally with a ``history`` column
holding earlier turns, and optionally a ``rejected_response`` for flat preference data. This is
old-ms-swift's own format and the shape most single-turn datasets take once you look past the column
names.

:class:`ResponseConverter` is the registered fallback: when a dataset carries no dialogue column and
matches no more specific format, treating some column as the response is the last reasonable guess.
Legacy made the same choice -- ``AutoPreprocessor`` ended its if-chain with ``ResponsePreprocessor``.
The Alpaca variant, which only differs by joining two columns into the user turn, lives in
``alpaca.py`` next to this class rather than inside it.
"""
from __future__ import annotations

import os
from typing import Any, Collection, Dict, Optional

from .base import FormatConverter, register_format

__all__ = ['ResponseConverter']


@register_format
class ResponseConverter(FormatConverter):
    """Flat ``query``/``response`` rows, with optional multi-turn ``history`` and preference pair."""

    format_name = 'response'
    # Last: this is the fallback, so every more specific format must be asked before it. Its own
    # `detect` is never the deciding call (the fallback pass in the factory is), but the high number
    # keeps it last should someone make its detect return True.
    priority = 1000
    is_fallback = True
    aliases = {
        'system_prompt': 'system',
        'prompt': 'query',
        'input': 'query',
        'instruction': 'query',
        'question': 'query',
        'problem': 'query',
        'answer': 'response',
        'output': 'response',
        'targets': 'response',
        'target': 'response',
        'solution': 'response',
        'text': 'response',
        'completion': 'response',
        'content': 'response',
    }

    @classmethod
    def detect(cls, columns: Collection[str]) -> bool:
        # A positive detect is offered for the common explicit case, but correctness does not rely on
        # it: even if this returns False, the factory's fallback pass routes here.
        standardised = {cls.aliases.get(column, column) for column in columns}
        return 'response' in standardised

    def convert(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.apply_aliases(row)
        response = self.pick_response(row.pop('response', None))
        query = row.pop('query', None)
        system = row.pop('system', None)

        history = self.parse_literal(row.pop('history', None)) or []
        # The current turn is the last turn; a None response is a valid unanswered final turn.
        turns = [*history, [query, response]]
        messages = self.history_to_messages(turns, system)
        if not messages:
            return None
        row['messages'] = messages

        # Flat preference data: the rejected answer shares the same prompt as the chosen one, so it
        # is the same dialogue with its final response swapped. Legacy left this flat form to a
        # separate check downstream and only converted the `rejected_messages` list form; here the
        # two forms leave this layer identically shaped.
        rejected = self.pick_response(row.pop('rejected_response', None))
        if rejected is not None:
            rejected_turns = [*history, [query, rejected]]
            row['rejected_messages'] = self.history_to_messages(rejected_turns, system)
        return row

    def pick_response(self, response: Any) -> Any:
        """Collapse a multi-answer response column to a single answer.

        Some datasets store several reference answers in a list. Default to the first for
        reproducibility; ``RANDOM_DATASET_RESPONSE=1`` picks one at random, matching legacy's env
        switch (used when the extra answers are meant as augmentation, not alternatives).
        """
        if not isinstance(response, (list, tuple)):
            return response
        if not response:
            return None
        from transformers.utils import strtobool
        if strtobool(os.environ.get('RANDOM_DATASET_RESPONSE', 'False')):
            import random
            return random.choice(response)
        return response[0]
