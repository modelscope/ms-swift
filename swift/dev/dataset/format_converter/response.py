# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Any, Collection, Dict, Optional

from .base import FormatConverter, register_format

__all__ = ['ResponseConverter']


@register_format
class ResponseConverter(FormatConverter):
    """Flat ``query``/``response`` rows, with optional multi-turn ``history`` and preference pair."""

    format_name = 'response'
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
        'answer_key': 'response',
        'answers': 'response',
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
        # Detection runs on alias-normalised names, so a column spelled `output`/`answer`/`text` is
        # recognised as `response` -- an alias never hides the format.
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
        rejected = self.pick_response(row.pop('rejected_response', None))
        if rejected is not None:
            row['rejected_response'] = rejected
        return row

    def pick_response(self, response: Any) -> Any:
        """Collapse a multi-answer response column to a single answer.

        Some datasets store several reference answers in a list. Default to the first for
        reproducibility; ``RANDOM_DATASET_RESPONSE`` picks one at random, for when the extra answers are
        augmentation rather than alternatives. The flag is read through :mod:`swift.dev.utils.env`, which
        owns env-var access; the import is local because ``swift.dev.utils`` pulls in torch on init.
        """
        if not isinstance(response, (list, tuple)):
            return response
        if not response:
            return None
        from swift.dev.utils.env import random_dataset_response
        if random_dataset_response() and self.random_state is not None:
            return response[int(self.random_state.randint(len(response)))]
        return response[0]
