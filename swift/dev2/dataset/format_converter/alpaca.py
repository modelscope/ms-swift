# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import Any, Collection, Dict, Optional

from .base import register_format
from .response import ResponseConverter

__all__ = ['AlpacaConverter']


@register_format
class AlpacaConverter(ResponseConverter):
    """``instruction`` + ``input`` + ``output``, where the first two join into one user turn."""

    format_name = 'alpaca'
    priority = 20
    is_fallback = False
    aliases = {
        'system_prompt': 'system',
        'output': 'response',
    }

    @classmethod
    def detect(cls, columns: Collection[str]) -> bool:
        return 'instruction' in columns and 'input' in columns

    def convert(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.apply_aliases(row)
        instruction = row.pop('instruction', None)
        input_ = row.pop('input', None)
        row['query'] = self.join_instruction_input(instruction, input_)
        return super().convert(row)

    @staticmethod
    def join_instruction_input(instruction: Optional[str], input_: Optional[str]) -> Optional[str]:
        """Join the two halves of the user turn with a newline, tolerating either being empty."""
        if instruction and input_:
            return f'{instruction}\n{input_}'
        return instruction or input_
