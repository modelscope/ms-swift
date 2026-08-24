# Copyright (c) ModelScope Contributors. All rights reserved.
"""An encoding pass kept only for the length it measures."""
from __future__ import annotations
from typing import Any, Dict, Optional

from .encode_preprocessor import EncodePreprocessor

__all__ = ['AddLengthPreprocessor']


class AddLengthPreprocessor(EncodePreprocessor):
    """Encode each row to measure it, then keep the row and throw the encoding away.

    Deliberate, and the reason is reuse: ``input_ids`` are tied to one tokenizer, template and
    ``max_length``, so a cache of them is worthless to any other model. The raw text is
    model-agnostic, so a cache of text plus ``lengths`` can be reused across runs and models -- which
    is what ``cached_dataset`` stores.

    The length itself is not free to obtain any other way: it is the *encoded* token count, so it
    cannot be estimated from the text without doing the encode. It is needed before training starts,
    because :class:`PackingDataset` plans its groups from lengths, and length-grouped sampling sorts
    by them.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        encoded = super().preprocess(row)
        row['lengths'] = encoded['lengths']
        return row
