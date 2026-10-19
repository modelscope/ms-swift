# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import Any, Dict, List

from .base import SwiftDataset

__all__ = ['EncodedDataset']


class EncodedDataset(SwiftDataset):
    """Rows that already hold ``input_ids``: nothing left to do when one is indexed.

    Needed where the row count itself depends on encoding (``truncation_strategy='split'``) or the
    dataset was stored already-encoded. The old arrangement expressed this difference twice
    (conditional preprocessor choice + conditional wrapper) and drifted into ``KeyError``/
    ``TypeError`` when paired wrongly. No substitution: the building pass resolved it already.
    """

    def __init__(self, dataset, template, **kwargs) -> None:
        super().__init__(dataset, template, **kwargs)
        missing = [name for name in ('input_ids', 'lengths') if name not in dataset.features]
        if missing:
            raise ValueError(f'{type(self).__name__} expects rows an earlier pass already encoded, but this '
                             f'dataset has no {missing} column. Its columns are {list(dataset.features)}. '
                             'Use `SwiftDataset` for standard rows -- it encodes them as they are indexed.')

    def encode_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Return the row as it is: it already holds what encoding would have produced."""
        return row

    def _measure(self) -> List[int]:
        """Read the counts the encoding pass recorded, rather than encoding everything again.

        The base would hand an already-encoded row to ``template.encode`` and fail on the
        ``messages`` it no longer has -- the pairing the class docstring describes.
        """
        return [sum(length) if isinstance(length, (list, tuple)) else length for length in self.dataset['lengths']]
