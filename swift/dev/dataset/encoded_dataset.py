# Copyright (c) ModelScope Contributors. All rights reserved.
"""A dataset whose rows were encoded by an earlier pass, not on access."""
from __future__ import annotations
from typing import Any, Dict, List

from swift.utils import get_logger
from .swift_dataset import SwiftDataset

__all__ = ['EncodedDataset']

logger = get_logger()


class EncodedDataset(SwiftDataset):
    """Rows that already hold ``input_ids``, so there is nothing left to do when one is indexed.

    The case that needs this is ``truncation_strategy='split'``, where one long text becomes several
    samples: the row count itself changes, so the encoding has to be materialised before anything can
    count steps or plan packs. Loading a dataset that was stored already-encoded lands here too.

    All of it is the two overrides below, and that is the point of putting the encode/measure pair on the
    base class as methods rather than deciding between them from outside. The arrangement this replaces
    expressed the same difference twice -- a conditional preprocessor choice in one place
    (``sft.py:323``) and a conditional wrapper in another (``sft.py:136``) -- two expressions of one
    decision that had to be kept in agreement by hand. Pairing them wrongly raised
    ``KeyError: 'messages'`` one way and ``TypeError`` the other.

    Note what is *not* here: no substitution, and no permutation of every index built to drive it. A row
    that could not be encoded is not in this dataset -- the pass that built it already resolved that --
    so :meth:`encode_row` cannot raise and the base class never reaches that branch.
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

        Overridden together with :meth:`encode_row`, which is the pairing the class docstring describes:
        the base implementation would hand an already-encoded row to ``template.encode`` and fail on the
        ``messages`` it no longer has.
        """
        return [sum(length) if isinstance(length, (list, tuple)) else length for length in self.dataset['lengths']]
