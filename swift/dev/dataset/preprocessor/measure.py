# Copyright (c) ModelScope Contributors. All rights reserved.
"""An encoding pass kept only for the token count it measures."""
from __future__ import annotations
from typing import Any, Dict, Optional

from swift.utils import get_logger
from .encode import EncodePreprocessor

__all__ = ['MeasurePreprocessor']

logger = get_logger()


class MeasurePreprocessor(EncodePreprocessor):
    """Encode each row to learn its token count, then keep the row and throw the encoding away.

    The count cannot be had any other way: it is the *encoded* token count, so estimating it from the
    text would mean guessing at what the template adds and how the tokenizer splits. Something needs it
    before the first step -- :class:`~swift.dev.dataset.PackingDataset` plans its groups from lengths,
    and length-grouped sampling sorts by them -- which is why a whole pass is worth it.

    Differs from legacy's ``AddLengthPreprocessor``, which this replaces, in one deliberate way: a row
    that cannot be encoded is marked with an empty ``lengths`` instead of being dropped. Two things
    follow, and both are why :class:`~swift.dev.dataset.SwiftDataset` uses this one:

    - ``len(dataset)`` no longer depends on whether the measuring pass has run, so the pass can be
      deferred to the first caller that actually needs a length without the row count moving underneath
      anyone.
    - The lengths line up one-to-one with the indices a consumer will ask for, so a consumer reads
      ``lengths[i]`` for row ``i`` rather than for the ``i``-th surviving row.

    An unusable row still cannot be served, but that is already handled where it belongs:
    ``SwiftDataset.__getitem__`` substitutes a different row for one that fails to encode. Marking the
    row here is what lets a consumer know to leave it out of a plan; dropping it here would only move the
    same decision earlier and take the row count with it.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            row['lengths'] = super().preprocess(row)['lengths']
        except Exception as e:
            if self.strict:
                raise
            self._note_unusable(e)
            # Empty rather than 0: the column stays `List[int]` whichever branch runs, and every
            # consumer already sums a list length, so an unmeasurable row contributes nothing.
            row['lengths'] = []
        return row

    def _note_unusable(self, e: BaseException) -> None:
        """Log a row that could not be measured, unless being too long is the whole reason."""
        from swift.template import MaxLengthError
        # Too long is an expected property of public data and the intended outcome of
        # `truncation_strategy='delete'`, so it neither warrants a traceback nor should it spend the
        # budget that real data errors need.
        if isinstance(e, MaxLengthError):
            return
        if self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
            import traceback
            logger.info(traceback.format_exc())
            logger.warning('👆👆👆There are errors in the template.encode; this row is marked '
                           'unmeasurable and will be substituted when indexed.')
            self._traceback_counter += 1
