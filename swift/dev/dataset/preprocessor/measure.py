# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import Any, Dict, Optional

from swift.dev.utils import get_logger
from .._diagnostics import note_row_error
from .encode import EncodePreprocessor

__all__ = ['MeasurePreprocessor']

logger = get_logger()


class MeasurePreprocessor(EncodePreprocessor):
    """Encode each row to learn its token count, keep the row, throw the encoding away.

    The encoded count cannot be estimated from text, and packing and length-grouped sampling need it
    before the first step. Unlike legacy's ``AddLengthPreprocessor``, a row that cannot be encoded is
    marked with empty ``lengths`` rather than dropped, so ``len(dataset)`` does not move when the pass
    runs and lengths line up one-to-one with queried indices; substitution lives in ``SwiftDataset``.
    """
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            row['lengths'] = super().preprocess(row)['lengths']
        except Exception as e:
            if self.strict:
                raise
            self._note_unusable(e)
            # Empty rather than 0: the column stays List[int] whichever branch runs, and consumers
            # sum a list length, so an unmeasurable row contributes nothing.
            row['lengths'] = []
        return row

    def _note_unusable(self, e: BaseException) -> None:
        """Log a row that could not be measured, unless being too long is the whole reason."""
        from swift.template import MaxLengthError
        # Too long is expected public data and the intended outcome of `truncation_strategy='delete'`,
        # so it spends no traceback budget that real data errors need.
        if isinstance(e, MaxLengthError):
            return
        note_row_error(self, '👆👆👆There are errors in the template.encode; this row is marked '
                       'unmeasurable and will be substituted when indexed.')
