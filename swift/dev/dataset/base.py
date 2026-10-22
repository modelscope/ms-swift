# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from datasets import Dataset as HfDataset
from torch.utils.data import Dataset

from swift.dev.utils import get_logger
from ._diagnostics import note_row_error
from .preprocessor import MeasurePreprocessor

if TYPE_CHECKING:
    from swift.template import Template

__all__ = ["SwiftDataset"]

logger = get_logger()


class SwiftDataset(Dataset):
    """Standard rows in, collator-ready rows out; encodes in ``__getitem__``.

    A subclass that encodes earlier overrides ``encode_row`` and ``_measure`` **together** -- the
    old arrangement paired an eager pass with a lazy wrapper in two separate places and raised
    ``KeyError: 'messages'`` one way, ``TypeError`` the other. Measuring is decided by whoever asks
    for :attr:`lengths`: a packing run pays exactly one pass, one that measures nothing pays none.
    """

    def __init__(
        self,
        dataset: HfDataset,
        template: "Template",
        *,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        batch_size: Optional[int] = None,
        strict: bool = False,
        traceback_limit: int = 10,
    ) -> None:
        self.dataset = dataset
        self.template = template
        self.num_proc = num_proc
        self.load_from_cache_file = load_from_cache_file
        self.batch_size = batch_size
        self.strict = strict
        self.traceback_limit = traceback_limit
        self._traceback_counter = 0
        self._lengths: Optional[List[int]] = None

    @property
    def lengths(self) -> List[int]:
        """Encoded token count per row, measured on first access and kept.

        One entry per row at the same index, so a consumer reads ``lengths[i]`` for the row it will
        later ask for; a row that cannot be encoded reads 0, and a planner leaves those out rather
        than putting a number in a plan the pack will not contain.
        """
        if self._lengths is None:
            self._lengths = self._measure()
        return self._lengths

    def _measure(self) -> List[int]:
        """Encode every row once to record its token count, keeping the rows as they are.

        A subclass whose rows are already encoded overrides this and :meth:`encode_row` together --
        this one to read the stored count, that one to stop re-encoding.
        """
        # MeasurePreprocessor's constructor takes only template/seed, so the budget knob is set
        # after construction; `strict` goes through the call, which overrides the instance default.
        preprocessor = MeasurePreprocessor(self.template)
        preprocessor.traceback_limit = self.traceback_limit
        measured = preprocessor(
            self.dataset,
            num_proc=self.num_proc,
            load_from_cache_file=self.load_from_cache_file,
            batch_size=self.batch_size,
            strict=self.strict,
        )
        # A row the template splits into several sequences carries one length per sequence; the row
        # travels as a unit, so what a consumer wants is the total.
        lengths = [sum(length) if isinstance(length, (list, tuple)) else length for length in measured["lengths"]]
        unusable = sum(1 for length in lengths if not length)
        if unusable:
            logger.info(
                f"{unusable} of {len(lengths)} rows could not be encoded and are marked unusable. "
                "Raise `max_length` or change `truncation_strategy` to keep them."
            )
        return lengths

    def encode_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Turn one standard row into a collator-ready row.

        The seam a subclass overrides to encode at a different time (see :meth:`_measure` for the
        pairing rule); raising is how a row is reported unusable.
        """
        return self.template.encode(row, return_length=True)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one collator-ready row, letting a row that cannot be encoded fail.

        Nothing here substitutes another row: the read goes through twinkle's ``MapRetryDataset``, which
        stands in a replacement drawn from ``(seed, index)`` -- reproducible across ranks and resume,
        independent of how many failures came before. The failure is logged before it propagates
        (except a merely too-long row); a string subscript is not a column read.
        """
        try:
            return self.encode_row(self.dataset[idx])
        except Exception as e:
            from swift.dev.template import MaxLengthError

            if not isinstance(e, MaxLengthError):
                note_row_error(
                    self,
                    "👆👆👆There are errors in the template.encode, so this row is "
                    "unusable and the reader will serve another in its place.",
                )
            raise

    def __len__(self) -> int:
        return len(self.dataset)
