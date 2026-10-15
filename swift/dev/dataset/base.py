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

__all__ = ['SwiftDataset']

logger = get_logger()


class SwiftDataset(Dataset):
    """Standard rows in, collator-ready rows out -- and nothing else needs arranging around it.

    This is the base of the torch-facing layer. It replaces the arrangement where a caller decided,
    from a handful of interacting arguments, whether to run an encoding pass and then which wrapper to
    put around the result. Here the two things that decision was really about are separate and each
    belongs to something:

    - *When a row is encoded* is a property of the class. This one encodes in :meth:`__getitem__`; a
      subclass that wants it earlier overrides that.
    - *Whether the dataset gets measured* is decided by whoever needs a length, by asking for
      :attr:`lengths`. Nothing measures a dataset nobody measures.

    The second point is the one that changes behaviour rather than just shape. A measuring pass costs a
    full encode of the dataset, and the old arrangement ran it whenever eager encoding was on --
    including for a run that packed nothing and sorted by nothing, where the numbers were computed and
    then only logged. Deferring to first access means that run does no pass at all, while a run that
    packs still pays for exactly one.

    Subclasses get :attr:`lengths` by inheriting it, so :class:`PackingDataset` can plan its groups from
    ``self.lengths`` in its constructor instead of reaching into a wrapped dataset's columns -- which is
    what forced the wrapper underneath it to pass string subscripts through to its own wrapped dataset.

    Args:
        dataset: Standard rows -- ``messages`` plus whatever media and ``objects`` columns the dataset
            has. Loading and format conversion happen before this, in ``loader`` and ``preprocessor``.
        template: Encodes a row. Also decides what counts as too long, so it is what makes a row
            unusable.
        num_proc: Worker processes for the measuring pass, when there is one.
        load_from_cache_file: Reuse a previous measuring pass's result if its fingerprint matches.
        batch_size: Rows per batch in the measuring pass.
        strict: Treat the first failure as the answer, in the measuring pass and on access both. On
            access this class simply raises; whether a raised row is substituted or stops the run is
            the reader's decision (``DataLoader(max_retries=0)`` makes it stop).
        traceback_limit: How many failures to log before going quiet.
    """

    def __init__(self,
                 dataset: HfDataset,
                 template: 'Template',
                 *,
                 num_proc: int = 1,
                 load_from_cache_file: bool = True,
                 batch_size: Optional[int] = None,
                 strict: bool = False,
                 traceback_limit: int = 10) -> None:
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

        One entry per row of ``self``, at the same index, so a consumer reads ``lengths[i]`` for the row
        it will later ask for. A row that could not be encoded reads 0: it has no token count, and
        substituting some other row's count would put a number in a pack plan that the pack will not
        actually contain. A consumer that plans from these should leave the zeros out.

        The pass this triggers is the expensive part of setting up a dataset, so it happens here rather
        than in the constructor -- constructing this class does not commit anyone to paying for it.
        """
        if self._lengths is None:
            self._lengths = self._measure()
        return self._lengths

    def _measure(self) -> List[int]:
        """Encode every row once to record its token count, keeping the rows as they are.

        Reads ``self.dataset`` as standard rows, which is the same assumption :meth:`encode_row` makes.
        A subclass whose rows are already encoded has to override **both** -- this one to read the
        stored count instead of encoding, that one to stop encoding what is already encoded. Overriding
        only one is what the arrangement this replaces got wrong: pairing an eager encoding pass with a
        lazy wrapper raised ``KeyError: 'messages'``, and pairing them the other way raised
        ``TypeError``.
        """
        preprocessor = MeasurePreprocessor(self.template)
        preprocessor.strict = self.strict
        preprocessor.traceback_limit = self.traceback_limit
        measured = preprocessor(
            self.dataset,
            num_proc=self.num_proc,
            load_from_cache_file=self.load_from_cache_file,
            batch_size=self.batch_size,
            strict=self.strict)
        # A row the template splits into several sequences carries one length per sequence; the row
        # travels as a unit, so what a consumer wants is the total.
        lengths = [sum(length) if isinstance(length, (list, tuple)) else length for length in measured['lengths']]
        unusable = sum(1 for length in lengths if not length)
        if unusable:
            logger.info(f'{unusable} of {len(lengths)} rows could not be encoded and are marked unusable. '
                        'Raise `max_length` or change `truncation_strategy` to keep them.')
        return lengths

    def encode_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Turn one standard row into a collator-ready row.

        The seam a subclass overrides to encode at a different time -- see :meth:`_measure` for why the
        two have to be overridden together. Raising is how a row is reported unusable;
        :meth:`__getitem__` is arranged around that.
        """
        return self.template.encode(row, return_length=True)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one collator-ready row, letting a row that cannot be encoded fail.

        Nothing here substitutes another row. The read goes through twinkle's ``RetryDataset`` (installed
        by ``DataLoader._read_through``), which stands in a replacement drawn from ``(seed, index)`` --
        reproducible across ranks and across a resume, because the draw does not depend on how many
        failures came before. There is only the one mechanism, so the reader owns it.

        The failure is logged before it propagates: a run that does substitute still says which row broke
        and why, while ``RetryDataset`` reports the replacement it chose. A row that is merely too long
        is skipped -- that is an expected property of public data, not a defect worth a traceback.

        Unlike the wrapper this replaces, a string subscript is not a column read: a consumer that wants
        lengths asks for :attr:`lengths`. The passthrough existed so that a wrapper placed on top could
        reach a column of the dataset underneath, and with lengths inherited there is nothing to reach
        through to.
        """
        try:
            return self.encode_row(self.dataset[idx])
        except Exception as e:
            from swift.template import MaxLengthError
            if not isinstance(e, MaxLengthError):
                note_row_error(self, '👆👆👆There are errors in the template.encode, so this row is '
                               'unusable and the reader will serve another in its place.')
            raise

    def __len__(self) -> int:
        return len(self.dataset)
