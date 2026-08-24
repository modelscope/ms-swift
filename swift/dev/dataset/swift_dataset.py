# Copyright (c) ModelScope Contributors. All rights reserved.
"""The base dataset the DataLoader consumes: encodes on access, measures on demand."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from torch.utils.data import Dataset

from swift.utils import get_logger
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
        n_try_fetch: How many substitutes to try before giving up on an index. A row that fails to
            encode cannot simply be skipped -- the number of steps is already fixed by ``len(self)``, so
            something has to come back.
        strict: Treat the first failure as the answer, in the measuring pass and on access both. Turns
            an unusable row from something worked around into an error.
        random_state: Seeds the substitution order.
        traceback_limit: How many failures to log before going quiet.
    """

    def __init__(self,
                 dataset: HfDataset,
                 template: 'Template',
                 *,
                 num_proc: int = 1,
                 load_from_cache_file: bool = True,
                 batch_size: Optional[int] = None,
                 n_try_fetch: int = 10,
                 strict: bool = False,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 traceback_limit: int = 10) -> None:
        self.dataset = dataset
        self.template = template
        self.num_proc = num_proc
        self.load_from_cache_file = load_from_cache_file
        self.batch_size = batch_size

        # `strict` means the first failure is the answer, so there is nothing to retry.
        n_try_fetch = 1 if strict else min(n_try_fetch, max(len(dataset), 1))
        assert n_try_fetch >= 1
        self.strict = strict
        self.n_try_fetch = n_try_fetch

        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

        self.traceback_limit = traceback_limit
        self._traceback_counter = 0
        # Built on the first failure rather than here: a subclass whose rows are already encoded cannot
        # fail, and a permutation of every index is a real cost to hand it for nothing.
        self._idx = 0
        self._idx_list: Optional[List[int]] = None

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

    def _next_substitute(self) -> int:
        """Pick the next replacement index for a row that could not be encoded.

        A cursor over one fixed permutation, rather than a fresh draw per failure: successive failures
        then walk distinct rows instead of possibly re-drawing the same bad one. A neighbouring index
        would be the wrong choice for the same reason -- it tends to fail for the same cause, e.g. a long
        document split across consecutive rows.
        """
        if self._idx_list is None:
            self._idx_list = self.random_state.permutation(len(self.dataset)).tolist()
        idx = self._idx_list[self._idx]
        self._idx = (self._idx + 1) % len(self.dataset)
        return idx

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one collator-ready row, substituting a different one if this row cannot be encoded.

        The substitution is what makes a failure survivable during training: the number of steps is
        already fixed by ``len(self)``, so an index that cannot be served cannot simply be skipped --
        something has to come back. A subclass whose rows are already encoded never reaches that branch,
        and pays nothing for it.

        Unlike the wrapper this replaces, a string subscript is not a column read: a consumer that wants
        lengths asks for :attr:`lengths`. The passthrough existed so that a wrapper placed on top could
        reach a column of the dataset underneath, and with lengths inherited there is nothing to reach
        through to.
        """
        for i in range(self.n_try_fetch):
            if i > 0:
                idx = self._next_substitute()
            row = self.dataset[idx]
            try:
                return self.encode_row(row)
            except Exception as e:
                from swift.template import MaxLengthError
                if self.strict:
                    logger.warning('To avoid errors, you can pass `strict=False`.')
                    raise
                # Too long is an expected property of public data, not a defect worth a traceback.
                if isinstance(e, MaxLengthError):
                    continue
                if self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                    import traceback
                    logger.info(traceback.format_exc())
                    logger.warning('👆👆👆There are errors in the template.encode, '
                                   'and another piece of data will be randomly selected.')
                    self._traceback_counter += 1

        raise ValueError('Failed to retrieve the dataset. You can avoid this issue by increasing `max_length` or '
                         'modifying the `truncation_strategy`.')

    def __len__(self) -> int:
        return len(self.dataset)
