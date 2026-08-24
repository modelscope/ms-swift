# Copyright (c) ModelScope Contributors. All rights reserved.
"""Encoding rows at access time instead of up front."""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from torch.utils.data import Dataset

from swift.utils import get_logger

__all__ = ['LazyLLMDataset']

logger = get_logger()


class LazyLLMDataset(Dataset):
    """A dataset that runs ``template.encode`` in ``__getitem__``, substituting a row that fails.

    Two reasons to encode late rather than in a preprocessing pass:

    - Encoding the whole dataset up front costs a full pass and stores every ``input_ids`` before the
      first training step. For a large dataset that is the dominant startup cost, and its output is
      tied to one tokenizer / template / ``max_length`` -- so it cannot be reused.
    - Encoding is where truncation happens, and truncation is where rows get rejected. Rejecting late
      means a bad row costs one substitution, not a re-run of the pass.

    The substitution is what makes this usable in training: a fixed number of steps has already been
    computed from ``len(dataset)``, so an item that cannot be produced cannot simply be skipped --
    something has to come back. A failed index draws a replacement from a fixed random permutation,
    which keeps the substitute unrelated to the position that failed (a neighbouring index would tend
    to fail for the same reason, e.g. a long document split across consecutive rows).
    """

    def __init__(self,
                 dataset: HfDataset,
                 encode_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                 *,
                 n_try_fetch: int = 10,
                 strict: bool = False,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 traceback_limit: int = 10) -> None:
        self.dataset = dataset
        self.encode_func = encode_func

        # `strict` means the first failure is the answer, so there is nothing to retry.
        n_try_fetch = 1 if strict else min(n_try_fetch, len(self.dataset))
        assert n_try_fetch >= 1
        self.strict = strict
        self.n_try_fetch = n_try_fetch

        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

        self.traceback_limit = traceback_limit
        self._traceback_counter = 0
        # A cursor over a fixed permutation, rather than a fresh draw per failure: successive failures
        # then walk distinct rows instead of possibly re-drawing the same bad one.
        self._idx = 0
        self._idx_list = self.random_state.permutation(len(self.dataset)).tolist()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # A string index is a column read (`dataset['lengths']`), not a row lookup, so it must reach
        # the underlying dataset unencoded.
        if isinstance(idx, str):
            return self.dataset[idx]
        for i in range(self.n_try_fetch):
            if i > 0:
                idx = self._idx_list[self._idx]
                self._idx = (self._idx + 1) % len(self.dataset)
            data = self.dataset[idx]
            try:
                return self.encode_func(data, return_length=True)
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
