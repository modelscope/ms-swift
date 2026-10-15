# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import Any, Callable, Dict

from datasets import Dataset as HfDataset
from torch.utils.data import Dataset

from ._diagnostics import note_row_error

__all__ = ['LazyLLMDataset']


class LazyLLMDataset(Dataset):
    """A dataset that runs ``template.encode`` in ``__getitem__``.

    Two reasons to encode late rather than in a preprocessing pass:

    - Encoding the whole dataset up front costs a full pass and stores every ``input_ids`` before the
      first training step. For a large dataset that is the dominant startup cost, and its output is
      tied to one tokenizer / template / ``max_length`` -- so it cannot be reused.
    - Encoding is where truncation happens, and truncation is where rows get rejected. Rejecting late
      means a bad row costs one substitution, not a re-run of the pass.

    The substitution is not this class's to make. The read goes through twinkle's ``RetryDataset``
    (installed by ``DataLoader._read_through``), which stands in a replacement drawn from
    ``(seed, index)`` -- reproducible across ranks and across a resume, because the draw does not depend
    on how many failures came before. What this class owes the training loop is a fixed ``len(dataset)``
    to compute the step count from, and that holds whether or not a given row encodes.
    """

    def __init__(self,
                 dataset: HfDataset,
                 encode_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                 *,
                 traceback_limit: int = 10) -> None:
        self.dataset = dataset
        self.encode_func = encode_func
        self.traceback_limit = traceback_limit
        self._traceback_counter = 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # A string index is a column read (`dataset['lengths']`), not a row lookup, so it must reach
        # the underlying dataset unencoded.
        if isinstance(idx, str):
            return self.dataset[idx]
        try:
            return self.encode_func(self.dataset[idx], return_length=True)
        except Exception as e:
            from swift.template import MaxLengthError
            # Too long is an expected property of public data, not a defect worth a traceback.
            if not isinstance(e, MaxLengthError):
                note_row_error(self, '👆👆👆There are errors in the template.encode, so this row is '
                               'unusable and the reader will serve another in its place.')
            raise

    def __len__(self) -> int:
        return len(self.dataset)
