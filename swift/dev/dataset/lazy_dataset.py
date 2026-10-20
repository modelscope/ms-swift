# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import Any, Callable, Dict

from datasets import Dataset as HfDataset
from torch.utils.data import Dataset

from ._diagnostics import note_row_error

__all__ = ['LazyLLMDataset']


class LazyLLMDataset(Dataset):
    """A dataset that runs ``template.encode`` in ``__getitem__``.

    Encoding late avoids a startup pass tied to one tokenizer/template/``max_length``, and rejects
    a too-long row at access time -- one substitution, not a pass re-run. Substitution is not this
    class's: the read goes through twinkle's ``MapRetryDataset``, which stands in a replacement drawn
    from ``(seed, index)``; what it owes the loop is a fixed ``len(dataset)``.
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
