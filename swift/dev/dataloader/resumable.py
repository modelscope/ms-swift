# Copyright (c) ModelScope Contributors. All rights reserved.
from torch.utils.data import DataLoader
from typing import Iterator, Optional


class ResumableDataLoaderWrapper:
    """DataLoader wrapper providing epoch-aware, observable resume state.

    Resume model (map-style only; iterable datasets are unsupported):
    - The wrapper tracks consumed BATCHES (exact; the trailing partial batch would make a
      samples//batch_size decomposition lossy). `consumed_samples` is derived for the
      observable state (== consumed_batches * batch_size).
    - On resume, consumed_batches is decomposed into (resume_epoch, resume_offset) using
      batches_per_epoch (== len(dataloader) == BatchSamplerShard.__len__, honoring drop_last).
    - The SFTLoop drives the epoch loop and calls set_epoch(epoch) each epoch; the wrapper's
      __iter__ skips `resume_offset` batches ONLY on the resume epoch, so the resumed stream
      reproduces the reference order from exactly the checkpoint position.
    """

    def __init__(self, dataloader: DataLoader, consumed_samples: int = 0):
        self.dataloader = dataloader
        self._batch_size = getattr(dataloader, 'batch_size', None)
        if self._batch_size is None:
            batch_sampler = getattr(dataloader, 'batch_sampler', None)
            self._batch_size = getattr(batch_sampler, 'batch_size', 1) if batch_sampler else 1
        try:
            self._batches_per_epoch: Optional[int] = len(dataloader)
        except TypeError:
            self._batches_per_epoch = None  # iterable: unsupported for deterministic resume

        # Consumed batches so far (exact resume granularity). Seed from consumed_samples.
        self._consumed_batches = int(consumed_samples) // max(1, self._batch_size)
        self._epoch = 0
        # Decompose the resume position into (epoch, offset-within-epoch).
        if self._batches_per_epoch:
            self._resume_epoch = self._consumed_batches // self._batches_per_epoch
            self._resume_offset = self._consumed_batches % self._batches_per_epoch
        else:
            self._resume_epoch, self._resume_offset = 0, self._consumed_batches

    def __iter__(self) -> Iterator:
        # SFTLoop sets the epoch before iterating; honor it. Skip the resume offset only on the
        # exact resume epoch (later epochs start from the top).
        skip = self._resume_offset if self._epoch == self._resume_epoch else 0
        for i, batch in enumerate(self.dataloader):
            if i < skip:
                continue
            self._consumed_batches += 1
            yield batch

    def __len__(self) -> int:
        return len(self.dataloader)

    def get_state(self) -> dict:
        return {
            'consumed_samples': self.consumed_samples,
            'consumed_batches': self._consumed_batches,
            'epoch': self._epoch,
        }

    def skip_consumed_samples(self, n: int) -> None:
        self._consumed_batches = max(int(n), 0) // max(1, self._batch_size)
        if self._batches_per_epoch:
            self._resume_epoch = self._consumed_batches // self._batches_per_epoch
            self._resume_offset = self._consumed_batches % self._batches_per_epoch
        else:
            self._resume_epoch, self._resume_offset = 0, self._consumed_batches

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        if hasattr(self.dataloader, 'set_epoch'):
            self.dataloader.set_epoch(epoch)

    @property
    def consumed_samples(self) -> int:
        return self._consumed_batches * self._batch_size

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def dataset(self):
        return self.dataloader.dataset

    @property
    def batch_sampler(self):
        return getattr(self.dataloader, 'batch_sampler', None)
