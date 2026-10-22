# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import math
import multiprocessing as mp
from itertools import chain
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch.distributed as dist
from torch.utils.data import IterableDataset
from tqdm import tqdm

from swift.dev.utils import get_logger, is_dist, is_master, split_list
from .base import SwiftDataset

__all__ = ["IterablePackingDataset", "PackingDataset"]

logger = get_logger()

# One sample as packing sees it: its index (or the encoded row itself, when streaming) and its length.
Weighted = Tuple[Any, int]


class PackingDataset(SwiftDataset):
    """A map-style dataset whose items are *groups* of samples that together fill ``packing_length``.

    Planning happens once in the constructor, on the master rank and broadcast: bin packing is
    deterministic in principle but its cost is real, and every rank must agree on the grouping or
    they would disagree on the number of steps. ``__getitem__`` returns the rows of one group;
    concatenation is the collator's (the template builds the block-diagonal attention mask).
    """

    # Rows per bin-packing call. Packing a whole dataset in one call would hold every length in one
    # list and emit nothing until the end; this bounds both, and lets the progress bar move.
    PACKING_BATCH_SIZE = 1000

    def __init__(
        self,
        template,
        dataset,
        num_proc: int = 1,
        *,
        strict: bool = False,
        load_from_cache_file: bool = True,
        packing_length: Optional[int] = None,
        packing_num_proc: int = 1,
        packing_strategy: str = "binpack",
        **kwargs,
    ):
        # The template has to know it is packing: it builds the attention mask that keeps the packed
        # samples from attending to each other.
        template.packing = True
        template.padding_free = True  # TODO: remove
        super().__init__(
            dataset, template, num_proc=num_proc, load_from_cache_file=load_from_cache_file, strict=strict, **kwargs
        )
        self.packing_strategy = packing_strategy
        self.packing_length = packing_length or self.template.max_length
        # More workers than batches would leave workers with nothing to do, and `split_list` would
        # hand them empty chunks.
        self.packing_num_proc = min(packing_num_proc, math.ceil(len(dataset) / self.PACKING_BATCH_SIZE))
        self._out_queue = mp.Queue()

        if is_master():
            # Asking for `lengths` is what triggers the measuring pass -- packing is the caller that
            # needs it, so packing is where it gets paid for.
            self.packed_idx, self.packed_length = self.plan_packing(self.lengths)
        else:
            self.packed_idx, self.packed_length = None, None
        if dist.is_initialized() and is_dist():
            obj_list = [(self.packed_idx, self.packed_length)]
            dist.broadcast_object_list(obj_list)
            self.packed_idx, self.packed_length = obj_list[0]

    def plan_packing(self, lengths: List[Any]) -> Tuple[List[List[int]], List[int]]:
        """Group every row index into packs, returning ``(packs, pack_lengths)``.

        Workers each pack a contiguous slice of the dataset and report groups back through a queue as
        they finish them, so the progress bar reflects real progress rather than a final dump.
        """
        offset = 0
        chunked_lengths = split_list(lengths, self.packing_num_proc)
        for rank in range(self.packing_num_proc):
            worker = mp.Process(target=self.create_packed_idx, args=(rank, offset, chunked_lengths[rank]), daemon=True)
            worker.start()
            offset += len(chunked_lengths[rank])

        packed_idx: List[List[List[int]]] = [[] for _ in range(self.packing_num_proc)]
        packed_length: List[List[int]] = [[] for _ in range(self.packing_num_proc)]
        desc = "Packing: " if self.packing_num_proc == 1 else f"Packing (num_proc={self.packing_num_proc}): "
        with tqdm(total=len(lengths), dynamic_ncols=True, desc=desc) as prog_bar:
            finished_workers = 0
            while finished_workers < self.packing_num_proc:
                rank, sequences, data_len = self._out_queue.get()
                # `-1` is the worker's end-of-stream marker, not a batch of zero rows.
                if data_len == -1:
                    finished_workers += 1
                    continue
                prog_bar.update(data_len)
                packed_idx[rank] += [[x[0] for x in seq] for seq in sequences]
                packed_length[rank] += [sum(x[1] for x in seq) for seq in sequences]
        # Concatenate per rank in rank order: the workers took contiguous slices, so this keeps the
        # packs in dataset order.
        return list(chain.from_iterable(packed_idx)), list(chain.from_iterable(packed_length))

    def create_packed_idx(self, rank: int, offset: int, lengths: List[Any]) -> None:
        """Pack one worker's slice, putting each batch's groups on the queue as they are formed."""
        # A list length belongs to a row the template splits into several sequences; the whole row
        # still travels together, so its lengths sum. A row that reads 0 could not be encoded at all.
        data = [
            (i + offset, sum(length) if isinstance(length, list) else length)
            for i, length in enumerate(lengths)
            if (sum(length) if isinstance(length, list) else length)
        ]
        i = 0
        input_data: List[Weighted] = []
        while True:
            new_data = data[i : i + self.PACKING_BATCH_SIZE]
            input_data += new_data
            if not input_data:
                break
            i += self.PACKING_BATCH_SIZE
            is_finished = i >= len(data)
            # Whatever did not fill a pack comes back as `input_data` and joins the next batch,
            # rather than being flushed as a short pack.
            sequences, input_data = self.calculate_matched_group(
                input_data, self.packing_length, is_finished=is_finished, strategy=self.packing_strategy
            )
            self._out_queue.put((rank, sequences, len(new_data)))
        self._out_queue.put((rank, [], -1))

    @staticmethod
    def calculate_matched_group(
        sequences: Sequence[Weighted], packing_length: int, is_finished: bool = True, strategy: str = "binpack"
    ) -> Tuple[List[List[Weighted]], List[Weighted]]:
        """Group ``(item, length)`` pairs into packs of at most ``packing_length``.

        Returns ``(packs, leftover)``; a false ``is_finished`` carries the last still-fillable pack
        into the next window instead of emitting it short. ``'binpack'`` (default, best-fit-decreasing,
        https://arxiv.org/pdf/2404.10830) fills fullest but reorders freely; ``'sequential'`` (next-fit)
        is less full but order-preserving -- use it with ``packing_num_proc=1`` for one global order.
        """
        if len(sequences) == 0:
            return [], []
        if strategy == "sequential":
            packs: List[List[Weighted]] = []
            cur: List[Weighted] = []
            cur_len = 0
            for item in sequences:
                seq_len = item[1]
                if cur and cur_len + seq_len > packing_length:
                    packs.append(cur)
                    cur, cur_len = [], 0
                cur.append(item)
                cur_len += seq_len
                if cur_len >= packing_length:
                    packs.append(cur)
                    cur, cur_len = [], 0
            if is_finished:
                if cur:
                    packs.append(cur)
                return packs, []
            return packs, cur

        import binpacking

        grouped = binpacking.to_constant_volume(sequences, packing_length, weight_pos=1)
        if grouped and not is_finished:
            grouped, leftover = grouped[:-1], grouped[-1]
        else:
            leftover = []
        return grouped, leftover

    def __getitem__(self, index: int) -> List[Dict[str, Any]]:
        """Return the encoded rows of one group, encoding each through the inherited access path.

        ``super().__getitem__`` is what encodes a member, so a group is assembled from the same
        access path a non-packing dataset serves -- not from a separate lazily-encoding wrapper
        underneath this one. Concatenation stays the collator's (the template does it).
        """
        return [super(PackingDataset, self).__getitem__(i) for i in self.packed_idx[index]]

    def __len__(self) -> int:
        return len(self.packed_idx)


class IterablePackingDataset(IterableDataset):
    """A streaming dataset that packs as rows arrive.

    A stream cannot be planned: lengths are only known once a row is read and encoded. So packing
    runs over a sliding window of ``packing_interval`` samples, and encoding happens in worker
    processes here -- for a map-style dataset that is a separate preprocessing pass, but a stream
    has no such pass to put it in.
    """

    def __init__(
        self,
        template,
        dataset,
        num_proc: int = 1,
        *,
        packing_interval: int = 128,
        packing_length: Optional[int] = None,
        strict: bool = False,
        cyclic: bool = False,
        packing_strategy: str = "binpack",
        data_seed: int = 0,
        **kwargs,
    ):
        template.packing = True
        template.padding_free = True  # TODO: remove
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.strict = strict
        self.packing_length = packing_length or self.template.max_length
        self.packing_interval = packing_interval
        self.cyclic = cyclic
        self.packing_strategy = packing_strategy
        # Kept for the twinkle DataLoader streaming protocol, which calls set_data_seed on a state
        # restore and set_epoch per pass; encode itself consumes no randomness.
        self.data_seed = int(data_seed or 0)

        self._in_queue = mp.Queue()
        self._out_queue = mp.Queue()
        self.workers = []
        for _ in range(self.num_proc):
            worker = mp.Process(target=self._processor, daemon=True)
            worker.start()
            self.workers.append(worker)

    def set_epoch(self, epoch: int) -> None:
        """Forward to the underlying stream, whose shuffle buffer seeds by epoch (twinkle calls this)."""
        set_epoch = getattr(self.dataset, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(epoch)

    def set_data_seed(self, data_seed: int) -> None:
        """Store the seed twinkle hands back on a state restore."""
        self.data_seed = int(data_seed or 0)

    def _processor(self) -> None:
        """Worker loop: encode whatever arrives, and never die on one bad row.

        A row that fails to encode goes back as an empty result rather than an exception, so one bad
        row in a stream costs that row and not the run. ``strict`` re-raises -- except for a sample
        that is merely too long, which is a property of the data and not a defect to stop for.
        """
        from swift.dev.template import MaxLengthError

        while True:
            i, data = self._in_queue.get()
            encoded_data = {}
            try:
                encoded_data = self.template.encode(data, return_length=True)
            except Exception as e:
                if self.strict and not isinstance(e, MaxLengthError):
                    raise
            self._out_queue.put((i, encoded_data))

    def _put_data_in_queue(self, iterator: Iterator) -> int:
        """Hand at most ``packing_interval`` rows to the workers; returns how many were available."""
        for i in range(self.packing_interval):
            try:
                data = next(iterator)
            except StopIteration:
                return i
            self._in_queue.put((i, data))
        return i + 1

    def _fetch_data_out_queue(self, last_res: List[Weighted], num_samples: int) -> List[Weighted]:
        """Collect ``num_samples`` encoded rows, restoring input order and dropping the failures."""
        res: List[Any] = [None] * num_samples
        for _ in range(num_samples):
            i, data = self._out_queue.get()
            # Results come back in completion order; the index puts them back in input order.
            if not data:
                continue
            res[i] = data if isinstance(data, list) else [data]
        res = [(item, len(item["input_ids"])) for group in res if group for item in group]
        return last_res + res

    @staticmethod
    def cyclic_iter(iterable):
        """Repeat a re-iterable dataset forever and stop cleanly if it is empty or exhausted."""
        while True:
            emitted = False
            for x in iterable:
                emitted = True
                yield x
            if not emitted:
                return

    def __iter__(self):
        iterator = self.cyclic_iter(self.dataset) if self.cyclic else iter(self.dataset)
        data: List[Weighted] = []
        while True:
            num_samples = self._put_data_in_queue(iterator)
            # A short window means the stream ended, which is also the signal to flush the leftover.
            finished = num_samples != self.packing_interval
            data = self._fetch_data_out_queue(data, num_samples)
            sequences, data = PackingDataset.calculate_matched_group(
                data, self.packing_length, is_finished=finished, strategy=self.packing_strategy
            )
            yield from ([r[0] for r in row] for row in sequences)
            if finished:
                break
