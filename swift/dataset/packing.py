# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import multiprocessing as mp
import torch.distributed as dist
from itertools import chain
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
from typing import Optional

from swift.template import MaxLengthError
from swift.utils import get_logger, is_dist, is_master, split_list

logger = get_logger()


def _resolve_mp_context(multiprocessing_context: Optional[str] = None):
    """Decide the multiprocessing context used by packing workers.

    When ``multiprocessing_context`` is unset, keep the historical ``fork`` behavior in the common case,
    and only switch to ``spawn`` when fork is actually unsafe (CUDA or torch.distributed already
    initialized), which otherwise dead-locks the forked worker.
    """
    if multiprocessing_context is not None:
        return mp.get_context(multiprocessing_context)
    try:
        import torch
        cuda_initialized = torch.cuda.is_initialized()
    except Exception:
        cuda_initialized = False
    if cuda_initialized or (dist.is_available() and dist.is_initialized()):
        return mp.get_context('spawn')
    # keep the historical default (fork on Linux); use the default context object.
    return mp.get_context()


def _is_fork_context(ctx) -> bool:
    return getattr(ctx, '_name', None) == 'fork'


def _get_fork_context():
    """Return the fork context, or the platform default if fork is unavailable (e.g. Windows)."""
    try:
        return mp.get_context('fork')
    except ValueError:
        return mp.get_context()


def _spawn_workers(ctx, *, target, jobs):
    """Start packing workers and return ``(context, workers)``.

    ``jobs`` is a list of ``args`` tuples (one per worker). All workers share the same context and
    the caller's queues must be created from that same context. If a non-fork context fails to start
    the first worker (e.g. an un-picklable template under ``spawn``), the returned context is ``None``
    to signal the caller to recreate its queues with fork and retry.
    """
    workers = []
    for idx, args in enumerate(jobs):
        try:
            worker = ctx.Process(target=target, daemon=True, args=args)
            worker.start()
        except Exception as e:
            if idx > 0 or _is_fork_context(ctx):
                raise
            logger.warning(f'Failed to start packing worker with multiprocessing context '
                           f'{getattr(ctx, "_name", ctx)!r} ({e}); falling back to fork.')
            return None, workers
        workers.append(worker)
    return ctx, workers


def calculate_matched_group(sequences, packing_length: int, is_finished: bool = True, strategy: str = 'binpack'):
    if len(sequences) == 0:
        return [], []
    if strategy == 'sequential':
        # Order-preserving greedy packing (next-fit): keep a single open pack and flush it
        # when the next sample doesn't fit, so the global sample order and pack boundaries
        # follow the input order (a sequential sampler). (Use packing_num_proc=1 for
        # a single global ordering.)
        packs, cur, cur_len = [], [], 0
        for item in sequences:  # item = (idx, length); weight_pos=1 -> length at item[1]
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
    # default: best-fit-decreasing bin packing (https://arxiv.org/pdf/2404.10830)
    import binpacking
    sequences = binpacking.to_constant_volume(sequences, packing_length, weight_pos=1)
    if sequences and not is_finished:
        sequences, ret_sequences = sequences[:-1], sequences[-1]
    else:
        ret_sequences = []
    return sequences, ret_sequences


class PackingDataset(Dataset):
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
        packing_strategy: str = 'binpack',
        multiprocessing_context: Optional[str] = None,
        **kwargs,
    ):
        template.packing = True
        template.padding_free = True  # TODO: remove
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.strict = strict
        self.load_from_cache_file = load_from_cache_file
        self.packing_strategy = packing_strategy
        self.multiprocessing_context = multiprocessing_context
        self.packing_length = packing_length or self.template.max_length
        self.packing_num_proc = min(packing_num_proc, math.ceil(len(dataset) / self.PACKING_BATCH_SIZE))
        if is_master():
            lengths = self.dataset['lengths']
            chunked_lengths = split_list(lengths, self.packing_num_proc)
            offsets = []
            offset = 0
            for chunk in chunked_lengths:
                offsets.append(offset)
                offset += len(chunk)
            jobs = [(i, offsets[i], chunked_lengths[i]) for i in range(self.packing_num_proc)]

            ctx = _resolve_mp_context(self.multiprocessing_context)
            self._out_queue = ctx.Queue()
            new_ctx, _ = _spawn_workers(ctx, target=self.create_packed_idx, jobs=jobs)
            if new_ctx is None:
                # non-fork context could not start workers: rebuild queue with fork and retry.
                ctx = _get_fork_context()
                self._out_queue = ctx.Queue()
                _spawn_workers(ctx, target=self.create_packed_idx, jobs=jobs)
            self.packed_idx = [[] for _ in range(self.packing_num_proc)]
            self.packed_length = [[] for _ in range(self.packing_num_proc)]
            desc = 'Packing: ' if self.packing_num_proc == 1 else f'Packing (num_proc={self.packing_num_proc}): '
            with tqdm(total=len(lengths), dynamic_ncols=True, desc=desc) as prog_bar:
                finished_workers = 0
                while finished_workers < self.packing_num_proc:
                    rank, sequences, data_len = self._out_queue.get()
                    if data_len == -1:
                        finished_workers += 1
                        continue
                    prog_bar.update(data_len)
                    self.packed_idx[rank] += [[x[0] for x in seq] for seq in sequences]
                    self.packed_length[rank] += [sum(x[1] for x in seq) for seq in sequences]
            self.packed_idx = list(chain.from_iterable(self.packed_idx))
            self.packed_length = list(chain.from_iterable(self.packed_length))
        else:
            self.packed_idx, self.packed_length = None, None
        if dist.is_initialized() and is_dist():
            obj_list = [(self.packed_idx, self.packed_length)]
            dist.broadcast_object_list(obj_list)
            self.packed_idx, self.packed_length = obj_list[0]

    def create_packed_idx(self, rank, offset, lengths):
        data = [(i + offset, sum(length) if isinstance(length, list) else length) for i, length in enumerate(lengths)]
        i = 0
        input_data = []
        while True:
            new_data = data[i:i + self.PACKING_BATCH_SIZE]
            input_data += new_data
            if not input_data:
                break
            i += self.PACKING_BATCH_SIZE
            is_finished = i >= len(data)
            sequences, input_data = calculate_matched_group(
                input_data, self.packing_length, is_finished=is_finished, strategy=self.packing_strategy)
            self._out_queue.put((rank, sequences, len(new_data)))
        self._out_queue.put((rank, [], -1))

    def __getitem__(self, index):
        sequence = self.packed_idx[index]
        row = [self.dataset[i] for i in sequence]
        return row

    def __len__(self):
        return len(self.packed_idx)


class IterablePackingDataset(IterableDataset):

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
        packing_strategy: str = 'binpack',
        multiprocessing_context: Optional[str] = None,
        **kwargs,
    ):
        template.packing = True
        template.padding_free = True  # TODO: remove
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.strict = strict
        self.multiprocessing_context = multiprocessing_context
        self.packing_length = packing_length or self.template.max_length

        self.packing_interval = packing_interval
        self.cyclic = cyclic
        self.packing_strategy = packing_strategy
        self.workers = []

        ctx = _resolve_mp_context(self.multiprocessing_context)
        self._in_queue = ctx.Queue()
        self._out_queue = ctx.Queue()
        new_ctx, workers = _spawn_workers(ctx, target=self._processor, jobs=[()] * self.num_proc)
        if new_ctx is None:
            # non-fork context could not start workers: rebuild queues with fork and retry.
            ctx = _get_fork_context()
            self._in_queue = ctx.Queue()
            self._out_queue = ctx.Queue()
            _, workers = _spawn_workers(ctx, target=self._processor, jobs=[()] * self.num_proc)
        self.workers = workers

    def _processor(self):
        while True:
            i, data = self._in_queue.get()
            encoded_data = {}
            try:
                encoded_data = self.template.encode(data, return_length=True)
            except Exception as e:
                if self.strict and not isinstance(e, MaxLengthError):
                    raise
            self._out_queue.put((i, encoded_data))

    def _put_data_in_queue(self, iterator) -> int:
        for i in range(self.packing_interval):
            try:
                data = next(iterator)
            except StopIteration:
                return i
            self._in_queue.put((i, data))
        return i + 1

    def _fetch_data_out_queue(self, last_res, num_samples):
        res = [None] * num_samples
        for _ in range(num_samples):
            i, data = self._out_queue.get()
            if not data:
                continue
            res[i] = data if isinstance(data, list) else [data]
        res = [(item, len(item['input_ids'])) for group in res if group for item in group]
        last_res += res
        return last_res

    @staticmethod
    def cyclic_iter(iterable):
        while True:
            for x in iterable:
                yield x

    def set_epoch(self, epoch: int):
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)

    def __iter__(self):
        try:
            next(iter(self.dataset))
        except StopIteration:
            return

        if self.cyclic:
            iterator = self.cyclic_iter(self.dataset)
        else:
            iterator = iter(self.dataset)
        data = []
        while True:
            num_samples = self._put_data_in_queue(iterator)
            finished = num_samples != self.packing_interval
            data = self._fetch_data_out_queue(data, num_samples)
            sequences, data = calculate_matched_group(
                data, self.packing_length, is_finished=finished, strategy=self.packing_strategy)
            res = []
            for row in sequences:
                res.append([r[0] for r in row])
            yield from res
            if finished:
                break
