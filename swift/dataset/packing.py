# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import multiprocessing as mp
import torch.distributed as dist
from itertools import chain
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
from typing import Optional

from swift.template import MaxLengthError
from swift.utils import get_external_files, get_logger, import_external_file, is_dist, is_master, split_list

logger = get_logger()


def _resolve_mp_context(multiprocessing_context: Optional[str] = None):
    """Decide the multiprocessing context used by packing workers.

    When ``multiprocessing_context`` is unset, keep the platform default (``fork`` on Python <=3.13,
    ``forkserver`` on 3.14+), and only switch to ``spawn`` when fork is actually unsafe (CUDA or
    torch.distributed already initialized), which otherwise dead-locks the forked worker.
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
    # keep the platform default (fork on Python <=3.13, forkserver on 3.14+); use the default context object.
    return mp.get_context()


def _spawn_workers(ctx, *, target, jobs):
    """Start packing workers and return them.

    ``jobs`` is a list of ``args`` tuples (one per worker); all workers share ``ctx`` and the caller's
    queues must be created from that same context. A failure to start a worker (e.g. an un-picklable
    arg under ``spawn``) propagates: we deliberately do NOT fall back to fork, because a forked child
    that inherited an active torch threadpool / CUDA state can dead-lock silently -- a loud error is
    strictly better than a hang.
    """
    workers = []
    for args in jobs:
        worker = ctx.Process(target=target, daemon=True, args=args)
        worker.start()
        workers.append(worker)
    return workers


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
            jobs = [(i, offsets[i], chunked_lengths[i], self.packing_length, self.packing_strategy)
                    for i in range(self.packing_num_proc)]

            ctx = _resolve_mp_context(self.multiprocessing_context)
            self._out_queue = ctx.Queue()
            # Pass the out-queue/params as explicit args (not via `self`) so a non-fork context does not
            # need to pickle the whole dataset held by `self`.
            _spawn_workers(ctx, target=self.create_packed_idx, jobs=self._worker_jobs(jobs))
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

    def _worker_jobs(self, jobs):
        # Bind the shared out-queue into each per-worker arg tuple at spawn time (queue may be rebuilt on
        # the fork fallback), keeping it out of `self` so non-fork contexts don't pickle the dataset.
        return [(rank, offset, lengths, self._out_queue, packing_length, packing_strategy)
                for rank, offset, lengths, packing_length, packing_strategy in jobs]

    @staticmethod
    def create_packed_idx(rank, offset, lengths, out_queue, packing_length, packing_strategy):
        data = [(i + offset, sum(length) if isinstance(length, list) else length) for i, length in enumerate(lengths)]
        i = 0
        input_data = []
        while True:
            new_data = data[i:i + PackingDataset.PACKING_BATCH_SIZE]
            input_data += new_data
            if not input_data:
                break
            i += PackingDataset.PACKING_BATCH_SIZE
            is_finished = i >= len(data)
            sequences, input_data = calculate_matched_group(
                input_data, packing_length, is_finished=is_finished, strategy=packing_strategy)
            out_queue.put((rank, sequences, len(new_data)))
        out_queue.put((rank, [], -1))

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
        # Pass the queues/template as explicit args (not via `self`) so a non-fork context only needs to
        # pickle the (model-stripped, see Template.__getstate__) template, not the whole dataset.
        self.workers = _spawn_workers(ctx, target=self._processor, jobs=self._worker_jobs())

    def _worker_jobs(self):
        # get_external_files() is resolved here, in the parent: plugins only ran in the main process and a
        # non-fork worker starts clean, so the paths have to travel with the job for the worker to replay them.
        return [(self._in_queue, self._out_queue, self.template, self.strict, get_external_files())] * self.num_proc

    @staticmethod
    def _processor(in_queue, out_queue, template, strict, external_files=()):
        for file_path in external_files:
            import_external_file(file_path)
        while True:
            i, data = in_queue.get()
            encoded_data = {}
            try:
                encoded_data = template.encode(data, return_length=True)
            except Exception as e:
                if strict and not isinstance(e, MaxLengthError):
                    raise
            out_queue.put((i, encoded_data))

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
