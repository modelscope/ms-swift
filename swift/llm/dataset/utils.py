# Copyright (c) Alibaba, Inc. and its affiliates.
import bisect
import mmap
import multiprocessing as mp
import os
import pickle
import threading
import time
from copy import copy
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from modelscope.hub.utils.utils import get_cache_dir
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from swift.utils import get_logger, is_master, safe_ddp_context
from ..template import MaxLengthError
from .preprocessor import RowPreprocessor

logger = get_logger()


def sample_dataset(
    dataset: HfDataset,
    dataset_sample: Optional[int],
    shuffle: bool = True,
    random_state: Optional[np.random.RandomState] = None,
) -> HfDataset:
    """Sample dataset by a dataset_sample number
    Args:
        dataset: The dataset instance, iterable dataset is not supported
        dataset_sample: The sample number
        shuffle: Whether to perform random sampling on non-streaming datasets
        random_state: The random state
    Returns:
        The sampled dataset
    """
    if dataset_sample is None:
        return dataset

    n_repeat_sample = dataset_sample // len(dataset)
    n_remain_sample = dataset_sample % len(dataset)
    if n_repeat_sample >= 1 and n_remain_sample >= 1:
        logger.warning(f'dataset_sample:{dataset_sample} is greater than len(dataset):{len(dataset)}, '
                       'repeated sampling will be performed.')
    idx = np.tile(range(len(dataset)), n_repeat_sample)
    if n_remain_sample >= 1:
        if shuffle:
            if random_state is None:
                random_state = np.random.RandomState()
            idx_remain = random_state.permutation(len(dataset))[:n_remain_sample]
        else:
            idx_remain = np.arange(n_remain_sample)
        idx = np.concatenate([idx, idx_remain])
    dataset = dataset.select(idx)
    return dataset


class LazyLLMDataset(Dataset):
    """This class if used to lazy tokenize the dataset, and skips bad ones when training"""

    def __init__(self,
                 dataset: HfDataset,
                 encode_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                 *,
                 n_try_fetch: int = 10,
                 strict: bool = False,
                 random_state: Union[np.random.RandomState, int, None] = None,
                 traceback_limit: int = 10) -> None:
        self.dataset = dataset
        self.encode_func = encode_func

        n_try_fetch = 1 if strict else min(n_try_fetch, len(self.dataset))
        assert n_try_fetch >= 1
        self.strict = strict
        self.n_try_fetch = n_try_fetch

        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

        self.traceback_limit = traceback_limit
        self._traceback_counter = 0
        self._idx = 0
        self._idx_list = self.random_state.permutation(len(self.dataset)).tolist()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        for i in range(self.n_try_fetch):
            n_try = i
            if i == 0:
                i = idx
            else:
                i = self._idx_list[self._idx]
                self._idx = (self._idx + 1) % len(self.dataset)
            data = self.dataset[i]
            try:
                return self.encode_func(data)
            except Exception:
                if n_try == self.n_try_fetch - 1 or self.strict:
                    if self.strict:
                        logger.warning('To avoid errors, you can pass `strict=False`.')
                    raise
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


class BasePackingDataset:

    def __init__(self, template, dataset, num_proc: int = 1, *, packing_interval: int = 128, strict: bool = False):
        template._packing = True
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.packing_interval = packing_interval
        self.strict = strict
        assert num_proc >= 1, f'num_proc: {num_proc}'
        self.workers = []

    @staticmethod
    def calculate_matched_group(template, sequences, is_finished: bool = True):
        if len(sequences) == 0:
            return [], []
        # https://arxiv.org/pdf/2404.10830
        import binpacking
        sequences = binpacking.to_constant_volume(sequences, template.max_length, weight_pos=1)
        res = []
        if sequences and not is_finished:
            sequences, ret_sequences = sequences[:-1], sequences[-1]
        else:
            ret_sequences = []
        for row in sequences:
            packed = template.packing_row(row)
            res.append(packed)
        return res, ret_sequences

    def _encode_data(self, data) -> Dict[str, Any]:
        res = None
        try:
            res = self.template.encode(data)
        except Exception as e:
            if self.strict and not isinstance(e, MaxLengthError):
                raise
        return res or {}


class IndexedDatasetBuilder:
    CHUNK_SIZE = 1e10

    def __init__(self, dataset_name: str):
        self.cache_dir = IndexedDataset.get_cache_dir(dataset_name)
        self.n_shard = 1
        self.bin_path = os.path.join(self.cache_dir, IndexedDataset.BIN_FNAME.format(0))
        self.idx_path = os.path.join(self.cache_dir, IndexedDataset.IDX_FNAME)
        if os.path.exists(self.bin_path):
            os.remove(self.bin_path)
        self.bin_file = open(self.bin_path, 'ab')
        self.length_list = []
        self.idx_list = [0]
        self.shard_offset = [0]
        self._thread = None
        self._queue = Queue(maxsize=1000)

    def _write_worker(self):
        while True:
            items = self._queue.get()
            if items is None:
                break
            bin_buffer = []
            for item in items:
                item_buffer = pickle.dumps(item)
                bin_buffer.append(item_buffer)
                self.idx_list.append(self.idx_list[-1] + len(item_buffer))
                self.length_list.append(item['length'])
            self.bin_file.write(b''.join(bin_buffer))
            offset = self.idx_list[-1] - self.shard_offset[-1]
            if offset >= self.CHUNK_SIZE:
                self.bin_file.close()
                self.bin_path = os.path.join(self.cache_dir, IndexedDataset.BIN_FNAME.format(self.n_shard))
                self.shard_offset.append(self.shard_offset[-1] + offset)
                self.n_shard += 1
                if os.path.exists(self.bin_path):
                    os.remove(self.bin_path)
                self.bin_file = open(self.bin_path, 'ab')

    def add_items(self, items: List[Any]) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._write_worker, daemon=True)
            self._thread.start()
        self._queue.put(items)

    def finalize(self):
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
        self.bin_file.close()
        idx_obj = {
            'idx': self.idx_list,
            'length': self.length_list,
            'n_shard': self.n_shard,
            'shard_offset': self.shard_offset,
        }
        with open(self.idx_path, 'wb') as f:
            pickle.dump(idx_obj, f)


class BinReader:

    def __init__(self, bin_path: str):
        self.bin_path = bin_path
        self.file = open(bin_path, 'rb')
        try:
            self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:
            # For example, self.file is an empty file.
            self.mm = None

    def read_buffer(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset + size > len(self.mm):
            raise ValueError('Invalid offset or size')
        return self.mm[offset:offset + size]

    def __del__(self):
        if self.mm is not None:
            self.mm.close()
        self.file.close()


class IndexedDataset(Dataset):
    BIN_FNAME = 'data-{:05d}.bin'
    IDX_FNAME = 'data.idx'

    @staticmethod
    def get_cache_dir(dataset_name: str):
        cache_dir = os.getenv('PACKING_CACHE') or os.path.join(get_cache_dir(), 'tmp')
        cache_dir = os.path.join(cache_dir, dataset_name)
        os.makedirs(cache_dir, exist_ok=True)
        assert dataset_name is not None, f'dataset_name: {dataset_name}'
        return cache_dir

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        cache_dir = self.get_cache_dir(dataset_name)
        self.idx_path = os.path.join(cache_dir, self.IDX_FNAME)
        with open(self.idx_path, 'rb') as f:
            idx_obj = pickle.load(f)
        self.idx_list = idx_obj['idx']
        self.length_list = idx_obj['length']
        self.n_shard = idx_obj['n_shard']
        self.shard_offset = idx_obj['shard_offset']
        self.bin_readers = []
        for i in range(self.n_shard):
            bin_path = os.path.join(cache_dir, self.BIN_FNAME.format(i))
            self.bin_readers.append(BinReader(bin_path))

    def __getitem__(self, index: int):
        if index < 0:
            index = index % len(self)
        idx, idx_next = self.idx_list[index], self.idx_list[index + 1]
        num_shard = bisect.bisect_right(self.shard_offset, idx)
        offset = self.shard_offset[num_shard - 1]
        buffer = self.bin_readers[num_shard - 1].read_buffer(idx - offset, idx_next - idx)
        return pickle.loads(buffer)

    def __len__(self):
        return len(self.idx_list) - 1


class PackingDataset(BasePackingDataset, Dataset):

    def __init__(
        self,
        template,
        dataset,
        num_proc: int = 1,
        *,
        packing_interval: int = 128,
        strict: bool = False,
        load_from_cache_file: bool = True,
        **kwargs,
    ):
        from datasets.fingerprint import update_fingerprint
        num_proc = min(len(dataset), num_proc)
        super().__init__(template, dataset, num_proc, packing_interval=packing_interval, strict=strict)
        self.load_from_cache_file = load_from_cache_file
        template = copy(template)
        template.model = None  # Avoid hashing the model.
        fingerprint = update_fingerprint(dataset._fingerprint, 'PackingDataset', {
            'template': template,
            'packing_interval': packing_interval,
            'strict': strict,
        })
        self.dataset_name = f'packing-cache-{fingerprint}'
        with safe_ddp_context(None, True):
            self.create_packed_dataset()

    def create_packed_dataset(self):
        cache_dir = IndexedDataset.get_cache_dir(self.dataset_name)
        logger.info(f'packing cache_dir: {cache_dir}')
        if not os.path.exists(os.path.join(cache_dir,
                                           IndexedDataset.IDX_FNAME)) or not self.load_from_cache_file and is_master():
            self._queue = mp.Queue(maxsize=1000)
            self._terminated_workers = 0
            self.prog_bar = tqdm(
                total=len(self.dataset), dynamic_ncols=True, desc=f'Packing (num_proc={self.num_proc})')
            for i in range(self.num_proc):
                shard_dataset = self.dataset.shard(self.num_proc, i)
                worker = mp.Process(target=self._producer, args=(shard_dataset, ), daemon=True)
                worker.start()
                self.workers.append(worker)

            self.packing_dataset()
            self.prog_bar.close()
            for worker in self.workers:
                worker.terminate()
        self.packed_dataset = IndexedDataset(self.dataset_name)

    def fetch_packing_data(self, res: Optional[list] = None):
        res = res or []
        for _ in range(self.packing_interval):
            data = self._queue.get()
            if data is None:
                self._terminated_workers += 1
                if self._terminated_workers == self.num_proc:
                    break
                continue
            self.prog_bar.update(1)
            if data:
                res.append((data, data['length']))
        return res

    def packing_dataset(self):
        data = []
        indexed_dataset_builder = IndexedDatasetBuilder(self.dataset_name)
        while True:
            data = self.fetch_packing_data(data)
            is_finished = self._terminated_workers == self.num_proc
            res, data = self.calculate_matched_group(self.template, data, is_finished=is_finished)
            indexed_dataset_builder.add_items(res)
            if is_finished:
                break
        indexed_dataset_builder.finalize()

    def _producer(self, shard_dataset):
        for data in shard_dataset:
            encoded_data = self._encode_data(data)  # ignore
            self._queue.put(encoded_data)
        self._queue.put(None)
        while True:
            # Wait for the main process to terminate to avoid fd anomalies.
            time.sleep(0.1)

    def __getitem__(self, index):
        return self.packed_dataset[index].copy()

    def __len__(self):
        return len(self.packed_dataset)


class IterablePackingDataset(BasePackingDataset, IterableDataset):

    def __init__(
        self,
        template,
        dataset,
        num_proc: int = 1,
        *,
        packing_interval: int = 128,
        strict: bool = False,
        cyclic: bool = False,
        **kwargs,
    ):
        super().__init__(template, dataset, num_proc, packing_interval=packing_interval, strict=strict)
        self._in_queue = mp.Queue()
        self._out_queue = mp.Queue()
        self.workers = []
        self.cyclic = cyclic
        for _ in range(self.num_proc):
            worker = mp.Process(target=self._processor, daemon=True)
            worker.start()
            self.workers.append(worker)

    def _processor(self):
        while True:
            i, data = self._in_queue.get()
            if data is None:
                encoded_data = None
            else:
                encoded_data = self._encode_data(data)
            self._out_queue.put((i, encoded_data))

    def _put_data_in_queue(self, iterator):
        for i in range(self.packing_interval):
            try:
                data = next(iterator)
            except StopIteration:
                self._in_queue.put((i, None))
                return True
            self._in_queue.put((i, data))
        return False

    def _fetch_data_out_queue(self, last_res):
        res = [None] * self.packing_interval
        for _ in range(self.packing_interval):
            i, data = self._out_queue.get()
            if data is None:
                break
            elif not data:
                continue
            res[i] = (data, len(data['input_ids']))
        res = [data for data in res if data]
        last_res += res
        return last_res

    @staticmethod
    def cyclic_iter(iterable):
        while True:
            for x in iterable:
                yield x

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
            finished = self._put_data_in_queue(iterator)
            data = self._fetch_data_out_queue(data)
            res, data = self.calculate_matched_group(self.template, data, is_finished=finished)
            yield from res
            if finished:
                break


class EncodePreprocessor(RowPreprocessor):

    def __init__(self, template: 'Template'):
        super().__init__()
        self.template = template

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.template.encode(row)
