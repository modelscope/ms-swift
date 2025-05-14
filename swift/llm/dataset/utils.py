# Copyright (c) Alibaba, Inc. and its affiliates.
import bisect
import multiprocessing as mp
import os
import pickle
import time
from queue import Empty
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch.distributed as dist
from datasets import Dataset as HfDataset
from modelscope.hub.utils.utils import get_cache_dir
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from swift.utils import get_logger, is_dist, is_master
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
        self.cached_dataset = None
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
        if self.cached_dataset:
            return self.cached_dataset[idx]
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
                if n_try == self.n_try_fetch - 1:
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
        if self.cached_dataset:
            return len(self.cached_dataset)
        return len(self.dataset)

    @staticmethod
    def encode_data(data, map_func) -> Dict[str, Any]:
        res = None
        try:
            res = map_func(data)
        except Exception as e:
            if not isinstance(e, MaxLengthError):
                raise
        return res or {}

    @staticmethod
    def _map_single(shard_dataset: HfDataset, map_func, queue, rank: int, dataset_name: str):
        indexed_dataset_builder = IndexedDatasetBuilder(dataset_name, rank=rank)
        i = 0
        pre_i = 0
        item_list = []
        while i < len(shard_dataset):
            item = LazyLLMDataset.encode_data(shard_dataset[i], map_func)
            if item:
                item_list.append(item)
            if i % 1000 == 0:
                queue.put(i - pre_i)
                pre_i = i
                indexed_dataset_builder.add_items(item_list)
                item_list = []
            i += 1
        queue.put(i - pre_i)
        indexed_dataset_builder.add_items(item_list)
        indexed_dataset_builder.finalize()

    def create_cached_dataset(self, num_proc: int) -> None:
        dataset = self.dataset
        prog_bar = tqdm(total=len(dataset), dynamic_ncols=True, desc=f'Packing (num_proc={num_proc})')
        async_results = []
        dataset_name = 'encode-cache'
        with mp.Pool(num_proc) as pool:
            with mp.Manager() as manager:
                queue = manager.Queue()
                for i in range(num_proc):
                    shard_dataset = dataset.shard(num_proc, i)
                    async_results.append(
                        pool.apply_async(
                            self._map_single, args=(shard_dataset, self.encode_func, queue, i, dataset_name)))
                try:
                    while True:
                        try:
                            num = queue.get(timeout=0.05)
                            prog_bar.update(num)
                        except Empty:
                            if all(async_result.ready() for async_result in async_results) and queue.empty():
                                break
                finally:
                    # we get the result in case there's an error to raise
                    [async_result.get(timeout=0.05) for async_result in async_results]
        self.cached_dataset = IndexedDataset(dataset_name, num_proc)


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


class IndexedDatasetBuilder:

    def __init__(self, dataset_name: str, rank: int = 1):
        self.prefix_path = IndexedDataset.get_default_prefix_path(dataset_name, rank)
        self.bin_path = f'{self.prefix_path}.bin'
        self.idx_path = f'{self.prefix_path}.idx'
        if os.path.exists(self.bin_path):
            os.remove(self.bin_path)
        self.bin_file = open(self.bin_path, 'ab')
        self.idx_list = [0]

    def add_items(self, items: List[Any]) -> None:
        bin_buffer = []
        for item in items:
            item_buffer = pickle.dumps(item)
            bin_buffer.append(item_buffer)
            self.idx_list.append(self.idx_list[-1] + len(item_buffer))
        if bin_buffer:
            self.bin_file.write(b''.join(bin_buffer))

    def finalize(self):
        self.bin_file.close()
        with open(self.idx_path, 'wb') as f:
            pickle.dump(self.idx_list, f)


class IndexedDataset(Dataset):

    @staticmethod
    def _get_shard_prefix_path(prefix_path: str, rank: int):
        return f'{prefix_path}-{rank:05}'

    @staticmethod
    def get_default_prefix_path(dataset_name: str, rank: int):
        tmp_dir = os.path.join(get_cache_dir(), 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        assert dataset_name is not None, f'dataset_name: {dataset_name}'
        return IndexedDataset._get_shard_prefix_path(os.path.join(tmp_dir, dataset_name), rank)

    def __init__(self, dataset_name: str, n_shard: int = 1):
        prefix_path_list = [self.get_default_prefix_path(dataset_name, i) for i in range(n_shard)]
        self.bin_path_list = [f'{prefix_path}.bin' for prefix_path in prefix_path_list]
        self.idx_path_list = [f'{prefix_path}.idx' for prefix_path in prefix_path_list]
        self.bin_file_list = [open(bin_path, 'rb') for bin_path in self.bin_path_list]
        self.n_shard = n_shard
        self.idx_list = []
        self.idx_table = []
        idx = 0
        for idx_path in self.idx_path_list:
            with open(idx_path, 'rb') as f:
                idx_list = pickle.load(f)
                idx += len(idx_list)
                self.idx_table.append(idx)
                self.idx_list += idx_list

    def __getitem__(self, index: int):
        bucket_idx = bisect.bisect_left(self.idx_table, index)
        idx, idx_next = self.idx_list[index], self.idx_list[index + 1]
        self.bin_file_list[bucket_idx].seek(idx)
        return pickle.loads(self.bin_file_list[bucket_idx].read(idx_next - idx))

    def __len__(self):
        return len(self.idx_list) - 1


class PackingDataset(Dataset):

    def __init__(self,
                 template,
                 dataset: Union[HfDataset, LazyLLMDataset],
                 *,
                 num_proc: int = 1,
                 packing_interval: int = 128):
        template._packing = True
        self.template = template
        if isinstance(dataset, LazyLLMDataset):
            dataset.create_cached_dataset(num_proc)
        self.dataset = dataset
        self.packing_interval = packing_interval
        self.packed_dataset = self.get_packed_dataset(dataset)

    def get_packed_dataset(self, dataset):
        data_list = []
        result = []
        it = iter(dataset)
        is_finished = False
        prog_bar = tqdm(total=len(dataset), dynamic_ncols=True, desc='Packing:')
        while not is_finished:
            try:
                for _ in range(self.packing_interval):
                    data = next(it)
                    prog_bar.update(1)
                    data_list.append((data, len(data['input_ids'])))
            except StopIteration:
                is_finished = True
            res, data = calculate_matched_group(self.template, data_list, is_finished=is_finished)
            result += res
            if is_finished:
                break
        prog_bar.close()
        return result

    def __getitem__(self, index):
        return self.packed_dataset[index].copy()

    def __len__(self):
        return len(self.packed_dataset)


class IterablePackingDataset(IterableDataset):

    def __init__(self, template, dataset, *, num_proc: int = 1, packing_interval: int = 128, cyclic: bool = False):
        template._packing = True
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.packing_interval = packing_interval

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
                encoded_data = LazyLLMDataset.encode_data(data, self.template.encode)
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
            res, data = calculate_matched_group(self.template, data, is_finished=finished)
            yield from res
            if finished:
                break


class EncodePreprocessor(RowPreprocessor):

    def __init__(self, template: 'Template'):
        super().__init__()
        self.template = template

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.template.encode(row)


class GetLengthPreprocessor(RowPreprocessor):

    def preprocess(self, row):
        length = max([len(row[k]) for k in row.keys() if k.endswith('input_ids')])
        return {'length': length}
