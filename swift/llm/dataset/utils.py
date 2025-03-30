# Copyright (c) Alibaba, Inc. and its affiliates.
import multiprocessing as mp
import time
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from swift.utils import get_logger
from ..template import MaxLengthError
from .preprocessor import RowPreprocessor

logger = get_logger()


def sample_dataset(dataset: HfDataset,
                   dataset_sample: Optional[int],
                   random_state: Optional[np.random.RandomState] = None) -> HfDataset:
    """Sample dataset by a dataset_sample number
    Args:
        dataset: The dataset instance, iterable dataset is not supported
        dataset_sample: The sample number
        random_state: The random state
    Returns:
        The sampled dataset
    """
    if dataset_sample is None:
        return dataset
    if random_state is None:
        random_state = np.random.RandomState()

    n_repeat_sample = dataset_sample // len(dataset)
    n_random_sample = dataset_sample % len(dataset)
    if n_repeat_sample >= 1 and n_random_sample >= 1:
        logger.warning(f'dataset_sample:{dataset_sample} is greater than len(dataset):{len(dataset)}, '
                       'repeated sampling will be performed.')
    idx = np.tile(range(len(dataset)), n_repeat_sample)
    if n_random_sample >= 1:
        idx_random = random_state.permutation(len(dataset))[:n_random_sample]
        idx = np.concatenate([idx, idx_random])
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
            if i == 0:
                i = idx
            else:
                i = self._idx_list[self._idx]
                self._idx = (self._idx + 1) % len(self.dataset)
            data = self.dataset[i]
            try:
                return self.encode_func(data)
            except Exception:
                if i == self.n_try_fetch - 1:
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

    def __init__(self,
                 template,
                 dataset,
                 num_workers: int = 1,
                 *,
                 packing_interval: int = 128,
                 queue_buffer: Optional[int] = None,
                 strict: bool = False):
        template._packing = True
        self.template = template
        self.dataset = dataset
        self.num_workers = num_workers
        self.packing_interval = packing_interval
        self.queue_buffer = queue_buffer or max(2 * packing_interval, 1000)
        self.strict = strict
        self.prog_bar = None
        assert num_workers >= 1, f'num_workers: {num_workers}'
        assert self.queue_buffer >= self.packing_interval, (
            f'queue_buffer: {queue_buffer}, packing_interval: {packing_interval}')
        self.workers = []
        self._queue = None
        self._terminated_workers = 0

    def get_process(self, rank: int):
        raise NotImplementedError

    def _init_workers(self):
        self._terminate_workers()
        for i in range(self.num_workers):
            worker = self.get_process(i)
            worker.start()
            self.workers.append(worker)

    def _terminate_workers(self):
        for worker in self.workers:
            worker.terminate()
        self.workers = []
        self._queue = mp.Queue(maxsize=self.queue_buffer)
        self._terminated_workers = 0

    def fetch_packing_data(self, res: Optional[list] = None):
        res = res or []
        for _ in range(self.packing_interval):
            data = self._queue.get()
            if data is None:
                self._terminated_workers += 1
                if self._terminated_workers == self.num_workers:
                    break
                continue
            if getattr(self, 'prog_bar'):
                self.prog_bar.update(1)
            res.append((data, len(data['input_ids'])))
        return res

    @staticmethod
    def calculate_matched_group(template, sequences, drop_last: bool = True):
        if len(sequences) == 0:
            return []
        # https://arxiv.org/pdf/2404.10830
        import binpacking
        sequences = binpacking.to_constant_volume(sequences, template.max_length, weight_pos=1)
        res = []
        if sequences and drop_last:
            sequences, ret_sequences = sequences[:-1], sequences[-1]
        else:
            ret_sequences = []
        for row in sequences:
            packed = template.packing_row(row)
            res.append(packed)
        return res, ret_sequences

    def run_packing(self):
        self._terminate_workers()
        self._init_workers()
        data = []
        while True:
            data = self.fetch_packing_data(data)
            is_finished = self._terminated_workers == self.num_workers
            res, data = self.calculate_matched_group(self.template, data, drop_last=not is_finished)
            yield from res
            if is_finished:
                break
        self._terminate_workers()

    def _encode_and_put_queue(self, data):
        try:
            data = self.template.encode(data)
            self._queue.put(data)
        except Exception as e:
            if self.strict and not isinstance(e, MaxLengthError):
                raise


class PackingDataset(BasePackingDataset, Dataset):

    def __init__(self,
                 template,
                 dataset,
                 num_workers: int = 1,
                 *,
                 packing_interval: int = 128,
                 queue_buffer: Optional[int] = None,
                 strict: bool = False):
        super().__init__(
            template, dataset, num_workers, packing_interval=packing_interval, queue_buffer=queue_buffer, strict=strict)
        self.prog_bar = tqdm(total=len(dataset), dynamic_ncols=True, desc='Packing')
        self._init_workers()
        self.packed_dataset = list(self.run_packing())
        self._terminate_workers()
        self.prog_bar.close()

    def _producer(self, shard_dataset):
        for data in shard_dataset:
            self._encode_and_put_queue(data)
        self._queue.put(None)
        while True:
            # Wait for the main process to terminate to avoid fd anomalies.
            time.sleep(0.1)

    def get_process(self, rank: int):
        self.dataset: HfDataset
        shard_dataset = self.dataset.shard(self.num_workers, rank)
        return mp.Process(target=self._producer, args=(shard_dataset, ), daemon=True)

    def __getitem__(self, index):
        return self.packed_dataset[index].copy()

    def __len__(self):
        return len(self.packed_dataset)


class IterablePackingDataset(BasePackingDataset, IterableDataset):

    def _producer(self, rank):
        dataset = iter(self.dataset)
        while True:
            data = None
            try:
                for i in range(self.num_workers):
                    _data = next(dataset)
                    if i == rank:
                        data = _data
            except StopIteration:
                if data is not None:
                    self._encode_and_put_queue(data)
                break
            self._encode_and_put_queue(data)
        self._queue.put(None)
        while True:
            # Wait for the main process to terminate to avoid fd anomalies.
            time.sleep(0.1)

    def get_process(self, rank: int):
        return mp.Process(target=self._producer, args=(rank,), daemon=True)

    def __iter__(self):
        yield from self.run_packing()


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
