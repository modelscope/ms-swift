# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from torch.utils.data import Dataset, IterableDataset
from swift.utils import get_logger
from functools import partial
import multiprocessing as mp
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
    def __init__(self, dataset, template, num_workers: int = 4, *,
                 packing_buffer: int = 1000,
                 queue_buffer: Optional[int] = None):
        self.dataset = dataset
        self.template = template
        self.template._packing = True
        self.num_workers = num_workers
        self.packing_buffer = packing_buffer
        self.queue_buffer = queue_buffer or 2 * packing_buffer
        assert self.queue_buffer >= self.packing_buffer, f'queue_buffer: {queue_buffer}, packing_buffer: {packing_buffer}'
        self.workers = []
        self._queue = mp.Queue(maxsize=self.queue_buffer)
        self._terminated_workers = 0

    def _terminate_workers(self):
        for worker in self.workers:
            worker.terminate()
        self.workers = []
        self._terminated_workers = 0

    def fetch_packing_data(self):
        res = []
        for _ in range(self.packing_buffer):
            data = self._queue.get()
            if data is None:
                self._terminated_workers += 1
                if self._terminated_workers == self.num_workers:
                    break
                continue
            res.append(data)
        return res


class PackingDataset(BasePackingDataset, Dataset):

    def _encode_worker(self, rank: int):
        pass

    def __init__(self, dataset, template, num_workers: int = 4, *,
                 packing_buffer: int = 1000,
                 queue_buffer: Optional[int] = None):
        super().__init__(dataset, template, num_workers, packing_buffer=packing_buffer, queue_buffer=queue_buffer)
        self._init_workers()
        self._terminate_workers()

    def _init_workers(self):
        for i in range(self.num_workers):
            worker = mp.Process(target=partial(self._encode_worker, rank=i), daemon=True)
            worker.start()
            self.workers.append(worker)

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class IterablePackingDataset(BasePackingDataset, IterableDataset):

    def __init__(self, dataset, template, num_workers: int = 4, strict: bool = True, *,
                 packing_buffer: int = 1000,
                 queue_buffer: Optional[int] = None):
        self.strict = strict
        super().__init__(dataset, template, num_workers, packing_buffer=packing_buffer, queue_buffer=queue_buffer)

    def _init_workers(self):
        self._terminate_workers()
        for i in range(self.num_workers):
            worker = mp.Process(target=partial(self._encode_worker, rank=i), daemon=True)
            worker.start()
            self.workers.append(worker)

    def _encode_worker(self, rank: int):
        dataset = iter(self.dataset)
        while True:
            try:
                data = None
                for i in range(self.num_workers):
                    _data = next(dataset)
                    if i == rank:
                        data = _data
            except StopIteration:
                self._queue.put(None)
                break
            try:
                inputs = self.template.encode(data)
                self._queue.put(inputs)
            except MaxLengthError:
                continue
            except Exception:
                if self.strict:
                    raise

    def __iter__(self):
        self._init_workers()
        while self._terminated_workers < self.num_workers:
            data = self.fetch_packing_data()
            yield from PackingPreprocessor.calculate_matched_group(self.template, data)
        self._terminate_workers()


class EncodePreprocessor(RowPreprocessor):

    def __init__(self, template: 'Template'):
        super().__init__()
        self.template = template

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.template.encode(row)


class PackingPreprocessor(EncodePreprocessor):

    def __init__(self, template: 'Template'):
        super().__init__(template=template)
        self.template._packing = True

    def _rename_columns(self, dataset):
        # fix streaming
        return dataset

    def batched_preprocess(self, batched_row: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        rows = self.batched_to_rows(batched_row)
        inputs_list = []
        for row in rows:
            try:
                inputs = self.template.encode(row)
            except MaxLengthError:
                continue
        packed = self.calculate_matched_group(inputs_list)
        return self.rows_to_batched(packed)

    @staticmethod
    def calculate_matched_group(template, sequences):
        if len(sequences) == 0:
            return []
        # https://arxiv.org/pdf/2404.10830
        sequences = [(seq, len(seq['input_ids'])) for seq in sequences]
        import binpacking
        sequences = binpacking.to_constant_volume(sequences, template.max_length, weight_pos=1)
        res = []
        for row in sequences:
            packed = template.packing_row(row)
            res.append(packed)
        return res


class GetLengthPreprocessor(RowPreprocessor):

    def preprocess(self, row):
        length = max([len(row[k]) for k in row.keys() if k.endswith('input_ids')])
        return {'length': length}
