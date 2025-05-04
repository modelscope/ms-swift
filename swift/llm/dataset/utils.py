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
        return len(self.dataset)


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


class PackingDataset(Dataset):

    def __init__(self, template, dataset, num_proc: int = 1, *, packing_interval: int = 128, strict: bool = False):
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.packing_interval = packing_interval
        dataset = dataset.to_iterable_dataset(num_shards=num_proc)
        dataset = EncodePreprocessor(template)(dataset, num_proc=num_proc, strict=strict)
        self.packed_dataset = self.get_packed_dataset(dataset)

    def get_packed_dataset(self, dataset):
        data_list = []
        result = []
        it = iter(dataset)
        is_finished = False
        prog_bar = tqdm(total=len(dataset), dynamic_ncols=True, desc=f'Packing (num_proc={num_proc}):')

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

    def __init__(self,
                 template,
                 dataset,
                 num_proc: int = 1,
                 *,
                 packing_interval: int = 128,
                 strict: bool = False,
                 cyclic: bool = False):
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.packing_interval = packing_interval
        self.strict = strict
        self.cyclic = cyclic

        self._in_queue = mp.Queue()
        self._out_queue = mp.Queue()
        self.workers = []
        for _ in range(self.num_proc):
            worker = mp.Process(target=self._processor, daemon=True)
            worker.start()
            self.workers.append(worker)

    def _encode_data(self, data) -> Dict[str, Any]:
        res = None
        try:
            res = self.template.encode(data)
        except Exception as e:
            if self.strict and not isinstance(e, MaxLengthError):
                raise
        return res or {}

    def _processor(self):
        while True:
            data = self._in_queue.get()
            if data is None:
                encoded_data = None
            else:
                encoded_data = self._encode_data(data)
            self._out_queue.put(encoded_data)

    def _put_data_in_queue(self, iterator):
        for _ in range(self.packing_interval):
            try:
                data = next(iterator)
            except StopIteration:
                self._in_queue.put(None)
                return True
            self._in_queue.put(data)
        return False

    def _fetch_data_out_queue(self, res):
        for _ in range(self.packing_interval):
            data = self._out_queue.get()
            if data is None:
                break
            elif not data:
                continue
            res.append((data, len(data['input_ids'])))
        return res

    @staticmethod
    def cyclic_iter(iterable):
        while True:
            for x in iterable:
                yield x

    def __iter__(self):
        if self.cyclic:
            try:
                next(iter(self.dataset))
            except StopIteration:
                return
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
