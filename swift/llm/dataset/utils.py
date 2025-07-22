# Copyright (c) Alibaba, Inc. and its affiliates.
import multiprocessing as mp
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

import numpy as np
import torch.distributed as dist
from datasets import Dataset as HfDataset
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from swift.utils import get_logger, is_dist, is_master
from ..template import MaxLengthError
from .preprocessor import RowPreprocessor

logger = get_logger()

if TYPE_CHECKING:
    from swift.llm import Template


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
        if isinstance(idx, str):
            return self.dataset[idx]
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


def calculate_matched_group(template, sequences, is_finished: bool = True):
    if len(sequences) == 0:
        return [], []
    # https://arxiv.org/pdf/2404.10830
    import binpacking
    sequences = binpacking.to_constant_volume(sequences, template.max_length, weight_pos=1)
    if sequences and not is_finished:
        sequences, ret_sequences = sequences[:-1], sequences[-1]
    else:
        ret_sequences = []
    return sequences, ret_sequences


class PackingDataset(Dataset):

    def __init__(
        self,
        template,
        dataset,
        num_proc: int = 1,
        *,
        strict: bool = False,
        load_from_cache_file: bool = True,
        **kwargs,
    ):
        template._packing = True
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.strict = strict
        self.load_from_cache_file = load_from_cache_file
        self.workers = []
        preprocessor = EncodePreprocessor(template=template)
        self.dataset = preprocessor(
            dataset, num_proc=num_proc, load_from_cache_file=load_from_cache_file, strict=strict)
        if template.model_meta.is_multimodal:
            self.dataset = LazyLLMDataset(self.dataset, encode_func=template.encode)
        self.packed_idx = self.create_packed_idx() if is_master() else None
        if dist.is_initialized() and is_dist():
            obj_list = [self.packed_idx]
            dist.broadcast_object_list(obj_list)
            self.packed_idx = obj_list[0]

    def create_packed_idx(self):
        lengths = self.dataset['length']
        data = [(i, length) for i, length in enumerate(lengths)]
        i = 0
        PACKING_BATCH_SIZE = 1000
        input_data, res = [], []
        with tqdm(total=len(data), dynamic_ncols=True, desc='Packing: ') as prog_bar:
            while True:
                new_data = data[i:i + PACKING_BATCH_SIZE]
                input_data += new_data
                prog_bar.update(len(new_data))
                if not input_data:
                    break
                i += PACKING_BATCH_SIZE
                is_finished = i >= len(data)
                sequences, input_data = calculate_matched_group(self.template, input_data, is_finished=is_finished)
                res += sequences
        return res

    def __getitem__(self, index):
        sequence = self.packed_idx[index]
        row = []
        for i, length in sequence:
            row.append((self.dataset[i], length))
        return self.template.packing_row(row)

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
        strict: bool = False,
        cyclic: bool = False,
        **kwargs,
    ):
        template._packing = True
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.strict = strict

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
            encoded_data = {}
            try:
                encoded_data = self.template.encode(data)
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
            num_samples = self._put_data_in_queue(iterator)
            finished = num_samples != self.packing_interval
            data = self._fetch_data_out_queue(data, num_samples)
            sequences, data = calculate_matched_group(self.template, data, is_finished=finished)
            res = []
            for row in sequences:
                packed = self.template.packing_row(row)
                res.append(packed)
            yield from res
            if finished:
                break


class EncodePreprocessor(RowPreprocessor):

    def __init__(self, template: 'Template'):
        super().__init__()
        self.template = template
        self.is_multimodal = template.model_meta.is_multimodal

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        encoded = self.template.encode(row)
        if self.is_multimodal:
            row['length'] = encoded['length']
            encoded = row
        return encoded
