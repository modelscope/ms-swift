# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from torch.utils.data import Dataset

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

    def batched_preprocess(self, batched_row: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        rows = self.batched_to_rows(batched_row)
        inputs_list = []
        for row in rows:
            try:
                inputs = self.template.encode(row)
            except MaxLengthError:
                continue
            inputs_list.append((inputs, len(inputs['input_ids'])))
        packed = self.calculate_matched_group(inputs_list)
        return self.rows_to_batched(packed)

    def calculate_matched_group(self, sequences):
        # https://arxiv.org/pdf/2404.10830
        import binpacking
        keys = list(sequences[0][0].keys())
        sequences = binpacking.to_constant_volume(sequences, self.template.max_length, weight_pos=1)
        res = []
        for row in sequences:
            packed = {}
            for key in keys:
                if key == 'labels':
                    labels_list = []
                    for x in row:
                        labels = x[0][key]
                        # https://github.com/huggingface/transformers/pull/31629
                        labels[0] = -100
                        labels_list.append(labels)
                    packed[key] = sum(labels_list, start=[])
                else:
                    packed[key] = sum((x[0][key] for x in row), start=[])
            packed['position_ids'] = sum((list(range(x[1])) for x in row), start=[])
            res.append(packed)
        return res


class GetLengthPreprocessor(RowPreprocessor):

    def preprocess(self, row):
        length = max([len(row[k]) for k in row.keys() if k.endswith('input_ids')])
        return {'length': length}
