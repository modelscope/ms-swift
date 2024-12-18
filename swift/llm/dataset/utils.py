# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from torch.utils.data import Dataset, IterableDataset

from swift.utils import get_logger
from ..template import MaxLengthError
from .preprocessor import DATASET_TYPE, RowPreprocessor

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


# Code borrowed from trl
class ConstantLengthDataset(IterableDataset):
    """This class wraps to do dataset packing
    Args:
        template: The template
        dataset: The dataset instance
        seq_length: The permitted sequence length
        num_of_sequences: Used to calculate the max_buffer_size fetched one time
        chars_per_token: Gives the chars per token, 3.6 if the default one, comes from `trl`
        append_concat_token: Reserved argument
        add_special_tokens: Reserved argument
    """

    def __init__(
        self,
        template: 'Template',
        dataset: DATASET_TYPE,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        self.template = template
        self.concat_token_id = self.template.tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.append_concat_token = append_concat_token
        self.add_special_tokens = add_special_tokens

    @staticmethod
    def get_packed_dataset(template: 'Template',
                           dataset: DATASET_TYPE,
                           seq_length=1024,
                           num_of_sequences=2048,
                           chars_per_token=3.6,
                           append_concat_token=True,
                           add_special_tokens=True):
        constant_length_iterator = ConstantLengthDataset(template, dataset, seq_length, num_of_sequences,
                                                         chars_per_token, append_concat_token, add_special_tokens)

        dataset_list = []
        for item in constant_length_iterator:
            dataset_list.append(item)
        return dataset_list

    def __len__(self):
        return len(self.dataset)

    def calculate_matched_group(self, sequences: Dict[str, List[int]]):
        # https://arxiv.org/pdf/2404.10830
        import binpacking
        binpacked = binpacking.to_constant_volume(sequences, self.seq_length, weight_pos=1)
        packed_sequence = []
        for sequence in binpacked:
            packed = {}
            position_id_lengths = [len(s[0]['input_ids']) for s in sequence]
            for key in sequence[0][0].keys():
                packed[key] = np.concatenate([s[0][key] for s in sequence])
            packed_sequence.append(packed)
            packed['position_ids'] = np.concatenate([list(range(pil)) for pil in position_id_lengths])
        return packed_sequence

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    example = next(iterator)
                    lens = sum([len(value) if value else 0 for value in example.values()])
                    buffer.append(example)
                    buffer_len += lens
                except StopIteration:
                    more_examples = False
                    break

            sequences = []
            for example in buffer:
                try:
                    inputs = self.template.encode(example)
                except MaxLengthError:
                    continue
                sequences.append((inputs, len(inputs['input_ids'])))

            if not sequences:
                return
            packed_sequences = self.calculate_matched_group(sequences)
            for sequence in packed_sequences:
                yield sequence


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
                    print(traceback.format_exc())
                    logger.error('👆👆👆There are errors in the template.encode, '
                                 'and another piece of data will be randomly selected.')
                    self._traceback_counter += 1

    def __len__(self) -> int:
        return len(self.dataset)


class EncodePreprocessor(RowPreprocessor):

    def __init__(self, template: 'Template'):
        super().__init__()
        self.template = template

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.template.encode(row)


class PackingPreprocessor(EncodePreprocessor):

    def batched_preprocess(self, batched_row: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        subset = self.batched_to_rows(batched_row)
        packed_dataset = ConstantLengthDataset.get_packed_dataset(
            self.template, dataset=subset, seq_length=self.template.max_length, num_of_sequences=4096)
        batched_row = self.rows_to_batched(packed_dataset)
        return batched_row


class GetLengthPreprocessor(RowPreprocessor):

    def __init__(self):
        super().__init__()

    def preprocess(self, row):
        length = max([len(row[k]) for k in row.keys() if k.endswith('input_ids')])
        return {'length': length}
