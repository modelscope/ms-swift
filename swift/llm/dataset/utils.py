# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import heapq
import os
from copy import deepcopy
from functools import partial
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import multiprocess
import numpy as np
import torch
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HFIterableDataset
from datasets.arrow_dataset import iflatmap_unordered
from torch.utils.data import Dataset, IterableDataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from swift.utils import get_logger, stat_array
from .preprocess import DATASET_TYPE, RowPreprocessor

logger = get_logger()


def sample_dataset(dataset: HfDataset,
                   dataset_sample: int,
                   random_state: Optional[np.random.RandomState] = None) -> HfDataset:
    """Sample dataset by a dataset_sample number
    Args:
        dataset: The dataset instance, iterable dataset is not supported
        dataset_sample: The sample number
        random_state: The random state
    Returns:
        The sampled dataset
    """
    if dataset_sample in {None, -1, len(dataset)}:
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


class LLMDataset(Dataset):
    """This class wraps the Dataset class, to offer the ability of custom dataset tokenizing"""

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self.data = data

    def __getitem__(self, idx: Union[int, str]) -> Dict[str, Any]:
        if isinstance(idx, int):
            data = self.data[idx]
            return data
        elif isinstance(idx, str):
            return [d[idx] for d in self.data]
        else:
            raise ValueError(f'idx: {idx}')

    def select(self, idx_list: List[int]) -> 'LLMDataset':
        data = [self.data[i] for i in idx_list]
        return self.__class__(data)

    def __len__(self) -> int:
        return len(self.data)


class LLMIterableDataset(HFIterableDataset):
    """This class offers abilities of deal with IterableDataset, and skip the bad samples"""

    def __init__(self, dataset: HFIterableDataset, max_retries=10):
        super().__init__(
            dataset._ex_iterable,
            dataset._info,
            dataset._split,
            dataset._formatting,
            dataset._shuffling,
            dataset._distributed,
            dataset._token_per_repo_id,
        )
        self.dataset = dataset
        self.max_retries = max_retries
        from swift.llm.dataset.dataset import standard_keys
        dataset._ex_iterable.remove_columns = standard_keys & next(iter(dataset)).keys()

    def __iter__(self):
        """Iter the dataset, skip bad ones. This iter will never stop until your max-length reached.
        Yields:
            An example
        """
        iterator = iter(self.dataset)
        while True:
            retries = 0
            while retries < self.max_retries:
                try:
                    value = next(iterator)
                    if value:
                        yield value
                        break
                    else:
                        raise ValueError
                except StopIteration:
                    iterator = iter(self.dataset)
                    break
                except Exception as e:
                    retries += 1
                    if retries >= self.max_retries:
                        raise e


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
                           add_special_tokens=True,
                           lazy_tokenize=False):
        constant_length_iterator = ConstantLengthDataset(template, dataset, seq_length, num_of_sequences,
                                                         chars_per_token, append_concat_token, add_special_tokens)

        if lazy_tokenize:
            return constant_length_iterator

        dataset_list = []
        for item in constant_length_iterator:
            dataset_list.append(item)
        return HfDataset.from_list(dataset_list)

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
                    buffer.append(next(iterator))
                    buffer_len += lens
                except StopIteration:
                    more_examples = False
                    break

            sequences = []
            for example in buffer:
                input, _ = self.template.encode(example)
                if not input:
                    continue
                sequences.append((input, len(input['input_ids'])))

            packed_sequences = self.calculate_matched_group(sequences)
            for sequence in packed_sequences:
                yield sequence


class LazyLLMDataset(Dataset):
    """This class if used to lazy tokenize the dataset, and skips bad ones when training"""

    def __init__(self,
                 dataset: HfDataset,
                 encode_func: Callable[[Dict[str, Any]], Union[Tuple[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]],
                 *,
                 try_fetch_time: int = 20) -> None:
        self.dataset = dataset
        self.encode_func = encode_func
        self.try_fetch_time = min(try_fetch_time, len(self.dataset))
        assert self.try_fetch_time >= 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        res = self._try_fetch(idx)
        if res is not None:
            return res
        raise ValueError('Please check if the max_length is appropriate.')

    def _try_fetch(self, first_idx: int) -> Optional[Dict[str, Any]]:
        idx = np.random.permutation(len(self))[:self.try_fetch_time - 1]
        for i in [first_idx] + idx.tolist():
            data = self.dataset[i]
            try:
                res = self.encode_func(data)
                if isinstance(res, (tuple, list)) and len(res) == 2:
                    res = res[0]
            except Exception as e:
                logger.error(f'Error occurs in lazy tokenize: {e}')
                continue
            if len(res) > 0:
                return res

    def __len__(self) -> int:
        return len(self.dataset)


class EncodePreprocessor(RowPreprocessor):

    def __init__(self, template: 'Template'):
        super().__init__(remove_useless_columns=False)
        self.template = template

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        res = self.template.encode(row)
        if len(res) == 0:
            res = None
        return res

def stat_dataset(llm_dataset: Dataset) -> str:
    """Statistical analysis was performed on the dataset"""
    token_len = _get_token_len(llm_dataset)
    _, stat_str = stat_array(token_len)
    logger.info(f'Dataset Token Length: {stat_str}')
    return stat_str


def _get_token_len(llm_dataset):
    token_len = []
    if isinstance(llm_dataset, HfDataset):  # compat hf_dataset
        input_ids = llm_dataset['input_ids']
        for ii in input_ids:
            token_len.append(len(ii))
    else:
        for d in llm_dataset:  # LLMDataset
            _len = 0
            for k, v in d.items():
                if k == 'input_ids' or k.endswith('_input_ids'):  # sft, rlhf
                    _len += len(v)
            token_len.append(_len)
    return token_len


def print_example(example: Dict[str, Any],
                  tokenizer: PreTrainedTokenizerBase,
                  tokenizer_kwargs: Optional[Dict[str, Any]] = None) -> None:
    """Print example"""
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
    for key in ['input', 'chosen_input', 'rejected_input', 'labels', 'chosen_labels', 'rejected_labels']:
        val = example.get(key)  # fix val is a tensor
        if val is None:
            val = example.get(f'{key}_ids')
        if val is not None:
            key_upper = key.upper()
            logger.info(f'[{key_upper}_IDS] {val}')
            val_str = safe_tokenizer_decode(tokenizer, val, **tokenizer_kwargs)
            logger.info(f'[{key_upper}] {val_str}')


def sort_by_max_length(llm_dataset: LLMDataset, num_dataset: int) -> LLMDataset:
    """Sort dataset by max length, this is always used in OOM testing scenario"""
    logger.info('sort by max length...')
    token_len = _get_token_len(llm_dataset)
    idx = heapq.nlargest(num_dataset, range(len(token_len)), key=lambda i: token_len[i])
    return llm_dataset.select(idx)
