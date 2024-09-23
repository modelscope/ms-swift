# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import heapq
import os
from functools import partial
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import multiprocess
import numpy as np
import torch
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HFIterableDataset
from torch.utils.data import Dataset, IterableDataset
from tqdm.auto import tqdm
from transformers import (PreTrainedTokenizerBase)

from swift.utils import get_logger, stat_array

DATASET_TYPE = Union[HfDataset, HFIterableDataset]

logger = get_logger()

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


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


MapFunc = Callable[[Dict[str, Any]], Tuple[Dict[str, Any], Dict[str, Any]]]


def _single_map(d: Dict[str, Any], map_func: MapFunc) -> Optional[Dict[str, Any]]:
    d = map_func(d)[0]
    if len(d) == 0:
        return None
    return d


def _map_mp_single(subset: HfDataset, map_func: MapFunc, queue: Queue, start_idx: int):
    for i, d in enumerate(subset, start=start_idx):
        queue.put((i, map_func(d)))  # idx, result


def _map_mp_i(dataset: HfDataset, map_func: MapFunc, num_proc: int) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with multiprocess.Pool(num_proc) as pool, multiprocess.Manager() as manager:
        queue = manager.Queue()
        async_results = []
        split_idx = np.linspace(0, len(dataset), num_proc + 1, dtype=np.int32)
        for i in range(num_proc):
            subset = dataset.select(range(split_idx[i], split_idx[i + 1]))
            async_results.append(pool.apply_async(_map_mp_single, args=(subset, map_func, queue, split_idx[i])))
        while True:
            try:
                yield queue.get(timeout=0.05)
            except Empty:
                if all(async_result.ready() for async_result in async_results) and queue.empty():
                    break


def _map_mp(dataset: HfDataset, map_func: MapFunc, num_proc: int) -> List[Dict[str, Any]]:
    # Solving the unordered problem
    data = [None] * len(dataset)
    num_proc = min(num_proc, len(dataset))
    for d in tqdm(_map_mp_i(dataset, map_func, num_proc), total=len(dataset)):
        data[d[0]] = d[1]
    return data


def dataset_map(dataset: DATASET_TYPE,
                map_func: MapFunc,
                num_proc: int = 1,
                streaming: bool = False) -> Optional[Union[LLMDataset, DATASET_TYPE]]:
    """Map and tokenize a dataset
    This function is used because datasets.map has a critical type checking, which is annoying.

    Args:
        dataset: The dataset instance
        map_func: The map(tokenize) function
        num_proc: Num proc to use
        streaming: In streaming mode

    Returns:

    """
    if streaming:
        return LLMIterableDataset(dataset.map(map_func))  # num_proc is not supported for IterableDataset

    single_map = partial(_single_map, map_func=map_func)
    if num_proc == 1:
        data = []
        for d in tqdm(dataset):
            d = single_map(d)
            data.append(d)
    else:
        assert num_proc > 1
        data = _map_mp(dataset, single_map, num_proc)
    data = [d for d in data if d is not None]
    if len(data) == 0:
        logger.warning('len(dataset): 0')
        return None
    return LLMDataset(data)


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


def safe_tokenizer_decode(tokenizer: PreTrainedTokenizerBase, input_ids: List[int], **tokenizer_kwargs) -> str:

    def _is_special(token: int) -> bool:
        if token < 0:
            return True
        if hasattr(tokenizer, 'placeholder_tokens'):
            return token in tokenizer.placeholder_tokens_id
        return False

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if len(input_ids) == 0:
        return ''
    result_str = ''
    for i in range(len(input_ids)):
        if i == 0:
            if _is_special(input_ids[i]):
                s = 0
            else:
                e = 0
            continue
        if _is_special(input_ids[i]) and not _is_special(input_ids[i - 1]):
            s = i
            result_str += tokenizer.decode(input_ids[e:s], **tokenizer_kwargs)
        if not _is_special(input_ids[i]) and _is_special(input_ids[i - 1]):
            e = i
            result_str += f'[{input_ids[i - 1]} * {e - s}]'
    if _is_special(input_ids[i]):
        result_str += f'[{input_ids[i]} * {len(input_ids) - s}]'
    else:
        result_str += tokenizer.decode(input_ids[e:], **tokenizer_kwargs)
    return result_str


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