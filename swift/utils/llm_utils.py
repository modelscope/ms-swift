# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset as HfDataset
from torch.nn.utils.rnn import pad_sequence

from .logger import get_logger

logger = get_logger()


def stat_dataset(dataset: HfDataset) -> None:
    """Statistical analysis was performed on the dataset"""
    _token_len = []
    for d in dataset:
        _token_len.append(len(d['input_ids']))
    _token_len = np.array(_token_len)
    mean = _token_len.mean().item()
    std = _token_len.std().item()
    min_ = _token_len.min().item()
    max_ = _token_len.max().item()
    logger.info(
        f'Dataset Token Length: {mean:.6f}±{std:.6f}, min={min_:.6f}, max={max_:.6f}, size={_token_len.shape[0]}'
    )


def tokenize_function(example: Dict[str,
                                    Optional[str]], tokenizer, prompt: str,
                      max_length: Optional[int]) -> Dict[str, Any]:
    instruction: str = example['instruction']
    output = example.get('output')
    src_text = prompt.format(instruction=instruction)
    src_input_ids: List[int] = tokenizer(
        src_text, return_attention_mask=False,
        add_special_tokens=True)['input_ids']
    if src_input_ids[-1] == tokenizer.eos_token_id:
        src_input_ids.pop()

    tgt_input_ids = []
    if output is not None:
        assert tokenizer.eos_token_id is not None
        tgt_input_ids += tokenizer(
            output, return_attention_mask=False,
            add_special_tokens=False)['input_ids']
        tgt_input_ids += [tokenizer.eos_token_id]
        labels = [-100] * len(src_input_ids) + tgt_input_ids
    else:
        labels = None
    input_ids = src_input_ids + tgt_input_ids

    if max_length is not None:
        input_ids = input_ids[-max_length:]
        if labels is not None:
            labels = labels[-max_length:]

    return {'input_ids': input_ids, 'labels': labels}


def data_collate_fn(batch: List[Dict[str, Any]], tokenizer) -> Dict[str, Any]:
    assert tokenizer.pad_token_id is not None
    input_ids = [torch.tensor(b['input_ids']) for b in batch]
    labels = [torch.tensor(b['labels']) for b in batch]
    attention_mask = [
        torch.ones(len(input_ids[i]), dtype=torch.int64)
        for i in range(len(input_ids))
    ]

    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(
        attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def _count_startswith(arr: List[int], val: int, lo: int = 0) -> int:
    res = 0
    for x in arr[lo:]:
        if x != val:
            break
        res += 1
    return res


def print_example(example: Dict[str, Any], tokenizer) -> None:
    input_ids, labels = example['input_ids'], example['labels']
    logger.info(f'[INPUT_IDS] {input_ids}')
    logger.info(f'[INPUT] {tokenizer.decode(input_ids)}')
    n_mask = _count_startswith(labels, -100)
    logger.info(f'[LABLES_IDS] {labels}')
    logger.info(
        f'[LABLES] [-100 * {n_mask}]{tokenizer.decode(labels[n_mask:])}')
