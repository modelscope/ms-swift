# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Callable, Dict, List, Optional

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


def lower_bound(lo: int, hi: int, cond: Callable[[int], bool]) -> int:
    # The lower bound satisfying the condition "cond".
    while lo < hi:
        mid = (lo + hi) >> 1
        if cond(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def print_example(example: Dict[str, Any], tokenizer) -> None:
    input_ids, labels = example['input_ids'], example['labels']
    logger.info(f'[INPUT_IDS] {input_ids}')
    logger.info(f'[INPUT] {tokenizer.decode(input_ids)}')
    n_mask = lower_bound(0, len(labels), lambda i: labels[i] != -100)
    logger.info(f'[LABLES_IDS] {labels}')
    logger.info(
        f'[LABLES] [-100 * {n_mask}]{tokenizer.decode(labels[n_mask:])}')
