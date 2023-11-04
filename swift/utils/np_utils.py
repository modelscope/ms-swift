# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Union, Tuple

import numpy as np
from numpy import ndarray
from torch import Tensor
from numpy.random import RandomState
from pandas import DataFrame


def transform_jsonl_to_df(dict_list: List[Dict[str, Any]]) -> DataFrame:
    """Relevant function: `io_utils.read_from_jsonl()`"""
    data_dict: Dict[str, List[Any]] = {}
    for i, obj in enumerate(dict_list):
        for k, v in obj.items():
            if k not in data_dict:
                data_dict[k] = [None] * i
            data_dict[k].append(v)
    return DataFrame.from_dict(data_dict)


def get_seed(random_state: RandomState) -> int:
    seed_max = np.iinfo(np.int32).max
    seed = random_state.randint(0, seed_max)
    return seed


def stat_array(
        array: Union[ndarray, List[int], Tensor]) -> Tuple[Dict[str, float], str]:
    if isinstance(array, list):
        array = np.array(array)
    mean = array.mean().item()
    std = array.std().item()
    min_ = array.min().item()
    max_ = array.max().item()
    string = f'{mean:.6f}±{std:.6f}, min={min_:.6f}, max={max_:.6f}, size={array.shape[0]}'
    return {'mean': mean, 'std': std, 'min': min_, 'max': max_}, string
