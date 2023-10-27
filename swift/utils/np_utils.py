# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List

import numpy as np
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
