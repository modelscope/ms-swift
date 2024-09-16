# Copyright (c) Alibaba, Inc. and its affiliates.
import itertools
import json
import os
from copy import deepcopy
from functools import partial
from typing import List, Literal, Optional, Tuple, Union, Dict, Any, Callable

import numpy as np
from datasets import Dataset as HfDataset, IterableDataset as HfIterableDataset
from datasets import concatenate_datasets, interleave_datasets
from numpy.random import RandomState
from swift.llm.utils.dataset import get_local_dataset
from tqdm import tqdm

from swift.hub.hub import HFHub, MSHub
from swift.llm.dataset.preprocess import RowPreprocessor, RenameColumnsPreprocessor, ConversationsPreprocessor, \
    SmartPreprocessor
from swift.utils import safe_ddp_context, get_seed
from swift.utils import get_logger

DATASET_TYPE = Union[HfDataset, HfIterableDataset]

SubsetSplit = Union[str, Tuple[str, str], List[str]]
DATASET_MAPPING: Dict[str, Dict[str, Any]] = {}
PreprocessFunc = Callable[[DATASET_TYPE], DATASET_TYPE]
logger = get_logger()


def register_dataset(dataset_name: str,
                     dataset_id_or_path: Optional[str] = None,
                     subsets: Optional[List[str]] = None,
                     preprocess_func: Optional[PreprocessFunc] = None,
                     get_function: Optional[Callable] = None,
                     *,
                     split: Optional[List[str]] = None,
                     hf_dataset_id: Optional[str] = None,
                     function_kwargs: Optional[Dict[str, Any]] = None,
                     exist_ok: bool = False,
                     is_local: bool = False,
                     **kwargs) -> Optional[Callable]:
    """Register dataset to the dataset mapping

    Args:
        dataset_name: The dataset code
        dataset_id_or_path: The ms dataset id or dataset file path
        subsets: The subsets of the dataset id
        preprocess_func: The preprocess function
        get_function: How to get this dataset, normally it's `get_dataset_from_repo`
        split: The dataset split
        hf_dataset_id: The hf dataset id
        function_kwargs: Extra kwargs passed to `get_dataset_from_repo`
        exist_ok: If the dataset_name exists, whether to raise an error or just override the record, default `False`
        is_local: If is a local dataset
    Returns:
        The dataset instance.
    """
    if preprocess_func is None:
        preprocess_func = SmartPreprocessor()
    if not exist_ok and dataset_name in DATASET_MAPPING:
        raise ValueError(f'The `{dataset_name}` has already been registered in the DATASET_MAPPING.')
    if subsets is None:
        subsets = []
    if split is None:
        split = ['train']
    if function_kwargs is None:
        function_kwargs = {}

    dataset_info = {
        'dataset_id_or_path': dataset_id_or_path,
        'subsets': subsets,
        'preprocess_func': preprocess_func,
        'split': split,
        'hf_dataset_id': hf_dataset_id,
        'is_local': is_local,
        **kwargs
    }
    if get_function is not None:
        if len(function_kwargs) > 0:
            get_function = partial(get_function, **function_kwargs)
        dataset_info['get_function'] = get_function
        DATASET_MAPPING[dataset_name] = dataset_info
        return

    def _register_dataset(get_function: Callable) -> Callable:
        _old_get_function = get_function
        if len(function_kwargs) > 0:
            get_function = partial(get_function, **function_kwargs)
        dataset_info['get_function'] = get_function
        DATASET_MAPPING[dataset_name] = dataset_info
        return _old_get_function

    return _register_dataset


def register_local_dataset(
        dataset_name: str,
        dataset_path: Optional[List[str]] = None,
        # Convert relative path to absolute path
        base_dir: Optional[str] = None,
        **kwargs) -> None:
    if dataset_path is None:
        dataset_path = []
    elif isinstance(dataset_path, str):
        dataset_path = [dataset_path]
    assert len(dataset_path) > 0
    if base_dir is not None:
        for i, path in enumerate(dataset_path):
            if not os.path.isabs(path):
                dataset_path[i] = os.path.join(base_dir, dataset_path[i])

    register_dataset(
        dataset_name, get_function=get_local_dataset, split=dataset_path, exist_ok=True, is_local=True, **kwargs)


def register_single_dataset(dataset_name: str, d_info: Dict[str, Any], **kwargs) -> None:
    """Register dataset to dataset mapping

    Args:
        dataset_name: The dataset name
        d_info: The dataset info

    """
    if 'columns' in d_info:
        preprocess_func = RenameColumnsPreprocessor(d_info['columns'])
        d_info.pop('columns')
        d_info['preprocess_func'] = preprocess_func
    elif 'conversations' in d_info:
        preprocess_func = ConversationsPreprocessor(**d_info['conversations'])
        d_info.pop('conversations')
        d_info['preprocess_func'] = preprocess_func

    if 'dataset_path' in d_info:
        base_dir = kwargs.pop('base_dir', None)
        register_local_dataset(dataset_name, d_info.pop('dataset_path', None), base_dir, **d_info)
        return

    assert 'dataset_id' in d_info or 'hf_dataset_id' in d_info

    dataset_id = d_info.pop('dataset_id', None)
    subsets = d_info.pop('subsets', None)
    preprocess_func = d_info.pop('preprocess_func', None)
    register_dataset(dataset_name, dataset_id, subsets, preprocess_func, get_dataset_from_repo, **d_info,
                     exist_ok=True)


def register_dataset_from_file(dataset_info_path: Optional[str] = None) -> None:
    """Register dataset from the `dataset_info.json` or a custom dataset info file

    Args:
        dataset_info_path: The dataset info path
    """
    # dataset_info_path: path, json or None
    if dataset_info_path is None:
        dataset_info_path = os.path.abspath(os.path.join(__file__, '..', '..', 'data', 'dataset_info.json'))
    if isinstance(dataset_info_path, str):
        if os.path.isfile(dataset_info_path):
            with open(dataset_info_path, 'r') as f:
                dataset_info = json.load(f)
            base_dir = os.path.dirname(dataset_info_path)
        else:
            dataset_info = json.loads(dataset_info_path)
            dataset_info_path = list(dataset_info.keys())
            base_dir = None
    else:
        assert isinstance(dataset_info_path, dict)
        dataset_info = deepcopy(dataset_info_path)
        dataset_info_path = list(dataset_info.keys())
        base_dir = None
    for dataset_name, d_info in dataset_info.items():
        register_single_dataset(dataset_name, d_info, base_dir=base_dir)
    logger.info(f'Successfully registered `{dataset_info_path}`')


register_dataset_from_file()