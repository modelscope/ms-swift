# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import json
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset

from swift.llm.dataset.preprocess import ConversationsPreprocessor, RenameColumnsPreprocessor, SmartPreprocessor
from swift.utils import get_logger
from .loader import DATASET_MAPPING

DATASET_TYPE = Union[HfDataset, HfIterableDataset]

SubsetSplit = Union[str, Tuple[str, str], List[str]]
PreprocessFunc = Callable[[DATASET_TYPE], DATASET_TYPE]
logger = get_logger()


@dataclass
class Dataset:
    ms_dataset_id: Optional[str] = None
    hf_dataset_id: Optional[str] = None
    subsets: Optional[List[str]] = field(default_factory=lambda: ['default'])
    splits: Optional[List[str]] = field(default_factory=lambda: ['train'])

    dataset_path: List[str] = field(default_factory=list)

    columns_mapping: Dict[str, Any] = field(default_factory=dict)
    preprocess_func: Optional[PreprocessFunc] = SmartPreprocessor()


LoadFunction = Callable[..., Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]]


def register_dataset(dataset_name: str,
                     dataset: Dataset,
                     load_function: LoadFunction,
                     *,
                     function_kwargs: Optional[Dict[str, Any]] = None,
                     exist_ok: bool = False,
                     **kwargs) -> None:
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
    if not exist_ok and dataset_name in DATASET_MAPPING:
        raise ValueError(f'The `{dataset_name}` has already been registered in the DATASET_MAPPING.')
    if function_kwargs is None:
        function_kwargs = {}

    dataset_info = {'dataset': dataset, **kwargs}
    if len(function_kwargs) > 0:
        load_function = partial(load_function, **function_kwargs)
    dataset_info['load_function'] = load_function
    DATASET_MAPPING[dataset_name] = dataset_info
    return


def _register_d_info(dataset_name: str, d_info: Dict[str, Any], *, base_dir: Optional[str] = None) -> None:
    """Register a single dataset to dataset mapping

    Args:
        dataset_name: The dataset name
        d_info: The dataset info
    """
    if 'conversations' in d_info:
        preprocess_func = ConversationsPreprocessor(**d_info.pop('conversations'))
    else:
        preprocess_func = SmartPreprocessor()

    if 'dataset_path' in d_info:
        dataset_path = d_info.pop('dataset_path')
        if isinstance(dataset_path, str):
            dataset_path = [dataset_path]
        for i, path in enumerate(dataset_path):
            if base_dir is not None and not os.path.isabs(path):
                dataset_path[i] = os.path.join(base_dir, dataset_path[i])
            dataset_path[i] = os.path.abspath(os.path.expanduser(dataset_path[i]))
        register_dataset(dataset_name, Dataset(dataset_path=dataset_path), **d_info)
    elif 'ms_dataset_id' in d_info or 'hf_dataset_id' in d_info:
        # register dataset from hub
        register_dataset(
            dataset_name,
            Dataset(
                ms_dataset_id=d_info.get('ms_dataset_id'),
                hf_dataset_id=d_info.get('hf_dataset_id'),
                subsets=d_info.get('subsets', ['default']),
                splits=d_info.get('splits', ['train']),
                columns_mapping=d_info.get('columns', {}),
                preprocess_func=preprocess_func), **d_info)
    else:
        raise ValueError(f'd_info: {d_info}')


def register_dataset_info(dataset_info: Union[str, Dict[str, Any], None] = None) -> None:
    """Register dataset from the `dataset_info.json` or a custom dataset info file
    This is used to deal with the datasets defined in the json info file.

    Args:
        dataset_info_path: The dataset info path
    """
    # dataset_info_path: path, json or None
    if dataset_info is None:
        dataset_info = os.path.join(__file__, '..', '..', 'data', 'dataset_info.json')
    base_dir = None
    if isinstance(dataset_info, str):
        dataset_info = os.path.abspath(os.path.expanduser(dataset_info))
        if os.path.isfile(dataset_info):
            log_msg = dataset_info
            base_dir = os.path.dirname(dataset_info)
            with open(dataset_info, 'r') as f:
                dataset_info = json.load(f)
        else:
            # json
            dataset_info = json.loads(dataset_info)
            log_msg = list(dataset_info.keys())
    elif isinstance(dataset_info, dict):
        log_msg = list(dataset_info.keys())
    else:
        raise ValueError(f'dataset_info: {dataset_info}')
    for dataset_name, d_info in dataset_info.items():
        _register_d_info(dataset_name, d_info, base_dir=base_dir)
    logger.info(f'Successfully registered `{log_msg}`')


register_dataset_info()
