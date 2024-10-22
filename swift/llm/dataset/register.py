# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import json
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset

from swift.utils import get_logger
from .preprocess import ConversationsPreprocessor, RenameColumnsPreprocessor, SmartPreprocessor

DATASET_TYPE = Union[HfDataset, HfIterableDataset]

PreprocessFunc = Callable[[DATASET_TYPE], DATASET_TYPE]
logger = get_logger()

DATASET_MAPPING: Dict[str, Dict[str, Any]] = {}


@dataclass
class SubsetDataset:
    subset_name: str = 'default'

    # Higher priority. If set to None, the attributes of the Dataset will be used.
    split: Optional[List[str]] = None
    columns_mapping: Optional[Dict[str, Any]] = None
    preprocess_func: Optional[PreprocessFunc] = None
    remove_useless_columns: Optional[bool] = None

    # If the dataset_name does not specify subsets, this parameter determines whether the dataset is used.
    is_weak_subset: bool = False

    def set_default(self, dataset_meta: 'DatasetMeta') -> 'SubsetDataset':
        subset_dataset = deepcopy(self)
        for k in ['split', 'columns_mapping', 'preprocess_func', 'remove_useless_columns']:
            v = getattr(subset_dataset, k)
            if v is None:
                setattr(subset_dataset, k, deepcopy(getattr(dataset_meta, k)))
        return subset_dataset


@dataclass
class DatasetMeta:
    ms_dataset_id: Optional[str] = None
    hf_dataset_id: Optional[str] = None
    dataset_path: Optional[str] = None
    ms_revision: Optional[str] = None
    hf_revision: Optional[str] = None

    subsets: List[SubsetDataset] = field(default_factory=lambda: [SubsetDataset()])
    # Applicable to all subsets.
    split: List[str] = field(default_factory=lambda: ['train'])
    # First perform column mapping, then proceed with the preprocess_func.
    columns_mapping: Dict[str, Any] = field(default_factory=dict)
    preprocess_func: PreprocessFunc = field(default_factory=lambda: SmartPreprocessor())
    remove_useless_columns: bool = True

    tags: List[str] = field(default_factory=list)
    help: Optional[str] = None
    huge_dataset: bool = False


LoadFunction = Callable[..., Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]]


def register_dataset(dataset_name: str,
                     dataset_meta: DatasetMeta,
                     load_function: Optional[LoadFunction] = None,
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
    from .loader import DatasetLoader
    if not exist_ok and dataset_name in DATASET_MAPPING:
        raise ValueError(f'The `{dataset_name}` has already been registered in the DATASET_MAPPING.')
    if function_kwargs is None:
        function_kwargs = {}

    dataset_info = {'dataset_meta': dataset_meta, **kwargs}
    if load_function is None:
        load_function = DatasetLoader.load
    if len(function_kwargs) > 0:
        load_function = partial(load_function, **function_kwargs)
    dataset_info['load_function'] = load_function
    DATASET_MAPPING[dataset_name] = dataset_info


def _preprocess_d_info(d_info: Dict[str, Any]) -> Dict[str, Any]:
    d_info = deepcopy(d_info)
    if 'conversations' in d_info:
        preprocess_func = ConversationsPreprocessor(**d_info.pop('conversations'))
        d_info['preprocess_func'] = preprocess_func

    if 'columns' in d_info:
        d_info['columns_mapping'] = d_info.pop('columns')

    if 'dataset_path' in d_info:
        dataset_path = d_info.pop('dataset_path')
        if base_dir is not None and not os.path.isabs(dataset_path):
            dataset_path = os.path.join(base_dir, dataset_path)
        dataset_path = os.path.abspath(os.path.expanduser(dataset_path))

        d_info['dataset_path'] = dataset_path

    if 'subsets' in d_info:
        subsets = d_info.pop('subsets')
        new_subsets = []
        for subset in subsets:
            if isinstance(subset, str):
                new_subsets.append(SubsetDataset(subset_name=subset))
            elif isinstance(subset, dict):
                new_subsets.append(SubsetDataset(**_preprocess_d_info(subset)))
        d_info['subsets'] = new_subsets
    return d_info


def _register_d_info(dataset_name: str, d_info: Dict[str, Any], *, base_dir: Optional[str] = None) -> None:
    """Register a single dataset to dataset mapping

    Args:
        dataset_name: The dataset name
        d_info: The dataset info
    """
    d_info = _preprocess_d_info(d_info)
    register_dataset(dataset_name, DatasetMeta(**d_info))


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
    if len(dataset_info) == 0:
        return
    for dataset_name, d_info in dataset_info.items():
        _register_d_info(dataset_name, d_info, base_dir=base_dir)
    logger.info(f'Successfully registered `{log_msg}`')
