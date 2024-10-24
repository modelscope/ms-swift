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
from .preprocess import AutoPreprocessor, MessagesPreprocessor

DATASET_TYPE = Union[HfDataset, HfIterableDataset]

PreprocessFunc = Callable[..., DATASET_TYPE]
LoadFunction = Callable[..., DATASET_TYPE]
logger = get_logger()

DATASET_MAPPING: Dict[str, Dict[str, Any]] = {}


@dataclass
class SubsetDataset:
    # `Name` is used for matching subsets of the dataset, and `subset` refers to the subset_name on the hub.
    name: str = 'default'
    # If set to None, then subset is set to subset_name.
    subset: Optional[str] = None

    # Higher priority. If set to None, the attributes of the Dataset will be used.
    split: Optional[List[str]] = None
    preprocess_func: Optional[PreprocessFunc] = None
    remove_useless_columns: Optional[bool] = None

    # If the dataset_name does not specify subsets, this parameter determines whether the dataset is used.
    is_weak_subset: bool = False

    def __post_init__(self):
        if self.subset is None:
            self.subset = self.name

    def set_default(self, dataset_meta: 'DatasetMeta') -> 'SubsetDataset':
        subset_dataset = deepcopy(self)
        for k in ['split', 'preprocess_func', 'remove_useless_columns']:
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

    subsets: List[Union[SubsetDataset, str]] = field(default_factory=lambda: [SubsetDataset()])
    # Applicable to all subsets.
    split: List[str] = field(default_factory=lambda: ['train'])
    # First perform column mapping, then proceed with the preprocess_func.
    preprocess_func: PreprocessFunc = field(default_factory=lambda: AutoPreprocessor())
    remove_useless_columns: bool = True

    tags: List[str] = field(default_factory=list)
    help: Optional[str] = None
    huge_dataset: bool = False

    def __post_init__(self):
        for i, subset in enumerate(self.subsets):
            if isinstance(subset, str):
                self.subsets[i] = SubsetDataset(name=subset)


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


def _preprocess_d_info(d_info: Dict[str, Any], *, base_dir: Optional[str] = None) -> Dict[str, Any]:
    d_info = deepcopy(d_info)

    columns_mapping = None
    if 'columns' in d_info:
        columns_mapping = d_info.pop('columns')

    if 'messages' in d_info:
        d_info['preprocess_func'] = MessagesPreprocessor(**d_info.pop('messages'), columns_mapping=columns_mapping)
    else:
        d_info['preprocess_func'] = AutoPreprocessor(columns_mapping=columns_mapping)

    if 'dataset_path' in d_info:
        dataset_path = d_info.pop('dataset_path')
        if base_dir is not None and not os.path.isabs(dataset_path):
            dataset_path = os.path.join(base_dir, dataset_path)
        dataset_path = os.path.abspath(os.path.expanduser(dataset_path))

        d_info['dataset_path'] = dataset_path

    if 'subsets' in d_info:
        subsets = d_info.pop('subsets')
        for i, subset in enumerate(subsets):
            if isinstance(subset, dict):
                subsets[i] = SubsetDataset(**_preprocess_d_info(subset))
        d_info['subsets'] = subsets
    return d_info


def _register_d_info(dataset_name: str, d_info: Dict[str, Any], *, base_dir: Optional[str] = None) -> None:
    """Register a single dataset to dataset mapping

    Args:
        dataset_name: The dataset name
        d_info: The dataset info
    """
    d_info = _preprocess_d_info(d_info, base_dir=base_dir)
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
    assert isinstance(dataset_info, (str, dict))
    base_dir = None
    log_msg = None
    if isinstance(dataset_info, str):
        dataset_path = os.path.abspath(os.path.expanduser(dataset_info))
        if os.path.isfile(dataset_path):
            log_msg = dataset_path
            base_dir = os.path.dirname(dataset_path)
            with open(dataset_path, 'r') as f:
                dataset_info = json.load(f)
        else:
            dataset_info = json.loads(dataset_info)  # json
    if len(dataset_info) == 0:
        return
    for dataset_name, d_info in dataset_info.items():
        _register_d_info(dataset_name, d_info, base_dir=base_dir)

    if log_msg is None:
        log_msg = dataset_info if len(dataset_info) < 5 else list(dataset_info.keys())
    logger.info(f'Successfully registered `{log_msg}`')
