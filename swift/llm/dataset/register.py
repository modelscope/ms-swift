# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import json

from swift.utils import get_logger, use_hf_hub
from .preprocessor import DATASET_TYPE, AutoPreprocessor, MessagesPreprocessor

PreprocessFunc = Callable[..., DATASET_TYPE]
LoadFunction = Callable[..., DATASET_TYPE]
logger = get_logger()


@dataclass
class SubsetDataset:
    # `Name` is used for matching subsets of the dataset, and `subset` refers to the subset_name on the hub.
    name: Optional[str] = None
    # If set to None, then subset is set to subset_name.
    subset: str = 'default'

    # Higher priority. If set to None, the attributes of the DatasetMeta will be used.
    split: Optional[List[str]] = None
    preprocess_func: Optional[PreprocessFunc] = None

    # If the dataset specifies "all," weak subsets will be skipped.
    is_weak_subset: bool = False

    def __post_init__(self):
        if self.name is None:
            self.name = self.subset

    def set_default(self, dataset_meta: 'DatasetMeta') -> 'SubsetDataset':
        subset_dataset = deepcopy(self)
        for k in ['split', 'preprocess_func']:
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
    load_function: Optional[LoadFunction] = None

    tags: List[str] = field(default_factory=list)
    help: Optional[str] = None
    huge_dataset: bool = False

    def __post_init__(self):
        from .loader import DatasetLoader
        if self.load_function is None:
            self.load_function = DatasetLoader.load
        for i, subset in enumerate(self.subsets):
            if isinstance(subset, str):
                self.subsets[i] = SubsetDataset(subset=subset)


DATASET_MAPPING: Dict[Tuple[str, str, str], DatasetMeta] = {}


def get_dataset_list():
    datasets = []
    for key in DATASET_MAPPING:
        if use_hf_hub():
            if key[1]:
                datasets.append(key[1])
        else:
            if key[0]:
                datasets.append(key[0])
    return datasets


def register_dataset(dataset_meta: DatasetMeta, *, exist_ok: bool = False) -> None:
    """Register dataset

    Args:
        dataset_meta: The `DatasetMeta` info of the dataset.
        exist_ok: If the dataset id exists, raise error or update it.
    """
    dataset_id = dataset_meta.ms_dataset_id, dataset_meta.hf_dataset_id, dataset_meta.dataset_path
    if not exist_ok and dataset_id in DATASET_MAPPING:
        raise ValueError(f'The `{dataset_id}` has already been registered in the DATASET_MAPPING.')

    DATASET_MAPPING[dataset_id] = dataset_meta


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


def _register_d_info(d_info: Dict[str, Any], *, base_dir: Optional[str] = None) -> DatasetMeta:
    """Register a single dataset to dataset mapping

    Args:
        d_info: The dataset info
    """
    d_info = _preprocess_d_info(d_info, base_dir=base_dir)
    dataset_meta = DatasetMeta(**d_info)
    register_dataset(dataset_meta)
    return dataset_meta


def register_dataset_info(dataset_info: Union[str, List[str], None] = None) -> List[DatasetMeta]:
    """Register dataset from the `dataset_info.json` or a custom dataset info file
    This is used to deal with the datasets defined in the json info file.

    Args:
        dataset_info: The dataset info path
    """
    # dataset_info_path: path, json or None
    if dataset_info is None:
        dataset_info = os.path.join(os.path.dirname(__file__), 'data', 'dataset_info.json')
    assert isinstance(dataset_info, (str, list))
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
        return []
    res = []
    for d_info in dataset_info:
        res.append(_register_d_info(d_info, base_dir=base_dir))

    if log_msg is None:
        log_msg = dataset_info if len(dataset_info) < 5 else list(dataset_info.keys())
    logger.info(f'Successfully registered `{log_msg}`')
    return res
