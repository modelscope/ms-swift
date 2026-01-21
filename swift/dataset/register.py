# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import json

from swift.utils import get_logger, use_hf_hub
from .dataset_meta import DATASET_MAPPING, DatasetMeta, SubsetDataset
from .preprocessor import AutoPreprocessor, MessagesPreprocessor

logger = get_logger()


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
    if dataset_meta.dataset_name:
        dataset_name = dataset_meta.dataset_name
    else:
        dataset_name = dataset_meta.ms_dataset_id, dataset_meta.hf_dataset_id, dataset_meta.dataset_path
    if not exist_ok and dataset_name in DATASET_MAPPING:
        raise ValueError(f'The `{dataset_name}` has already been registered in the DATASET_MAPPING.')

    DATASET_MAPPING[dataset_name] = dataset_meta


def _preprocess_d_info(d_info: Dict[str, Any], *, base_dir: Optional[str] = None) -> Dict[str, Any]:
    d_info = deepcopy(d_info)

    columns = None
    if 'columns' in d_info:
        columns = d_info.pop('columns')

    if 'messages' in d_info:
        d_info['preprocess_func'] = MessagesPreprocessor(**d_info.pop('messages'), columns=columns)
    else:
        d_info['preprocess_func'] = AutoPreprocessor(columns=columns)

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
            with open(dataset_path, 'r', encoding='utf-8') as f:
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
    logger.info(f'Successfully registered `{log_msg}`.')
    return res
