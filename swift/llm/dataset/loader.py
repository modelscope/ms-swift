# Copyright (c) Alibaba, Inc. and its affiliates.
import itertools
import os
import shutil
from abc import ABC
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from datasets import concatenate_datasets, interleave_datasets
from modelscope.hub.api import ModelScopeConfig
from modelscope.utils.config_ds import MS_CACHE_HOME
from numpy.random import RandomState
from pandas import DataFrame

from swift.hub import HFHub, MSHub, default_hub
from swift.utils import download_ms_file, get_logger, get_seed, safe_ddp_context, use_hf_hub
from .preprocess import RowPreprocessor
from .register import Dataset, register_dataset_info, DATASET_MAPPING, SubsetDataset

DATASET_TYPE = Union[HfDataset, HfIterableDataset]

SubsetSplit = Union[str, Tuple[str, str], List[str]]
PreprocessFunc = Callable[[DATASET_TYPE], DATASET_TYPE]
logger = get_logger()


@dataclass
class DatasetSyntax:
    dataset: str
    use_hf: Optional[bool] = None
    subsets: List[str] = field(default_factory=list)
    dataset_sample: Optional[int] = None

    def __post_init__(self):
        if self.use_hf is None:
            self.use_hf = use_hf_hub()
        if self.dataset in DATASET_MAPPING:
            self.dataset_type = 'name'
        elif os.path.isfile(self.dataset) or self.dataset.startswith('/'):
            self.dataset_type = 'path'
            assert os.path.isfile(self.dataset)
        elif self.use_hf:
            self.dataset_type = 'hf_repo'
        else:
            self.dataset_type = 'ms_repo'

    @classmethod
    def parse(cls, dataset: str) -> 'DatasetSyntax':
        """Parse the dataset from the command line"""
        # HF::dataset_name:subset1/subset2/subset3#dataset_sample
        use_hf, other = DatasetLoader._safe_split(dataset, '::', False)
        if os.path.isfile(other):
            part1, dataset_sample = other, None
        else:
            part1, dataset_sample = DatasetLoader._safe_split(other, '#', True, 'right')
        if os.path.isfile(part1):
            dataset, subsets = part1, None
        else:
            dataset, subsets = DatasetLoader._safe_split(part1, ':', True)

        dataset_name = dataset.strip()
        if use_hf is not None:
            use_hf = {'ms': False, 'hf': True}[use_hf.strip().lower()]
        if subsets is not None:
            subsets = [subset.strip() for subset in subsets.split('/')]
        if dataset_sample is not None:
            dataset_sample = int(dataset_sample)
        return cls(dataset_name, use_hf, subsets, dataset_sample)

    def to_dict(self):
        # Convert to a format that can be parsed by register_dataset_info.
        assert self.dataset_type != 'name'
        res = {}
        mapping = {'path': 'dataset_path', 'hf_repo': 'hf_dataset_id', 'ms_repo': 'ms_dataset_id'}
        key = mapping[self.dataset_type]
        res[key] = self.dataset
        return res


class DatasetNameMapping:
    # dataset_id/path -> dataset_name
    def __init__(self):
        self._init_mapping()

    def _init_mapping(self) -> None:
        dataset_name_mapping = {}
        for dataset_name, v in DATASET_MAPPING.items():
            dataset: Dataset = v['dataset']
            if len(dataset.dataset_path) > 0:
                dataset_name_mapping[self._encode_key(dataset.dataset_path, 'path')] = dataset_name
            else:
                if dataset.ms_dataset_id is not None:
                    k = self._encode_key(dataset.ms_dataset_id, 'ms_repo')
                    assert k not in dataset_name_mapping
                    dataset_name_mapping[k] = dataset_name
                if dataset.hf_dataset_id is not None:
                    k = self._encode_key(dataset.hf_dataset_id, 'hf_repo')
                    assert k not in dataset_name_mapping
                    dataset_name_mapping[k] = dataset_name
        self.mapping = dataset_name_mapping

    def _encode_key(self, d_id_or_path: Union[str, List[str]], dataset_type: Literal['hf_repo', 'ms_repo', 'path']):
        assert dataset_type != 'name'
        if dataset_type == 'hf_repo':
            return f'HF::{d_id_or_path}'
        elif dataset_type == 'ms_repo':
            return f'MS::{d_id_or_path}'
        else:
            if isinstance(d_id_or_path, str):
                return (d_id_or_path, )
            else:
                return tuple(d_id_or_path)

    def map_to_name(self, dataset_info: DatasetSyntax) -> Optional[str]:
        key = self._encode_key(dataset_info.dataset, dataset_info.dataset_type)
        return self.mapping.get(key)


class DatasetLoader:

    @staticmethod
    def _load_local_dataset(
            dataset_path: List[str], *,
            columns_mapping: Dict[str, Any],
            preprocess_func: PreprocessFunc,
            #
            dataset_sample: int,
            split_dataset_ratio: float,
            dataset_seed: RandomState,
            num_proc: int,
            streaming: bool,
            **kwargs
    ):
        pass

    @staticmethod
    def _load_repo_dataset(
            dataset_id: str,
            *,
            use_hf: bool,
            subsets: List[SubsetDataset],
            split: List[str],
            columns_mapping: Dict[str, Any],
            preprocess_func: PreprocessFunc,
            #
            split_dataset_ratio: float,
            dataset_seed: RandomState,
            num_proc: int,
            streaming: bool,
            **kwargs
    ):
        pass

        self.split_dataset_ratio = split_dataset_ratio
        self.dataset_seed: RandomState = dataset_seed
        self.use_hf = use_hf
        self.load_from_cache_file = load_from_cache_file
        self.num_proc = num_proc
        self.force_redownload = force_redownload

        self.model_name = model_name
        self.model_author = model_author

        self.streaming = streaming
        self.streaming_val_size = streaming_val_size
        self.streaming_buffer_size = streaming_buffer_size

    def _select_subsets(self):
        pass


    @staticmethod
    def load(dataset: str, *,
             dataset_syntax: DatasetSyntax,
             dataset_info: Dataset,
             ) -> Tuple[HfDataset, Optional[HfDataset]]:

        dataset_name = dataset_syntax.dataset
        subsets = dataset_syntax.subsets
        subset_names = [subset.subset_name for subset in dataset_info.subsets]
        if not subsets:
            if len(subsets) > 1:
                subsets = ['default']
        elif len(subsets) == 1 and subsets[0] == 'all' and 'all' not in subset_names:
            subsets = subset_names

        use_hf = dataset_syntax.use_hf
        dataset_id_or_path = dataset_info.dataset_path
        if dataset_id_or_path:
            dataset = DatasetLoader._load_local_dataset()
        else:
            dataset_str_f = 'Downloading the dataset from {hub}, dataset_id: {dataset_id}'
            if use_hf:
                dataset_id_or_path = dataset_info.hf_dataset_id
                dataset_str = dataset_str_f.format(hub='HuggingFace', dataset_id=dataset_id_or_path)
            else:
                dataset_id_or_path = dataset_info.ms_dataset_id
                dataset_str = dataset_str_f.format(hub='ModelScope', dataset_id=dataset_id_or_path)
            logger.info(dataset_str)
            assert dataset_id_or_path is not None, (f'dataset_name: {dataset_name}, use_hf: {use_hf}, '
                                                    f'dataset_id_or_path: {dataset_id_or_path}.')
            dataset = DatasetLoader._load_repo_dataset()

        load_function(dataset_info)


    @staticmethod
    def _safe_split(s: str,
                    sep: str,
                    use_0: bool,
                    split_mode: Literal['left', 'right'] = 'left') -> Tuple[Optional[str], Optional[str]]:
        """
        use_0: When the length of the part is 1, is it considered as part0 or part1.
        split_mode: use split or rsplit
        """
        if s is None or len(s) == 0:
            return None, None
        if split_mode == 'left':
            part = s.split(sep, 1)
        else:
            part = s.rsplit(sep, 1)
        if len(part) == 1:
            if use_0:
                part = part[0], None
            else:
                part = None, part[0]
        else:
            assert len(part) == 2
        return part

    @staticmethod
    def _parse_datasets(datasets: List[str]) -> List[str]:
        # ms_dataset_id/hf_dataset_id/dataset_path -> dataset_name mapping
        dataset_name_mapping = DatasetNameMapping()

        # register_dataset
        res_datasets: List[str] = []  # dataset_names
        register_idx = 0
        dataset_info = {}
        for dataset in datasets:
            d_info = DatasetSyntax.parse(dataset)
            if d_info.dataset_type == 'name':
                res_datasets.append(d_info.dataset)
            else:
                # dataset_path/dataset_id
                dataset_name = dataset_name_mapping.map_to_name(d_info)
                res_datasets.append(dataset.replace(d_info.dataset, dataset_name))
                if dataset_name is None:
                    # This dataset needs to be registered.
                    dataset_info[f'_{register_idx}'] = d_info.to_dict()
                    register_idx += 1
        register_dataset_info(dataset_info)

        return res_datasets


def load_dataset(
        datasets: List[str],
        split_dataset_ratio: float = 0.,
        dataset_seed: Union[int, RandomState] = 42,
        *,
        use_hf: Optional[bool] = None,
        load_from_cache_file: bool = False,
        num_proc: int = 1,
        force_redownload: bool = False,
        # self-cognition
        model_name: Union[Tuple[str, str], List[str], None] = None,
        model_author: Union[Tuple[str, str], List[str], None] = None,
        # streaming
        streaming: bool = False,
        streaming_val_size: int = 0,
        streaming_buffer_size: int = 16384) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
    """The interface to load any registered dataset

    Args:
        datasets: The dataset name list
        split_dataset_ratio: The dataset split ratio
        dataset_seed: The dataset random seed
        model_name: Model name in self-cognition task
        model_author: Model author in self-cognition task
        streaming: Streaming mode or not
    Returns:
        The train dataset and val dataset
    """
    if isinstance(datasets, str):
        datasets = [datasets]

    datasets: List[str] = DatasetLoader._parse_datasets(datasets)  # to dataset_names and register
    train_datasets = []
    val_datasets = []
    load_kwargs = {
        'split_dataset_ratio': split_dataset_ratio,
        'dataset_seed': dataset_seed,
        'use_hf': use_hf,
        'load_from_cache_file': load_from_cache_file,
        'num_proc': num_proc,
        'force_redownload': force_redownload,
        'model_name': model_name,
        'model_author': model_author,
        'streaming': streaming,
        'streaming_val_size': streaming_val_size,
        'streaming_buffer_size': streaming_buffer_size
    }
    for dataset in datasets:
        d_info = DatasetSyntax.parse(dataset)
        assert d_info.dataset_type == 'name'
        dataset_name = d_info.dataset
        load_function = DATASET_MAPPING[dataset_name]['load_function']
        if load_function is None:
            load_function = DatasetLoader.load
        kwargs = {
            'dataset_syntax': d_info,
            'dataset_info': DATASET_MAPPING[dataset_name]['dataset'],
        }
        train_dataset, val_dataset = load_function(dataset, **load_kwargs, **kwargs)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    gather_function = interleave_datasets if streaming else concatenate_datasets
    if len(train_datasets) > 1:
        train_datasets = gather_function(train_datasets)
    if len(val_datasets) > 1:
        val_datasets = gather_function(val_datasets)
    return train_datasets, val_datasets

    dataset_loader = DatasetLoader(
        split_dataset_ratio,
        dataset_seed,
        use_hf,
        load_from_cache_file,
        num_proc,
        force_redownload,
        model_name=model_name,
        model_author=model_author,
        streaming=streaming,
        streaming_val_size=streaming_val_size,
        streaming_buffer_size=streaming_buffer_size)
    return dataset_loader.load_dataset(datasets)
