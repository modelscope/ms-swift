# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import shutil
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets, interleave_datasets
from modelscope.hub.api import ModelScopeConfig
from modelscope.utils.config_ds import MS_CACHE_HOME

from swift.utils import download_ms_file, get_logger, get_seed, safe_ddp_context
from .preprocessor import DATASET_TYPE, AutoPreprocessor
from .utils import sample_dataset

PreprocessFunc = Callable[..., DATASET_TYPE]
logger = get_logger()

if TYPE_CHECKING:
    from .dataset_syntax import DatasetSyntax


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


class BaseDatasetLoader(ABC):

    @abstractmethod
    def load(
        self,
        dataset_syntax: Optional['DatasetSyntax'] = None,
        dataset_meta: Optional['DatasetMeta'] = None,
        *,
        use_hf: Optional[bool] = None,
    ) -> HfDataset:
        pass

    @staticmethod
    def download_ms_dataset(ms_dataset_id: str, files: List[str], force_download: bool = False) -> str:
        """Download dataset from repo manually
        Args:
            ms_dataset_id: The dataset id of ModelScope
            files: Which files to download
            force_download: Force download or not
        Returns:
            The dataset dir
        """
        assert isinstance(files, list)
        url = f'http://www.modelscope.cn/api/v1/datasets/{ms_dataset_id}/repo?Revision=master&FilePath={{fpath}}'
        cache_dir = os.path.join(MS_CACHE_HOME, 'datasets', ms_dataset_id, 'master')
        local_dir = os.path.join(cache_dir, 'raw')
        tmp_dir = os.path.join(cache_dir, 'tmp')
        os.makedirs(local_dir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)
        cookies = ModelScopeConfig.get_cookies()
        with TemporaryDirectory(dir=tmp_dir) as temp_dir:
            for remote_fpath in files:
                url = url.format(fpath=remote_fpath)
                temp_fpath = os.path.join(temp_dir, remote_fpath)
                local_fpath = os.path.join(local_dir, remote_fpath)
                if not force_download and os.path.exists(local_fpath):
                    continue
                download_ms_file(url, temp_fpath, cookies)
                shutil.copy2(temp_fpath, local_fpath)

        return local_dir

    @staticmethod
    def concat_datasets(datasets: List[HfDataset]) -> Optional[HfDataset]:
        if len(datasets) == 0:
            return
        if len(datasets) == 1:
            return datasets[0]
        return concatenate_datasets(datasets)

    @staticmethod
    def interleave_datasets(datasets, *args, **kwargs):
        if len(datasets) == 0:
            return
        if len(datasets) == 1:
            return datasets[0]
        return interleave_datasets(datasets, *args, **kwargs)

    @staticmethod
    def shuffle_dataset(dataset, seed: int, buffer_size: int = 1000):
        if isinstance(dataset, HfDataset):
            with safe_ddp_context(None, True):
                return dataset.shuffle(seed=seed)
        else:
            return dataset.shuffle(seed=seed, buffer_size=buffer_size)

    @staticmethod
    def post_process(
        train_dataset: DATASET_TYPE,
        *,
        dataset_sample: Optional[int] = None,
        split_dataset_ratio: float = 0.,
        streaming: bool = False,
        shuffle: bool = True,
        random_state: Optional[np.random.RandomState] = None,
    ) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
        """Split into train/val datasets and perform dataset sampling."""
        assert dataset_sample is None or dataset_sample > 0
        assert 0 <= split_dataset_ratio <= 1
        if streaming:
            if dataset_sample is None:
                if split_dataset_ratio == 0:
                    val_dataset = None
                elif split_dataset_ratio == 1:
                    train_dataset, val_dataset = None, train_dataset
                else:
                    raise ValueError('The IterableDataset does not support splitting the training set '
                                     'and validation set when dataset_sample is None.')
            else:
                # not shuffle
                train_dataset = train_dataset.take(dataset_sample)
                val_sample = int(dataset_sample * split_dataset_ratio)
                val_dataset = None if val_sample == 0 else train_dataset.take(val_sample)
                if val_sample:
                    train_dataset = train_dataset.skip(val_sample)
        else:
            if dataset_sample is None:
                dataset_sample = len(train_dataset)
            if split_dataset_ratio == 0:
                train_dataset = sample_dataset(train_dataset, dataset_sample, shuffle, random_state)
                val_dataset = None
            elif split_dataset_ratio == 1:
                train_dataset, val_dataset = None, train_dataset
                val_sample = dataset_sample
                # Avoid duplication in the val_dataset.
                assert val_sample <= len(val_dataset), f'val_sample: {val_sample}, len(val_dataset): {len(val_dataset)}'
                val_dataset = sample_dataset(val_dataset, val_sample, shuffle, random_state)
            else:
                # Avoid duplication in the val_dataset.
                train_len = min(len(train_dataset), dataset_sample)
                val_sample = max(int(train_len * split_dataset_ratio), 1)
                train_sample = dataset_sample - val_sample
                assert train_sample > 0
                with safe_ddp_context(None, True):
                    train_dataset, val_dataset = train_dataset.train_test_split(
                        test_size=val_sample, shuffle=shuffle, seed=get_seed(random_state)).values()
                train_dataset = sample_dataset(train_dataset, train_sample, shuffle, random_state)
        return train_dataset, val_dataset


@dataclass
class DatasetMeta:
    ms_dataset_id: Optional[str] = None
    hf_dataset_id: Optional[str] = None
    dataset_path: Optional[str] = None  # or dataset_dir
    dataset_name: Optional[str] = None
    ms_revision: Optional[str] = None
    hf_revision: Optional[str] = None

    subsets: List[Union[SubsetDataset, str]] = field(default_factory=lambda: ['default'])
    # Applicable to all subsets.
    split: List[str] = field(default_factory=lambda: ['train'])
    # First perform column mapping, then proceed with the preprocess_func.
    preprocess_func: PreprocessFunc = field(default_factory=lambda: AutoPreprocessor())
    loader: Optional[BaseDatasetLoader] = None

    tags: List[str] = field(default_factory=list)
    help: Optional[str] = None
    huge_dataset: bool = False

    def __post_init__(self):
        from .loader import DatasetLoader
        if self.loader is None:
            self.loader = DatasetLoader
        for i, subset in enumerate(self.subsets):
            if isinstance(subset, str):
                self.subsets[i] = SubsetDataset(subset=subset)


DATASET_MAPPING: Dict[Tuple[str, str, str], DatasetMeta] = {}
