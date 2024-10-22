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
from .register import DATASET_MAPPING, DatasetMeta, SubsetDataset, register_dataset_info

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
        return cls(dataset_name, use_hf, subsets or [], dataset_sample)

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
            dataset_meta: DatasetMeta = v['dataset_meta']
            if dataset_meta.dataset_path is not None:
                dataset_name_mapping[self._encode_key(dataset_meta.dataset_path, 'path')] = dataset_name
            else:
                if dataset_meta.ms_dataset_id is not None:
                    k = self._encode_key(dataset_meta.ms_dataset_id, 'ms_repo')
                    assert k not in dataset_name_mapping
                    dataset_name_mapping[k] = dataset_name
                if dataset_meta.hf_dataset_id is not None:
                    k = self._encode_key(dataset_meta.hf_dataset_id, 'hf_repo')
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
            return f'PATH::{d_id_or_path}'

    def map_to_name(self, dataset_info: DatasetSyntax) -> Optional[str]:
        key = self._encode_key(dataset_info.dataset, dataset_info.dataset_type)
        return self.mapping.get(key)

    def _remove_useless_columns(dataset: DATASET_TYPE) -> DATASET_TYPE:
        standard_keys = {'messages', 'rejected_response', 'images', 'objects', 'videos', 'audios', 'tools', 'label'}
        k_list = []
        if isinstance(dataset, HfIterableDataset) and dataset.features is None:
            features = next(iter(dataset)).keys()
        else:
            features = dataset.features.keys()

        for k in features:
            if k in standard_keys:
                k_list.append(k)
        dataset = dataset.select_columns(k_list)
        return dataset

    def _concat_datasets(datasets: List[HfDataset], streaming: bool) -> Optional[HfDataset]:
        if len(datasets) == 0:
            return
        if len(datasets) == 1:
            return datasets[0]
        return interleave_datasets(datasets) if streaming else concatenate_datasets(datasets)

    @staticmethod
    def _load_local_dataset(
            dataset_path: str,
            *,
            columns_mapping: Dict[str, Any],
            preprocess_func: PreprocessFunc,
            remove_useless_columns: bool,
            #
            num_proc: int,
            streaming: bool,
            **kwargs) -> HfDataset:
        pass

    @staticmethod
    def _load_repo_dataset(
        dataset_id: str,
        subset: SubsetDataset,
        use_hf: bool,
        *,
        num_proc: int,
        streaming: bool,
        revision: Optional[str],
        download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'],
    ) -> HfDataset:
        datasets = []
        for split in subset.split:
            if use_hf:
                hub = HFHub
                retry = 1
            else:
                hub = MsHub
                retry = 3
            with safe_ddp_context():
                while True:
                    try:
                        dataset = hub.load_dataset(
                            dataset_id,
                            subset.subset_name,
                            split,
                            streaming=streaming,
                            revision=revision,
                            download_mode=download_mode)
                    except Exception as e:
                        if retry == 0:
                            raise
                        retry -= 1
                        logger.error(f'Dataset {dataset_id} load failed: subset_name={subset_name},'
                                     f'split={split} with error: {e}')
                    else:
                        break
                if streaming and hasattr(dataset, '_hf_ds'):
                    dataset = dataset._hf_ds
                    if not isinstance(dataset, HfIterableDataset):
                        dataset = dataset.to_iterable_dataset()
                if hasattr(dataset, 'to_hf_dataset'):
                    dataset = dataset.to_hf_dataset()
            dataset = subset.preprocess_func(dataset)
            if subset.remove_useless_columns:
                dataset = _remove_useless_columns(dataset)
            datasets.append(dataset)
        return DatasetLoader._concat_datasets(datasets)

    @staticmethod
    def _select_subsets(dataset_syntax: DatasetSyntax, dataset_meta: DatasetMeta) -> List[SubsetDataset]:
        subsets = dataset_syntax.subsets
        subset_mapping = {subset.subset_name: subset for subset in dataset_meta.subsets}
        subset_names = list(subset_mapping.keys())
        if not subsets:
            if len(subset_names) <= 1:
                subsets = subset_names
            else:
                raise ValueError(f'Please provide subsets; available subsets: {subset_names}')
        elif len(subsets) == 1 and subsets[0] == 'all' and 'all' not in subset_names:
            subsets = subset_names
        subsets = [subset_mapping[subset_name].set_default(dataset_meta) for subset_name in subset_names]
        return subsets

    def sample_dataset(dataset: HfDataset,
                       dataset_sample: int,
                       random_state: Optional[RandomState] = None) -> HfDataset:
        """Sample dataset by a dataset_sample number
        Args:
            dataset: The dataset instance, iterable dataset is not supported
            dataset_sample: The sample number
            random_state: The random state
        Returns:
            The sampled dataset
        """
        if random_state is None:
            random_state = RandomState()

        n_repeat_sample = dataset_sample // len(dataset)
        n_random_sample = dataset_sample % len(dataset)
        if n_repeat_sample >= 1 and n_random_sample >= 1:
            logger.info(f'dataset_sample:{dataset_sample} is greater than len(dataset):{len(dataset)}, '
                        'repeated sampling will be performed')
        idx = np.tile(range(len(dataset)), n_repeat_sample)
        if n_random_sample >= 1:
            idx_random = random_state.permutation(len(dataset))[:n_random_sample]
            idx = np.concatenate([idx, idx_random])
        dataset = dataset.select(idx)
        return dataset

    def _post_preprocess(
        train_dataset: DATASET_TYPE,
        dataset_sample: Optional[int] = None,
        split_dataset_ratio: float = 0.,
        random_state: Optional[RandomState] = None,
        streaming: bool = False,
        *,
        streaming_val_size: int = 0,
        streaming_buffer_size: int = 16384,
    ) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
        """Split into train/val datasets and perform dataset sampling."""
        if streaming:
            val_dataset = None
            if split_dataset_ratio == 1:
                train_dataset, val_dataset = None, train_dataset
            else:
                streaming_val_size = kwargs.get('streaming_val_size', 0)
                streaming_buffer_size = kwargs.get('streaming_buffer_size', 16384)
                if streaming_val_size > 0:
                    train_dataset = train_dataset.shuffle(
                        seed=get_seed(random_state), buffer_size=streaming_buffer_size)
                    val_dataset = train_dataset.take(int(streaming_val_size))
                    train_dataset = train_dataset.skip(int(streaming_val_size))
        else:
            if dataset_sample is None:
                dataset_sample = len(train_dataset)
            assert 0 <= split_dataset_ratio <= 1
            if split_dataset_ratio == 1:
                train_dataset, val_dataset = None, train_dataset
                val_sample = dataset_sample
                assert val_sample <= len(
                    val_dataset), f'dataset_sample: {dataset_sample}, len(val_dataset): {len(val_dataset)}'
                val_dataset = sample_dataset(val_dataset, val_sample, random_state)
            else:
                if split_dataset_ratio == 0:
                    train_sample = dataset_sample
                    val_dataset = None
                else:
                    # Avoid having a high train_sample causing a high val_sample.
                    train_len = min(len(train_dataset), dataset_sample)
                    val_sample = max(int(train_len * split_dataset_ratio), 1)
                    train_sample = dataset_sample - val_sample
                    train_dataset, val_dataset = train_dataset.train_test_split(
                        test_size=val_sample, seed=get_seed(random_state),
                        load_from_cache_file=dataset_enable_cache).values()
                assert train_sample > 0
                train_dataset = sample_dataset(train_dataset, train_sample, random_state)
        return train_dataset, val_dataset

    @staticmethod
    def load(
        dataset: str,
        *,
        split_dataset_ratio: float = 0.,
        dataset_seed: Optional[RandomState] = None,
        load_from_cache_file: bool = False,
        num_proc: int = 1,
        download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
        # self-cognition
        model_name: Union[Tuple[str, str], List[str], None] = None,
        model_author: Union[Tuple[str, str], List[str], None] = None,
        # streaming
        streaming: bool = False,
        streaming_val_size: int = 0,
        streaming_buffer_size: int = 16384,
        # dataset info
        dataset_syntax: Optional[DatasetSyntax] = None,
        dataset_meta: Optional[DatasetMeta] = None,
    ) -> Tuple[HfDataset, Optional[HfDataset]]:
        dataset_name = dataset_syntax.dataset
        subsets = DatasetLoader._select_subsets(dataset_syntax, dataset_meta)

        use_hf = dataset_syntax.use_hf
        if dataset_meta.dataset_path:
            dataset = DatasetLoader._load_local_dataset(dataset_meta.dataset_path)
        else:
            dataset_str_f = 'Downloading the dataset from {hub}, dataset_id: {dataset_id}'
            if use_hf:
                dataset_id = dataset_meta.hf_dataset_id
                revision = dataset_meta.hf_revision
                dataset_str = dataset_str_f.format(hub='HuggingFace', dataset_id=dataset_id)
            else:
                dataset_id = dataset_meta.ms_dataset_id
                revision = dataset_meta.ms_revision
                dataset_str = dataset_str_f.format(hub='ModelScope', dataset_id=dataset_id)
            logger.info(dataset_str)
            assert dataset_id is not None, (f'dataset_name: {dataset_name}, use_hf: {use_hf}, '
                                            f'dataset_id: {dataset_id}.')
            datasets = []
            for subset in subsets:
                datasets.append(
                    DatasetLoader._load_repo_dataset(
                        dataset_id,
                        subset,
                        use_hf,
                        num_proc=num_proc,
                        revision=revision,
                        streaming=streaming,
                        download_mode=download_mode))
            dataset = self._concat_datasets(dataset, streaming)
        return DatasetLoader._post_preprocess(
            dataset,
            dataset_sample,
            split_dataset_ratio,
            random_state,
            streaming,
            streaming_val_size=streaming_val_size,
            streaming_buffer_size=streaming_buffer_size)

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
        load_from_cache_file: bool = False,
        num_proc: int = 1,
        download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
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
    if isinstance(dataset_seed, int):
        dataset_seed = RandomState(dataset_seed)
    datasets: List[str] = DatasetLoader._parse_datasets(datasets)  # to dataset_names and register
    train_datasets = []
    val_datasets = []
    load_kwargs = {
        'split_dataset_ratio': split_dataset_ratio,
        'random_state': dataset_seed,
        'load_from_cache_file': load_from_cache_file,
        'num_proc': num_proc,
        'download_mode': download_mode,
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
            'dataset_meta': DATASET_MAPPING[dataset_name]['dataset'],
        }
        train_dataset, val_dataset = load_function(dataset, **load_kwargs, **kwargs)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    train_datasets = DatasetLoader._concat_datasets(train_datasets)
    val_datasets = DatasetLoader._concat_datasets(val_datasets)
    return train_datasets, val_datasets
