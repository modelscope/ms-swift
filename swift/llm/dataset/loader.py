# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from datasets import concatenate_datasets, interleave_datasets
from modelscope.hub.api import ModelScopeConfig
from modelscope.utils.config_ds import MS_CACHE_HOME

from swift.hub import HFHub, MSHub
from swift.utils import download_ms_file, get_logger, get_seed, safe_ddp_context, use_hf_hub
from .register import DATASET_MAPPING, DATASET_TYPE, DatasetMeta, SubsetDataset, register_dataset_info
from .utils import sample_dataset

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
        if os.path.isfile(self.dataset) or self.dataset.startswith('/'):
            self.dataset_type = 'path'
            assert os.path.isfile(self.dataset)
        elif self.use_hf:
            self.dataset_type = 'hf'
        else:
            self.dataset_type = 'ms'

    def get_raw(self):
        use_hf_mapping = {None: '', 1: 'HF::', 0: 'MS::'}
        subsets = '/'.join(self.subsets)
        dataset_sample = '' if self.dataset_sample is None else f'#{self.dataset_sample}'
        return f'{use_hf_mapping[self.use_hf]}{self.dataset}{subsets}{dataset_sample}'

    @classmethod
    def parse(cls, dataset: str) -> 'DatasetSyntax':
        """Parse the dataset from the command line"""
        # HF::dataset_id or dataset_path:subset1/subset2/subset3#dataset_sample
        use_hf, other = DatasetLoader._safe_split(dataset, '::', False)
        if os.path.isfile(other):
            part1, dataset_sample = other, None
        else:
            part1, dataset_sample = DatasetLoader._safe_split(other, '#', True, 'right')
        if os.path.isfile(part1):
            dataset, subsets = part1, None
        else:
            dataset, subsets = DatasetLoader._safe_split(part1, ':', True)

        if use_hf is not None:
            use_hf = {'ms': False, 'hf': True}[use_hf.strip().lower()]
        if subsets is not None:
            subsets = [subset.strip() for subset in subsets.split('/')]
        if dataset_sample is not None:
            dataset_sample = int(dataset_sample)
        return cls(dataset.strip(), use_hf, subsets or [], dataset_sample)

    def to_dict(self):
        # Convert to a format that can be parsed by register_dataset_info.
        res = {}
        mapping = {'path': 'dataset_path', 'hf': 'hf_dataset_id', 'ms': 'ms_dataset_id'}
        key = mapping[self.dataset_type]
        res[key] = self.dataset
        return res


_dataset_meta_mapping = None


def get_dataset_meta_mapping() -> Dict[str, str]:
    global _dataset_meta_mapping
    if _dataset_meta_mapping is not None:
        return _dataset_meta_mapping
    _dataset_meta_mapping = {}
    for dataset_meta in DATASET_MAPPING.values():
        if dataset_meta.dataset_path is not None:
            _dataset_meta_mapping[('path', dataset_meta.dataset_path.lower())] = dataset_meta
        if dataset_meta.ms_dataset_id is not None:
            _dataset_meta_mapping[('ms', dataset_meta.ms_dataset_id.lower())] = dataset_meta
        if dataset_meta.hf_dataset_id is not None:
            _dataset_meta_mapping[('hf', dataset_meta.hf_dataset_id.lower())] = dataset_meta
    return _dataset_meta_mapping


class DatasetLoader:

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
    def _concat_datasets(datasets: List[HfDataset], streaming: bool) -> Optional[HfDataset]:
        if len(datasets) == 0:
            return
        if len(datasets) == 1:
            return datasets[0]
        return interleave_datasets(datasets) if streaming else concatenate_datasets(datasets)

    @staticmethod
    def _load_local_dataset(dataset_meta: DatasetMeta,
                            *,
                            num_proc: int = 1,
                            strict: bool = True,
                            load_from_cache_file: bool = False,
                            streaming: bool = False) -> HfDataset:
        dataset_path = dataset_meta.dataset_path

        if dataset_path.endswith('.csv'):
            dataset = HfDataset.from_csv(dataset_path, na_filter=False)
        elif dataset_path.endswith('.jsonl') or dataset_path.endswith('.json'):
            dataset = HfDataset.from_json(dataset_path)
        elif dataset_path.endswith('.txt'):
            dataset = HfDataset.from_text(dataset_path)
        elif dataset_path.endswith('.parquet'):
            dataset = HfDataset.from_parquet(dataset_path)
        else:
            raise ValueError('The custom dataset only supports csv, jsonl, json, txt, parquet format.')
        if streaming:
            dataset = dataset.to_iterable_dataset()

        dataset = dataset_meta.preprocess_func(
            dataset, num_proc=num_proc, strict=strict, load_from_cache_file=load_from_cache_file)
        return dataset

    @staticmethod
    def _load_repo_dataset(
        dataset_id: str,
        subset: SubsetDataset,
        use_hf: bool,
        *,
        num_proc: int = 1,
        strict: bool = True,
        load_from_cache_file: bool = False,
        streaming: bool = False,
        revision: Optional[str] = None,
        download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
    ) -> HfDataset:
        datasets = []
        for split in subset.split:
            retry = 3
            if use_hf:
                hub = HFHub
            else:
                hub = MSHub
            with safe_ddp_context():
                while True:
                    try:
                        dataset = hub.load_dataset(
                            dataset_id,
                            subset.subset,
                            split,
                            streaming=streaming,
                            revision=revision,
                            download_mode=download_mode)
                    except Exception as e:
                        if retry == 0:
                            raise
                        retry -= 1
                        logger.error(f'Dataset {dataset_id} load failed: subset_name={subset.subset},'
                                     f'split={split} with error: {e}')
                    else:
                        break
                if streaming and hasattr(dataset, '_hf_ds'):
                    dataset = dataset._hf_ds
                    if not isinstance(dataset, HfIterableDataset):
                        dataset = dataset.to_iterable_dataset()
                if hasattr(dataset, 'to_hf_dataset'):
                    dataset = dataset.to_hf_dataset()
            dataset = subset.preprocess_func(
                dataset, num_proc=num_proc, strict=strict, load_from_cache_file=load_from_cache_file)
            datasets.append(dataset)
        return DatasetLoader._concat_datasets(datasets, streaming)

    @staticmethod
    def _select_subsets(subsets: List[str], dataset_meta: DatasetMeta) -> List[SubsetDataset]:
        subset_mapping = {subset.name: subset for subset in dataset_meta.subsets}
        subset_names = list(subset_mapping.keys())
        if not subsets:
            if len(subset_names) <= 1:
                subsets = subset_names
            elif 'default' in subset_names:
                subsets = ['default']
            else:
                raise ValueError(f'Please provide subsets. available subsets: {subset_names}')
        elif len(subsets) == 1 and subsets[0] == 'all' and 'all' not in subset_names:
            subsets = subset_names
        subsets = [subset_mapping[subset_name].set_default(dataset_meta) for subset_name in subsets]
        return subsets

    @staticmethod
    def post_process(
        train_dataset: DATASET_TYPE,
        dataset_sample: Optional[int] = None,
        split_dataset_ratio: float = 0.,
        random_state: Optional[np.random.RandomState] = None,
        streaming: bool = False,
        *,
        load_from_cache_file: bool = False,
        streaming_val_size: int = 0,
        streaming_buffer_size: int = 16384,
    ) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
        """Split into train/val datasets and perform dataset sampling."""
        if streaming:
            val_dataset = None
            if split_dataset_ratio == 1:
                train_dataset, val_dataset = None, train_dataset
            else:
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
                        load_from_cache_file=load_from_cache_file).values()
                assert train_sample > 0
                train_dataset = sample_dataset(train_dataset, train_sample, random_state)
        return train_dataset, val_dataset

    @staticmethod
    def load(
        dataset_syntax: Optional[DatasetSyntax] = None,
        dataset_meta: Optional[DatasetMeta] = None,
        *,
        num_proc: int = 1,
        strict: bool = True,
        load_from_cache_file: bool = False,
        download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
        # self-cognition
        model_name: Union[Tuple[str, str], List[str], None] = None,
        model_author: Union[Tuple[str, str], List[str], None] = None,
        # streaming
        streaming: bool = False,
    ) -> HfDataset:

        if dataset_meta.dataset_path:
            dataset = DatasetLoader._load_local_dataset(
                dataset_meta=dataset_meta,
                num_proc=num_proc,
                strict=strict,
                load_from_cache_file=load_from_cache_file,
                streaming=streaming,
            )
        else:
            subsets: List[SubsetDataset] = DatasetLoader._select_subsets(dataset_syntax.subsets, dataset_meta)
            dataset_str_f = 'Downloading the dataset from {hub}, dataset_id: {dataset_id}'
            if dataset_syntax.use_hf:
                dataset_id = dataset_meta.hf_dataset_id
                revision = dataset_meta.hf_revision
                dataset_str = dataset_str_f.format(hub='HuggingFace', dataset_id=dataset_id)
            else:
                dataset_id = dataset_meta.ms_dataset_id
                revision = dataset_meta.ms_revision
                dataset_str = dataset_str_f.format(hub='ModelScope', dataset_id=dataset_id)
            logger.info(dataset_str)
            assert dataset_id is not None, (f'dataset: {dataset_syntax.dataset}, use_hf: {dataset_syntax.use_hf}, '
                                            f'dataset_id: {dataset_id}.')
            datasets = []
            for subset in subsets:
                datasets.append(
                    DatasetLoader._load_repo_dataset(
                        dataset_id,
                        subset,
                        dataset_syntax.use_hf,
                        num_proc=num_proc,
                        strict=strict,
                        load_from_cache_file=load_from_cache_file,
                        revision=revision,
                        streaming=streaming,
                        download_mode=download_mode))
            dataset = DatasetLoader._concat_datasets(datasets, streaming)
        return dataset

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
    def parse_dataset(datasets: List[str]) -> List[Tuple[DatasetSyntax, DatasetMeta]]:
        # ms_dataset_id/hf_dataset_id/dataset_path -> dataset_syntax, dataset_meta
        dataset_meta_mapping = get_dataset_meta_mapping()

        # register_dataset
        res_datasets: List[str] = []
        register_idx = 0
        dataset_info = []
        for dataset in datasets:
            dataset_syntax = DatasetSyntax.parse(dataset)
            dataset_meta = dataset_meta_mapping.get((dataset_syntax.dataset_type, dataset_syntax.dataset.lower()))
            if dataset_meta is None:
                # This dataset needs to be registered.
                register_idx += 1
                dataset_info.append(dataset_syntax.to_dict())
            res_datasets.append((dataset_syntax, dataset_meta))
        register_dataset_info(dataset_info)

        return res_datasets


def load_dataset(
        datasets: List[str],
        split_dataset_ratio: float = 0.,
        dataset_seed: Union[int, np.random.RandomState] = 42,
        *,
        num_proc: int = 1,
        strict: bool = True,
        load_from_cache_file: bool = False,
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
        dataset_seed = np.random.RandomState(dataset_seed)
    train_datasets = []
    val_datasets = []
    load_kwargs = {
        'num_proc': num_proc,
        'strict': strict,
        'load_from_cache_file': load_from_cache_file,
        'download_mode': download_mode,
        'model_name': model_name,
        'model_author': model_author,
        'streaming': streaming,
    }
    for dataset_syntax, dataset_meta in DatasetLoader.parse_dataset(datasets):
        load_function = dataset_meta.load_function
        train_dataset = load_function(dataset_syntax, dataset_meta, **load_kwargs)
        train_dataset, val_dataset = DatasetLoader.post_process(
            train_dataset,
            dataset_syntax.dataset_sample,
            split_dataset_ratio,
            dataset_seed,
            streaming,
            load_from_cache_file=load_from_cache_file,
            streaming_val_size=streaming_val_size,
            streaming_buffer_size=streaming_buffer_size)
        if train_dataset is not None:
            train_datasets.append(train_dataset)
        if val_dataset is not None:
            val_datasets.append(val_dataset)

    train_datasets = DatasetLoader._concat_datasets(train_datasets, streaming)
    val_datasets = DatasetLoader._concat_datasets(val_datasets, streaming)
    return train_datasets, val_datasets
