# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import platform
import re
import shutil
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from tempfile import TemporaryDirectory
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets, interleave_datasets
from datasets import load_dataset as hf_load_dataset
from modelscope.hub.api import ModelScopeConfig
from modelscope.hub.utils.utils import get_cache_dir
from modelscope.utils.config_ds import MS_CACHE_HOME

from swift.hub import get_hub
from swift.utils import download_ms_file, get_logger, get_seed, safe_ddp_context, use_hf_hub
from .preprocessor import RowPreprocessor
from .register import DATASET_MAPPING, DATASET_TYPE, DatasetMeta, SubsetDataset
from .utils import sample_dataset

logger = get_logger()

_dataset_meta_mapping = None


@dataclass
class DatasetSyntax:
    dataset: str
    subsets: List[str] = field(default_factory=list)
    dataset_sample: Optional[int] = None
    use_hf: Optional[bool] = None

    def __post_init__(self):
        if os.path.isfile(self.dataset):
            self.dataset_type = 'path'
        else:  # dataset_id or dataset_dir
            self.dataset_type = 'repo'

    def get_raw(self):
        subsets = '/'.join(self.subsets)
        dataset_sample = '' if self.dataset_sample is None else f'#{self.dataset_sample}'
        return f'{self.dataset}{subsets}{dataset_sample}'

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

    @classmethod
    def parse(cls, dataset: str) -> 'DatasetSyntax':
        """Parse the dataset from the command line"""
        # hf/ms::dataset_id or dataset_path:subset1/subset2/subset3#dataset_sample
        if os.path.exists(dataset):
            use_hf = None
        else:
            use_hf, dataset = cls._safe_split(dataset, '::', False)
            if isinstance(use_hf, str):
                use_hf = use_hf.lower()
            use_hf = {'hf': True, 'ms': False}.get(use_hf)
        if os.path.exists(dataset):
            other, dataset_sample = dataset, None
        else:
            other, dataset_sample = cls._safe_split(dataset, '#', True, 'right')
        if os.path.exists(other):
            dataset, subsets = other, None
        else:
            dataset, subsets = cls._safe_split(other, ':', True)

        if subsets is not None:
            subsets = [subset.strip() for subset in subsets.split('/')]
        if dataset_sample is not None:
            dataset_sample = int(dataset_sample)
        return cls(dataset.strip(), subsets or [], dataset_sample, use_hf)

    def get_dataset_meta(self, use_hf: bool):
        dataset_meta_mapping = self._get_dataset_meta_mapping()
        dataset_type = self.dataset_type
        if dataset_type == 'path':
            dataset_meta = dataset_meta_mapping.get((dataset_type, self.dataset))
        else:
            dataset_type = 'repo' if os.path.isdir(self.dataset) else {True: 'hf', False: 'ms'}[use_hf]
            dataset_meta = dataset_meta_mapping.get((dataset_type, self.dataset))
        return dataset_meta or self._get_matched_dataset_meta(dataset_meta_mapping) or DatasetMeta()

    @staticmethod
    def _get_dataset_meta_mapping() -> Dict[Tuple[str, str], DatasetMeta]:
        global _dataset_meta_mapping
        if _dataset_meta_mapping is not None:
            return _dataset_meta_mapping
        _dataset_meta_mapping = {}
        for dataset_meta in DATASET_MAPPING.values():
            if dataset_meta.dataset_path is not None:
                dataset_type = 'repo' if os.path.isdir(dataset_meta.dataset_path) else 'path'
                _dataset_meta_mapping[(dataset_type, dataset_meta.dataset_path)] = dataset_meta
            if dataset_meta.ms_dataset_id is not None:
                _dataset_meta_mapping[('ms', dataset_meta.ms_dataset_id)] = dataset_meta
            if dataset_meta.hf_dataset_id is not None:
                _dataset_meta_mapping[('hf', dataset_meta.hf_dataset_id)] = dataset_meta
        return _dataset_meta_mapping

    @staticmethod
    def get_dataset_name(dataset_id: str) -> str:
        # compat hf hub
        dataset_id = dataset_id.rstrip('/')
        match_ = re.search('/datasets--.+?--(.+?)/snapshots/', dataset_id)
        if match_ is not None:
            return match_.group(1)

        dataset_name = dataset_id.rsplit('/', 1)[-1]
        if platform.system().lower() == 'windows':
            dataset_name = dataset_name.rsplit('\\', 1)[-1]
        return dataset_name

    def _get_matched_dataset_meta(self, dataset_meta_mapping):
        suffix_dataset_meta_mapping = {}
        for dataset_name, dataset_meta in dataset_meta_mapping.items():
            dataset_name = self.get_dataset_name(dataset_name[1])
            suffix_dataset_meta_mapping[dataset_name] = dataset_meta
        dataset_name = self.get_dataset_name(self.dataset)
        dataset_meta = suffix_dataset_meta_mapping.get(dataset_name)
        return dataset_meta


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
    def _concat_datasets(datasets: List[HfDataset]) -> Optional[HfDataset]:
        if len(datasets) == 0:
            return
        if len(datasets) == 1:
            return datasets[0]
        return concatenate_datasets(datasets)

    @staticmethod
    def _interleave_datasets(datasets, *args, **kwargs):
        if len(datasets) == 0:
            return
        if len(datasets) == 1:
            return datasets[0]
        return interleave_datasets(datasets, *args, **kwargs)

    @staticmethod
    def _load_dataset_path(
        dataset_path: str,
        dataset_meta: DatasetMeta,
        *,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        strict: bool = False,
        streaming: bool = False,
        columns: Optional[Dict[str, str]] = None,
        remove_unused_columns: bool = True,
    ) -> HfDataset:
        ext = os.path.splitext(dataset_path)[1].lstrip('.')
        file_type = {'jsonl': 'json', 'txt': 'text'}.get(ext) or ext
        kwargs = {'split': 'train', 'streaming': streaming, 'num_proc': num_proc}
        if file_type == 'csv':
            kwargs['na_filter'] = False
        with safe_ddp_context(None, True):
            kwargs['cache_dir'] = os.path.join(get_cache_dir(), 'datasets')
            dataset = hf_load_dataset(file_type, data_files=dataset_path, **kwargs)
        if columns:
            dataset = RowPreprocessor.safe_rename_columns(dataset, columns)
        dataset = dataset_meta.preprocess_func(
            dataset, num_proc=num_proc, load_from_cache_file=load_from_cache_file, strict=strict)
        if remove_unused_columns:
            dataset = RowPreprocessor.remove_useless_columns(dataset)
        return dataset

    @staticmethod
    def _load_repo_dataset(
        dataset_id: str,
        subset: SubsetDataset,
        *,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        streaming: bool = False,
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        strict: bool = False,
        revision: Optional[str] = None,
        download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
        columns: Optional[Dict[str, str]] = None,
        remove_unused_columns: bool = True,
    ) -> HfDataset:
        datasets = []
        if os.path.isdir(dataset_id):
            retry = 1
            load_context = nullcontext
            use_hf = True
            dataset_str = f'Use local folder, dataset_dir: {dataset_id}'
            # The dataset downloaded from modelscope will have an additional dataset_infos.json file.
            with safe_ddp_context('dataset_infos_rename'):
                dataset_infos_path = os.path.join(dataset_id, 'dataset_infos.json')
                if os.path.isfile(dataset_infos_path):
                    os.rename(dataset_infos_path, f'{dataset_infos_path}_bak')
        elif dataset_id.startswith('/'):
            raise ValueError(f'The local path does not exist, dataset_id: `{dataset_id}`. '
                             f'os.path.exists(dataset_id): {os.path.exists(dataset_id)}')
        else:
            retry = 3
            load_context = partial(safe_ddp_context, hash_id=dataset_id, use_barrier=True)
            dataset_str_f = 'Downloading the dataset from {hub}, dataset_id: {dataset_id}'
            if use_hf:
                dataset_str = dataset_str_f.format(hub='HuggingFace', dataset_id=dataset_id)
            else:
                dataset_str = dataset_str_f.format(hub='ModelScope', dataset_id=dataset_id)
        logger.info(dataset_str)
        hub = get_hub(use_hf)
        for split in subset.split:
            i = 1
            with load_context():
                while True:
                    try:
                        dataset = hub.load_dataset(
                            dataset_id,
                            subset.subset,
                            split,
                            streaming=streaming,
                            revision=revision,
                            download_mode=download_mode,
                            hub_token=hub_token,
                            num_proc=num_proc)
                    except Exception as e:
                        if i == retry:
                            raise
                        i += 1
                        logger.error(f'Dataset {dataset_id} load failed: subset_name={subset.subset},'
                                     f'split={split} with error: {e}')
                    else:
                        break
            if hasattr(dataset, '_hf_ds'):
                dataset = dataset._hf_ds
                if streaming and isinstance(dataset, HfDataset):
                    dataset = dataset.to_iterable_dataset()
            if columns:
                dataset = RowPreprocessor.safe_rename_columns(dataset, columns)
            dataset = subset.preprocess_func(
                dataset, num_proc=num_proc, load_from_cache_file=load_from_cache_file, strict=strict)
            if remove_unused_columns:
                dataset = RowPreprocessor.remove_useless_columns(dataset)
            datasets.append(dataset)
        return DatasetLoader._concat_datasets(datasets)

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
            subsets = [subset_name for subset_name in subset_names if not subset_mapping[subset_name].is_weak_subset]

        subsets = [
            subset_mapping[subset_name] if subset_name in subset_mapping else SubsetDataset(subset=subset_name)
            for subset_name in subsets
        ]
        return [subset.set_default(dataset_meta) for subset in subsets]

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

    @staticmethod
    def load(
        dataset_syntax: Optional[DatasetSyntax] = None,
        dataset_meta: Optional[DatasetMeta] = None,
        *,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        streaming: bool = False,
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        strict: bool = False,
        download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
        columns: Optional[Dict[str, str]] = None,
        remove_unused_columns: bool = True,
    ) -> HfDataset:
        if dataset_syntax.dataset_type == 'path':
            dataset = DatasetLoader._load_dataset_path(
                dataset_syntax.dataset,
                dataset_meta=dataset_meta,
                num_proc=num_proc,
                load_from_cache_file=load_from_cache_file,
                strict=strict,
                streaming=streaming,
                columns=columns,
                remove_unused_columns=remove_unused_columns,
            )
        else:
            subsets: List[SubsetDataset] = DatasetLoader._select_subsets(dataset_syntax.subsets, dataset_meta)
            revision = dataset_meta.hf_revision if use_hf else dataset_meta.ms_revision
            datasets = []
            for subset in subsets:
                dataset = DatasetLoader._load_repo_dataset(
                    dataset_syntax.dataset,
                    subset,
                    use_hf=use_hf,
                    hub_token=hub_token,
                    num_proc=num_proc,
                    load_from_cache_file=load_from_cache_file,
                    strict=strict,
                    revision=revision,
                    streaming=streaming,
                    download_mode=download_mode,
                    columns=columns,
                    remove_unused_columns=remove_unused_columns,
                )
                datasets.append(dataset)
            dataset = DatasetLoader._concat_datasets(datasets)
        return dataset


def init_self_cognition_preprocessor(
    dataset_meta: Optional[DatasetMeta],
    model_name: Optional[Union[Tuple[str, str], List[str]]] = None,
    model_author: Optional[Union[Tuple[str, str], List[str]]] = None,
) -> None:
    if dataset_meta is None or model_name is None and model_author is None:
        return
    kwargs = {}
    # zh, en
    for key in ['name', 'author']:
        val = locals()[f'model_{key}']
        if isinstance(val, str):
            val = [val]
        if val is not None and val[0] is not None and (len(val) == 1 or val[1] is None):
            val = (val[0], val[0])
        kwargs[key] = val

    from .dataset.llm import SelfCognitionPreprocessor
    preprocess_funcs = [dataset_meta.preprocess_func]
    preprocess_funcs += [subset.preprocess_func for subset in dataset_meta.subsets if isinstance(subset, SubsetDataset)]
    for preprocess_func in preprocess_funcs:
        if isinstance(preprocess_func, SelfCognitionPreprocessor):
            preprocess_func.set_name_author(**kwargs)
    logger.info_once(f"SelfCognitionPreprocessor has been successfully configured with name: {kwargs['name']}, "
                     f"author: {kwargs['author']}.")


def load_dataset(
    datasets: Union[List[str], str],
    *,
    split_dataset_ratio: float = 0.,
    seed: Union[int, np.random.RandomState, None] = 42,
    num_proc: int = 1,
    load_from_cache_file: bool = True,
    shuffle: bool = False,
    streaming: bool = False,
    interleave_prob: Optional[List[float]] = None,
    stopping_strategy: Literal['first_exhausted', 'all_exhausted'] = 'first_exhausted',
    shuffle_buffer_size: int = 1000,
    use_hf: Optional[bool] = None,
    hub_token: Optional[str] = None,
    strict: bool = False,
    download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
    columns: Optional[Dict[str, str]] = None,  # columns_mapping
    remove_unused_columns: bool = True,
    # self-cognition
    model_name: Optional[Union[Tuple[str, str], List[str]]] = None,  # zh, en
    model_author: Optional[Union[Tuple[str, str], List[str]]] = None,
) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
    """The interface to load any registered dataset

    Args:
        datasets: The dataset name list

        split_dataset_ratio: The dataset split ratio
        seed: The dataset random seed
        num_proc: Proc number to use when preprocess the dataset.
        shuffle: Whether to shuffle the dataset.
        streaming: Streaming mode or not
        use_hf: Use hf dataset or ms dataset.
        hub_token: The token of the hub.
        strict: Raise if any row is not correct.
        download_mode: Download mode, default is `reuse_dataset_if_exists`.
        columns: Used for manual column mapping of datasets.

        model_name: Model name in self-cognition task.
        model_author: Model author in self-cognition task
    Returns:
        The train dataset and val dataset
    """
    init_self_cognition_preprocessor(DATASET_MAPPING.get('self-cognition'), model_name, model_author)
    if isinstance(datasets, str):
        datasets = [datasets]
    if not isinstance(seed, np.random.RandomState):
        seed = np.random.RandomState(seed)
    if streaming:
        num_proc = None
    train_datasets = []
    val_datasets = []
    load_kwargs = {
        'num_proc': num_proc,
        'load_from_cache_file': load_from_cache_file,
        'strict': strict,
        'download_mode': download_mode,
        'columns': columns,
        'streaming': streaming,
        'hub_token': hub_token,
        'remove_unused_columns': remove_unused_columns,
    }
    use_hf_default = use_hf
    if use_hf_default is None:
        use_hf_default = True if use_hf_hub() else False
    for dataset in datasets:
        dataset_syntax = DatasetSyntax.parse(dataset)
        use_hf = dataset_syntax.use_hf or use_hf_default
        # compat dataset_name
        if dataset_syntax.dataset in DATASET_MAPPING:
            dataset_meta = DATASET_MAPPING[dataset_syntax.dataset]
            if dataset_syntax.use_hf is None and dataset_meta.dataset_path is not None:
                dataset_syntax.dataset = dataset_meta.dataset_path
                dataset_syntax.dataset_type = 'path'
            else:
                dataset_syntax.dataset = dataset_meta.hf_dataset_id if use_hf else dataset_meta.ms_dataset_id
        else:
            dataset_meta = dataset_syntax.get_dataset_meta(use_hf)
        load_function = dataset_meta.load_function
        train_dataset = load_function(dataset_syntax, dataset_meta, **load_kwargs, use_hf=use_hf)
        train_dataset, val_dataset = DatasetLoader.post_process(
            train_dataset,
            dataset_sample=dataset_syntax.dataset_sample,
            split_dataset_ratio=split_dataset_ratio,
            streaming=streaming,
            shuffle=shuffle,
            random_state=seed,
        )
        if train_dataset is not None:
            train_datasets.append(train_dataset)
        if val_dataset is not None:
            val_datasets.append(val_dataset)

    if interleave_prob is None:
        train_datasets = DatasetLoader._concat_datasets(train_datasets)
        val_datasets = DatasetLoader._concat_datasets(val_datasets)
    else:
        train_datasets = DatasetLoader._interleave_datasets(
            train_datasets, interleave_prob, seed=get_seed(seed), stopping_strategy=stopping_strategy)
        val_datasets = DatasetLoader._interleave_datasets(
            val_datasets, interleave_prob, seed=get_seed(seed), stopping_strategy=stopping_strategy)

    if shuffle:
        if train_datasets:
            train_datasets = DatasetLoader.shuffle_dataset(
                train_datasets, seed=get_seed(seed), buffer_size=shuffle_buffer_size)
        if val_datasets:
            val_datasets = DatasetLoader.shuffle_dataset(
                val_datasets, seed=get_seed(seed), buffer_size=shuffle_buffer_size)
    return train_datasets, val_datasets
