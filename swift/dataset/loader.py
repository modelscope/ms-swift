# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from contextlib import nullcontext
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import load_dataset as hf_load_dataset
from modelscope.hub.utils.utils import get_cache_dir

from swift.hub import get_hub
from swift.utils import get_logger, get_seed, safe_ddp_context, use_hf_hub
from .dataset_meta import DATASET_TYPE, BaseDatasetLoader
from .dataset_syntax import DatasetSyntax
from .preprocessor import RowPreprocessor
from .register import DATASET_MAPPING, DatasetMeta, SubsetDataset

logger = get_logger()


class DatasetLoader(BaseDatasetLoader):

    def __init__(
        self,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        streaming: bool = False,
        hub_token: Optional[str] = None,
        strict: bool = False,
        download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
        columns: Optional[Dict[str, str]] = None,
        remove_unused_columns: bool = True,
    ):
        self.num_proc = num_proc
        self.load_from_cache_file = load_from_cache_file
        self.streaming = streaming
        self.hub_token = hub_token
        self.strict = strict
        self.download_mode = download_mode
        self.columns = columns
        self.remove_unused_columns = remove_unused_columns

    def _load_dataset_path(
        self,
        dataset_path: str,
        dataset_meta: DatasetMeta,
    ) -> HfDataset:
        ext = os.path.splitext(dataset_path)[1].lstrip('.')
        file_type = {'jsonl': 'json', 'txt': 'text'}.get(ext) or ext
        kwargs = {'split': 'train', 'streaming': self.streaming, 'num_proc': self.num_proc}
        if file_type == 'csv':
            kwargs['na_filter'] = False
        with safe_ddp_context(None, True):
            kwargs['cache_dir'] = os.path.join(get_cache_dir(), 'datasets')
            dataset = hf_load_dataset(file_type, data_files=dataset_path, **kwargs)
        if self.columns:
            dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)
        dataset = dataset_meta.preprocess_func(
            dataset, num_proc=self.num_proc, load_from_cache_file=self.load_from_cache_file, strict=self.strict)
        if self.remove_unused_columns:
            dataset = RowPreprocessor.remove_useless_columns(dataset)
        return dataset

    def _load_repo_dataset(
        self,
        dataset_id: str,
        subset: SubsetDataset,
        *,
        use_hf: Optional[bool] = None,
        revision: Optional[str] = None,
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
                            streaming=self.streaming,
                            revision=revision,
                            download_mode=self.download_mode,
                            hub_token=self.hub_token,
                            num_proc=self.num_proc)
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
                if self.streaming and isinstance(dataset, HfDataset):
                    dataset = dataset.to_iterable_dataset()
            if self.columns:
                dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)
            dataset = subset.preprocess_func(
                dataset, num_proc=self.num_proc, load_from_cache_file=self.load_from_cache_file, strict=self.strict)
            if self.remove_unused_columns:
                dataset = RowPreprocessor.remove_useless_columns(dataset)
            datasets.append(dataset)
        return self.concat_datasets(datasets)

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

    def load(
        self,
        dataset_syntax: Optional[DatasetSyntax] = None,
        dataset_meta: Optional[DatasetMeta] = None,
        *,
        use_hf: Optional[bool] = None,
    ) -> HfDataset:
        if dataset_syntax.dataset_type == 'path':
            dataset = self._load_dataset_path(
                dataset_syntax.dataset,
                dataset_meta=dataset_meta,
            )
        else:
            subsets: List[SubsetDataset] = self._select_subsets(dataset_syntax.subsets, dataset_meta)
            revision = dataset_meta.hf_revision if use_hf else dataset_meta.ms_revision
            datasets = []
            for subset in subsets:
                dataset = self._load_repo_dataset(
                    dataset_syntax.dataset,
                    subset,
                    use_hf=use_hf,
                    revision=revision,
                )
                datasets.append(dataset)
            dataset = self.concat_datasets(datasets)
        return dataset


def init_self_cognition_preprocessor(
    dataset_meta: Optional[DatasetMeta],
    model_name: Optional[Union[Tuple[str, str], List[str]]] = None,
    model_author: Optional[Union[Tuple[str, str], List[str]]] = None,
) -> None:
    from .dataset.llm import SelfCognitionPreprocessor
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
    """Load and preprocess datasets.

    This function provides a unified interface to load datasets from various sources (HuggingFace,
    ModelScope, or local paths), with support for splitting, shuffling, streaming, and interleaving
    multiple datasets. It also handles self-cognition dataset preprocessing for model training.

    Args:
        datasets: Single dataset name or list of dataset names to load. Can use special syntax
            for advanced configurations (e.g., 'dataset_name#1000' for sampling).
        split_dataset_ratio: Ratio for splitting dataset into train/validation sets.
            Value between 0 and 1. If 0, no validation split is created. Default: 0.
        seed: Random seed for reproducibility. Can be an integer or numpy RandomState object.
            If None, results will be non-deterministic. Default: 42.
        num_proc: Number of processes to use for dataset preprocessing. Set to None for
            streaming mode. Default: 1.
        load_from_cache_file: Whether to load preprocessed data from cache if available.
            Default: True.
        shuffle: Whether to shuffle the dataset(s) after loading. Default: False.
        streaming: Enable streaming mode for large datasets that don't fit in memory.
            When True, num_proc is automatically set to None. Default: False.
        interleave_prob: Probability weights for interleaving multiple datasets. Must have
            same length as datasets list. If None, datasets are concatenated instead. Default: None.
        stopping_strategy: Strategy when interleaving datasets of different lengths:
            - 'first_exhausted': Stop when shortest dataset is exhausted
            - 'all_exhausted': Continue until all datasets are exhausted
            Default: 'first_exhausted'.
        shuffle_buffer_size: Buffer size for shuffling in streaming mode. Larger values
            provide better randomization but use more memory. Default: 1000.
        use_hf: Force using HuggingFace Hub (True) or ModelScope (False). If None,
            it is controlled by the environment variable `USE_HF`, which defaults to '0'.
            Default: None.
        hub_token: Authentication token for accessing private datasets on the hub. Default: None.
        strict: If True, raise exceptions when encountering malformed data rows.
            If False, skip invalid rows with warnings. Default: False.
        download_mode: How to handle existing cached datasets:
            - 'reuse_dataset_if_exists': Use cached version if available
            - 'force_redownload': Always download fresh copy
            Default: 'reuse_dataset_if_exists'.
        columns: Manual column name mapping for datasets. Dictionary mapping source column
            names to target column names (e.g., {'text': 'content'}). Default: None.
        remove_unused_columns: Whether to remove columns not used in preprocessing.
            Helps reduce memory usage. Default: True.
        model_name: Model name for self-cognition task preprocessing. Can be a tuple of
            (Chinese_name, English_name) or list of names. Default: None.
        model_author: Model author for self-cognition task preprocessing. Can be a tuple of
            (Chinese_author, English_author) or list of authors. Default: None.

    Returns:
        A tuple of (train_dataset, val_dataset):
            - train_dataset: The training dataset
            - val_dataset: The validation dataset if split_dataset_ratio > 0, otherwise None

    Examples:
        >>> # Load single dataset
        >>> train_ds, val_ds = load_dataset('AI-ModelScope/alpaca-gpt4-data-zh', split_dataset_ratio=0.1)

        >>> # Load multiple datasets
        >>> train_ds, _ = load_dataset(
        ...     ['AI-ModelScope/alpaca-gpt4-data-zh#500', 'swift/self-cognition#500'],
        ...     model_name=('我的模型', 'MyModel'),
        ...     model_author=('作者', 'Author')
        ... )
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
    loader = DatasetLoader(
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        streaming=streaming,
        hub_token=hub_token,
        strict=strict,
        download_mode=download_mode,
        columns=columns,  # columns_mapping
        remove_unused_columns=remove_unused_columns,
    )

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
        train_dataset = loader.load(dataset_syntax, dataset_meta, use_hf=use_hf)
        train_dataset, val_dataset = loader.post_process(
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
        train_datasets = loader.concat_datasets(train_datasets)
        val_datasets = loader.concat_datasets(val_datasets)
    else:
        train_datasets = loader.interleave_datasets(
            train_datasets, interleave_prob, seed=get_seed(seed), stopping_strategy=stopping_strategy)
        val_datasets = loader.interleave_datasets(
            val_datasets, interleave_prob, seed=get_seed(seed), stopping_strategy=stopping_strategy)

    if shuffle:
        if train_datasets:
            train_datasets = loader.shuffle_dataset(
                train_datasets, seed=get_seed(seed), buffer_size=shuffle_buffer_size)
        if val_datasets:
            val_datasets = loader.shuffle_dataset(val_datasets, seed=get_seed(seed), buffer_size=shuffle_buffer_size)
    return train_datasets, val_datasets
