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
from .register import Dataset, register_dataset_info

DATASET_TYPE = Union[HfDataset, HfIterableDataset]

SubsetSplit = Union[str, Tuple[str, str], List[str]]
DATASET_MAPPING: Dict[str, Dict[str, Any]] = {}
PreprocessFunc = Callable[[DATASET_TYPE], DATASET_TYPE]
logger = get_logger()


class DatasetLoader:

    @classmethod
    def dataset_get_function(cls,
                             dataset_id: str,
                             subsets: Optional[List[str]],
                             preprocess_func: Union[PreprocessFunc, RowPreprocessor],
                             split: List[str],
                             dataset_sample: int = -1,
                             *,
                             random_state: Optional[RandomState] = None,
                             split_dataset_ratio: float = 0.,
                             remove_useless_columns: bool = True,
                             use_hf: bool = False,
                             **kwargs) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
        """Get dataset from repo and post process them.

        Args:
            dataset_id: The dataset id.
            subsets: The subsets info
            preprocess_func: The preprocess function
            split: The dataset split
            dataset_sample: The sample number, default `-1`, means all data
            random_state: The random state, default `None`
            split_dataset_ratio: The dataset split ratio, default `0`
            remove_useless_columns: Remove useless columns or not, default `True`
            use_hf: Using hf hub or ms hub, default `False` means ms hub.
        Returns:
            The loaded dataset
        """
        raise NotImplementedError

    @classmethod
    def remove_useless_columns(cls, dataset: DATASET_TYPE) -> DATASET_TYPE:
        """Remove useless columns from the dataset unless the columns are standard ones.

        Args:
            dataset: The dataset instance

        Returns:
            The processed dataset instance
        """
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

    @classmethod
    def sample_dataset(cls,
                       dataset: HfDataset,
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
        if dataset_sample in {None, -1, len(dataset)}:
            return dataset
        if random_state is None:
            random_state = RandomState()

        idx_repeat = np.tile(range(len(dataset)), dataset_sample // len(dataset))
        idx_random = random_state.permutation(len(dataset))[:dataset_sample % len(dataset)]
        idx = np.concatenate([idx_repeat, idx_random])
        dataset = dataset.select(idx)
        return dataset

    @classmethod
    def post_preprocess(
        cls,
        train_dataset: DATASET_TYPE,
        dataset_sample: int,
        random_state: Optional[RandomState] = None,
        preprocess_func: Optional[PreprocessFunc] = None,
        split_dataset_ratio: float = 0.,
        remove_useless_columns: bool = True,
        **kwargs,
    ) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
        """Post process the dataset, this function will do the following things:
        1. Sample from dataset
        2. Split the dataset to train and validation
        3. Post process

        Args:
            train_dataset: The training dataset, required
            dataset_sample: The sample number, required
            random_state: The random state, default `None`
            preprocess_func: The preprocessor, default is `None`
            split_dataset_ratio: The dataset split ratio, Default `0`
            remove_useless_columns: Remove useless columns or not, default `True`
            **kwargs:
                dataset_enable_cache: Enable cache or not, default `False`
        Returns:
            The processed dataset
        """
        assert train_dataset is not None
        streaming = kwargs.get('streaming', False)
        if not streaming:
            if dataset_sample == -1:
                dataset_sample = len(train_dataset)
            assert 0 <= split_dataset_ratio <= 1
            if split_dataset_ratio == 1:
                # The validation scenario, switch all data to validation set
                train_dataset, val_dataset = None, train_dataset
                val_sample = dataset_sample
                assert val_sample <= len(
                    val_dataset), f'dataset_sample: {dataset_sample}, len(val_dataset): {len(val_dataset)}'
                val_dataset = cls.sample_dataset(val_dataset, val_sample, random_state)
            else:
                # The training scenario
                if split_dataset_ratio == 0:
                    train_sample = dataset_sample
                    val_dataset = None
                else:
                    # Avoid having a high train_sample causing a high val_sample.
                    _train_len = min(len(train_dataset), dataset_sample)
                    val_sample = max(int(_train_len * split_dataset_ratio), 1)
                    train_sample = dataset_sample - val_sample
                    assert isinstance(val_sample, int)
                    train_dataset, val_dataset = train_dataset.train_test_split(
                        test_size=val_sample,
                        seed=get_seed(random_state),
                        load_from_cache_file=kwargs.get('dataset_enable_cache', False)).values()

                assert train_sample > 0
                train_dataset = cls.sample_dataset(train_dataset, train_sample, random_state)
        else:
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

        res = []
        for dataset in [train_dataset, val_dataset]:
            if dataset is not None and preprocess_func is not None:
                dataset = preprocess_func(dataset, num_proc=kwargs.get('num_proc'))
            if dataset is not None and (streaming or len(dataset) > 0) and remove_useless_columns:
                dataset = cls.remove_useless_columns(dataset)
            res.append(dataset)
        return tuple(res)

    @classmethod
    def download_dataset(cls, dataset_id: str, files: List[str], force_download: bool = False) -> str:
        """Download dataset from repo manually

        Args:
            dataset_id: The dataset id of ModelScope
            files: Which files to download
            force_download: Force download or not

        Returns:
            The dataset dir
        """
        assert isinstance(files, list)
        url = f'http://www.modelscope.cn/api/v1/datasets/{dataset_id}/repo?Revision=master&FilePath={{fpath}}'
        cache_dir = os.path.join(MS_CACHE_HOME, 'datasets', dataset_id, 'master')
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

    @classmethod
    def preprocess_self_cognition_dataset(
        cls,
        dataset_list: Tuple[DATASET_TYPE, Optional[DATASET_TYPE]],
        model_name: Tuple[str, Optional[str]],
        model_author: Tuple[str, Optional[str]],
    ) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
        """Preprocess for self cognition task

        Args:
            dataset_list: The self cognition dataset list
            model_name: The model name
            model_author: The model author

        Returns:
            The processed dataset tuple
        """
        # model_name: Tuple[zh, en]
        assert model_name[0] is not None
        assert model_author[0] is not None
        if len(model_name) == 1 or model_name[1] is None:
            model_name = (model_name[0], model_name[0])
        if len(model_author) == 1 or model_author[1] is None:
            model_author = (model_author[0], model_author[0])
        res_d_list = []
        for dataset in dataset_list:  # train_dataset, val_dataset
            if dataset is None:
                res_d_list.append(dataset)
                continue

            if isinstance(dataset, HfIterableDataset):

                def generate_example(dataset):
                    for row in dataset:
                        if row['tag'] == 'zh':
                            model_n, model_a = model_name[0], model_author[0]
                        else:
                            model_n, model_a = model_name[1], model_author[1]
                        yield {
                            'messages': [{
                                'role':
                                'user',
                                'content':
                                row['query'].replace('{{NAME}}', model_n).replace('{{AUTHOR}}', model_a)
                            }, {
                                'role':
                                'assistant',
                                'content':
                                row['response'].replace('{{NAME}}', model_n).replace('{{AUTHOR}}', model_a)
                            }]
                        }

                dataset = HfIterableDataset.from_generator(generate_example, gen_kwargs={'dataset': dataset})
            else:
                messages = []
                for row in dataset:
                    if row['tag'] == 'zh':
                        model_n, model_a = model_name[0], model_author[0]
                    else:
                        model_n, model_a = model_name[1], model_author[1]

                    messages.append({
                        'messages': [{
                            'role': 'user',
                            'content': row['query'].replace('{{NAME}}', model_n).replace('{{AUTHOR}}', model_a)
                        }, {
                            'role':
                            'assistant',
                            'content':
                            row['response'].replace('{{NAME}}', model_n).replace('{{AUTHOR}}', model_a)
                        }]
                    })
                dataset = HfDataset.from_list(messages)
            res_d_list.append(dataset)
        return tuple(res_d_list)


class HubDatasetLoader(DatasetLoader):

    @classmethod
    def get_dataset(cls,
                    dataset_id: str,
                    subsets: Optional[List[str]],
                    preprocess_func: Union[PreprocessFunc, RowPreprocessor],
                    split: List[str],
                    dataset_sample: int = -1,
                    *,
                    random_state: Optional[RandomState] = None,
                    split_dataset_ratio: float = 0.,
                    remove_useless_columns: bool = True,
                    use_hf: bool = False,
                    **kwargs) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
        if subsets is None:
            subsets = []
        assert len(split) > 0
        if len(subsets) == 0:
            subset_split_list = split
        else:
            subset_split_list = list(itertools.product(subsets, split))

        dataset = cls.load_dataset_from_hub(
            dataset_id,
            subset_split_list,
            use_hf,
            streaming=kwargs.get('streaming', False),
            revision=kwargs.get('revision'),
            force_redownload=kwargs.get('force_redownload', False))

        return cls.post_preprocess(dataset, dataset_sample, random_state, preprocess_func, split_dataset_ratio,
                                   remove_useless_columns, **kwargs)

    @classmethod
    def load_dataset_from_hub(cls,
                              dataset_id: str,
                              subset_split_list: Optional[List[SubsetSplit]],
                              use_hf: bool = False,
                              streaming: bool = False,
                              revision: Optional[str] = None,
                              force_redownload: bool = False) -> Optional[DATASET_TYPE]:
        """Load dataset from hub

        Args:
            dataset_id: The dataset id
            subset_split_list: The subset info list
            use_hf: Using hf hub, default `False` which means ModelScope hub
            streaming: Use streaming mode or not, default `False`
            revision: The dataset revision
            force_redownload: Force Redownload the dataset, default `False`
        Returns:
            The dataset instance

        """
        if subset_split_list is None or len(subset_split_list) == 0:
            return None
        dataset_list = []
        hub = HFHub() if use_hf else MSHub()
        for subset_split in subset_split_list:
            if isinstance(subset_split, str):
                subset_split = ('default', subset_split)
            assert len(subset_split) == 2
            subset_name, split = subset_split

            with safe_ddp_context():
                for i in range(5):
                    try:
                        dataset = hub.load_dataset(dataset_id, subset_name, split, streaming, revision,
                                                   force_redownload)
                    except Exception as e:
                        logger.error(f'Dataset {dataset_id} load failed: subset_name={subset_name},'
                                     f'split={split} with error: {e}')
                    else:
                        break
                else:
                    continue

                if streaming and hasattr(dataset, '_hf_ds'):
                    dataset = dataset._hf_ds
                    if not isinstance(dataset, HfIterableDataset):
                        dataset = dataset.to_iterable_dataset()
                if hasattr(dataset, 'to_hf_dataset'):
                    dataset = dataset.to_hf_dataset()
            dataset_list.append(dataset)

        if len(dataset_list) == 1:
            return dataset_list[0]
        if not streaming:
            return concatenate_datasets(dataset_list)
        else:
            return interleave_datasets(dataset_list)


class LocalDatasetLoader(DatasetLoader):

    @classmethod
    def dataset_get_function(cls,
                             dataset_id: str,
                             subsets: Optional[List[str]],
                             preprocess_func: Union[PreprocessFunc, RowPreprocessor],
                             split: List[str],
                             dataset_sample: int = -1,
                             *,
                             random_state: Optional[RandomState] = None,
                             split_dataset_ratio: float = 0.,
                             remove_useless_columns: bool = True,
                             use_hf: bool = False,
                             **kwargs) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
        dataset = cls.load_dataset_from_local(split, preprocess_func, streaming=kwargs.get('streaming', False))
        return cls.post_preprocess(dataset, dataset_sample, random_state, None, split_dataset_ratio,
                                   remove_useless_columns, **kwargs)

    @classmethod
    def load_dataset_from_local(cls,
                                dataset_path_list: Optional[Union[str, List[str]]],
                                preprocess_func: PreprocessFunc,
                                streaming: bool = False) -> Optional[DATASET_TYPE]:
        if isinstance(dataset_path_list, str):
            dataset_path_list = [dataset_path_list]
        if dataset_path_list is None or len(dataset_path_list) == 0:
            return None
        assert isinstance(dataset_path_list, (list, tuple))

        dataset_list = []
        for dataset_path in dataset_path_list:
            assert isinstance(dataset_path, str)
            df: DataFrame
            if dataset_path.endswith('.csv'):
                dataset = HfDataset.from_csv(dataset_path, na_filter=False)
            elif dataset_path.endswith('.jsonl') or dataset_path.endswith('.json'):
                dataset = HfDataset.from_json(dataset_path)
            else:
                raise ValueError('The custom dataset only supports CSV, JSONL or JSON format.')
            dataset = preprocess_func(dataset)
            if streaming:
                dataset = dataset.to_iterable_dataset()
            dataset_list.append(dataset)

        if len(dataset_list) == 1:
            return dataset_list[0]
        return concatenate_datasets(dataset_list) if not streaming else interleave_datasets(dataset_list)


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
    def parse_dataset_syntax(cls, dataset: str) -> 'DatasetSyntax':
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
                    k = self._encode_key(dataset.ms_dataset_id, 'hf_repo')
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

    def __init__(
            self,
            split_dataset_ratio: float = 0.,
            dataset_seed: Union[int, RandomState] = 42,
            use_hf: Optional[bool] = None,
            load_from_cache_file: bool = False,
            num_proc: int = 1,
            force_redownload: bool = False,
            *,
            # self-cognition
            model_name: Union[Tuple[str, str], List[str], None] = None,
            model_author: Union[Tuple[str, str], List[str], None] = None,
            # streaming
            streaming: bool = False,
            streaming_val_size: int = 0,
            streaming_buffer_size: int = 16384):
        if isinstance(dataset_seed, int):
            dataset_seed = RandomState(dataset_seed)

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

    def load(self, dataset: str) -> Tuple[HfDataset, Optional[HfDataset]]:
        pass

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
    def _parse_datasets(datasets: List[Union[str, Dataset]]) -> List[str]:
        # ms_dataset_id/hf_dataset_id/dataset_path -> dataset_name mapping
        dataset_name_mapping = DatasetNameMapping()

        # register_dataset
        res_datasets: List[str] = []  # dataset_names
        register_idx = 0
        dataset_info = {}
        for dataset in datasets:
            d_info = DatasetSyntax.parse_dataset_syntax(dataset)
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

    def load_datasets(self, datasets: List[Union[str, Dataset]]) -> Tuple[HfDataset, Optional[HfDataset]]:
        datasets: List[str] = self._parse_datasets(datasets)
        train_datasets = []
        val_datasets = []
        for dataset in datasets:
            train_dataset, val_dataset = self.load(dataset)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

        gather_function = interleave_datasets if self.streaming else concatenate_datasets
        if len(train_datasets) > 1:
            train_datasets = gather_function(train_datasets)
        if len(val_datasets) > 1:
            val_datasets = gather_function(val_datasets)
        return train_datasets, val_datasets


def load_datasets(
        datasets: List[Union[str, Dataset]],
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
    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]
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
    return dataset_loader.load_datasets(datasets)
