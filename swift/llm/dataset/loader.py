# Copyright (c) Alibaba, Inc. and its affiliates.
import itertools
import os
import shutil
from abc import ABC
from tempfile import TemporaryDirectory
from typing import List, Literal, Optional, Tuple, Union, Dict, Any, Callable

import numpy as np
from datasets import Dataset as HfDataset, IterableDataset as HfIterableDataset
from datasets import concatenate_datasets, interleave_datasets
from modelscope.hub.api import ModelScopeConfig
from modelscope.utils.config_ds import MS_CACHE_HOME
from numpy.random import RandomState
from pandas import DataFrame
from transformers.utils import strtobool

from swift.hub.hub import HFHub, MSHub
from swift.llm.dataset.preprocess import RowPreprocessor
from swift.utils import get_logger
from swift.utils import get_seed
from swift.utils.io_utils import download_files
from swift.utils.torch_utils import safe_ddp_context
from swift.utils.utils import _safe_split

DATASET_TYPE = Union[HfDataset, HfIterableDataset]

SubsetSplit = Union[str, Tuple[str, str], List[str]]
DATASET_MAPPING: Dict[str, Dict[str, Any]] = {}
PreprocessFunc = Callable[[DATASET_TYPE], DATASET_TYPE]
logger = get_logger()


class DatasetLoader(ABC):

    @classmethod
    def dataset_get_function(cls,
                             dataset_id: str,
                             subsets: Optional[List[str]],
                             preprocess_func: Union[PreprocessFunc, RowPreprocessor],
                             split: List[str],
                             dataset_sample: int = -1,
                             *,
                             random_state: Optional[RandomState] = None,
                             dataset_test_ratio: float = 0.,
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
            dataset_test_ratio: The dataset split ratio, default `0`
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
        standard_keys = {
            'messages', 'rejected_response', 'images', 'objects', 'videos', 'audios', 'tools', 'label'
        }
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
    def sample_dataset(cls, dataset: HfDataset, dataset_sample: int,
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
            dataset_test_ratio: float = 0.,
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
            dataset_test_ratio: The dataset split ratio, Default `0`
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
            assert 0 <= dataset_test_ratio <= 1
            if dataset_test_ratio == 1:
                # The validation scenario, switch all data to validation set
                train_dataset, val_dataset = None, train_dataset
                val_sample = dataset_sample
                assert val_sample <= len(
                    val_dataset), f'dataset_sample: {dataset_sample}, len(val_dataset): {len(val_dataset)}'
                val_dataset = cls.sample_dataset(val_dataset, val_sample, random_state)
            else:
                # The training scenario
                if dataset_test_ratio == 0:
                    train_sample = dataset_sample
                    val_dataset = None
                else:
                    # Avoid having a high train_sample causing a high val_sample.
                    _train_len = min(len(train_dataset), dataset_sample)
                    val_sample = max(int(_train_len * dataset_test_ratio), 1)
                    train_sample = dataset_sample - val_sample
                    assert isinstance(val_sample, int)
                    train_dataset, val_dataset = train_dataset.train_test_split(
                        test_size=val_sample, seed=get_seed(random_state),
                        load_from_cache_file=kwargs.get('dataset_enable_cache', False)).values()

                assert train_sample > 0
                train_dataset = cls.sample_dataset(train_dataset, train_sample, random_state)
        else:
            val_dataset = None
            if dataset_test_ratio == 1:
                train_dataset, val_dataset = None, train_dataset
            else:
                streaming_val_size = kwargs.get('streaming_val_size', 0)
                streaming_buffer_size = kwargs.get('streaming_buffer_size', 16384)
                if streaming_val_size > 0:
                    train_dataset = train_dataset.shuffle(seed=get_seed(random_state),
                                                          buffer_size=streaming_buffer_size)
                    val_dataset = train_dataset.take(int(streaming_val_size))
                    train_dataset = train_dataset.skip(int(streaming_val_size))

        res = []
        for dataset in [train_dataset, val_dataset]:
            if dataset is not None and preprocess_func is not None:
                dataset = preprocess_func(dataset)
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
                download_files(url, temp_fpath, cookies)
                shutil.copy2(temp_fpath, local_fpath)

        return local_dir

    @classmethod
    def load_dataset(
            cls,
            dataset_name_list: Union[List[str], str],
            dataset_test_ratio: float = 0.,
            dataset_seed: Union[int, RandomState] = 42,
            check_dataset_strategy: Literal['none', 'discard', 'error', 'warning'] = 'none',
            *,
            # for self-cognition
            model_name: Union[Tuple[str, str], List[str], None] = None,
            model_author: Union[Tuple[str, str], List[str], None] = None,
            streaming: bool = False,
            **kwargs) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
        """The interface to load any registered dataset

        Args:
            dataset_name_list: The dataset name list
            dataset_test_ratio: The dataset split ratio
            dataset_seed: The dataset random seed
            check_dataset_strategy: The check_dataset_strategy
            model_name: Model name in self-cognition task
            model_author: Model author in self-cognition task
            streaming: Streaming mode or not
        Returns:
            The train dataset and val dataset
        """
        if isinstance(dataset_name_list, str):
            dataset_name_list = [dataset_name_list]
        train_dataset_list: List[DATASET_TYPE] = []
        val_dataset_list: List[DATASET_TYPE] = []

        # dataset_id_or_path -> dataset_name
        dataset_name_list = cls.dataset_id_to_name(dataset_name_list)
        for dataset_name in dataset_name_list:
            use_hf, dataset_name, subsets, dataset_sample = parse_dataset_name(dataset_name)
            dataset_info = DATASET_MAPPING[dataset_name]
            if subsets is None:
                subsets = dataset_info['subsets']
            if dataset_sample == -1:
                dataset_sample = dataset_info.get('dataset_sample', -1)
            if isinstance(dataset_seed, int):
                random_state = RandomState(dataset_seed)
            else:
                random_state = dataset_seed

            get_function = dataset_info['get_function'] or cls.dataset_get_function
            is_local = dataset_info.get('is_local', False)
            dataset_id_or_path = dataset_info['dataset_id_or_path']
            remove_useless_columns = dataset_info.get('remove_useless_columns', True)

            if not is_local:
                dataset_str_f = 'Downloading the dataset from {hub}, dataset_id: {dataset_id}'
                if not dataset_id_or_path:
                    use_hf = True
                if use_hf:
                    dataset_id_or_path = dataset_info['hf_dataset_id']
                    dataset_str = dataset_str_f.format(hub='HuggingFace', dataset_id=dataset_id_or_path)
                else:
                    dataset_str = dataset_str_f.format(hub='ModelScope', dataset_id=dataset_id_or_path)
                logger.info(dataset_str)
                assert dataset_id_or_path is not None, (f'dataset_name: {dataset_name}, use_hf: {use_hf}, '
                                                        f'dataset_id_or_path: {dataset_id_or_path}.')
            dataset = get_function(
                dataset_id_or_path,
                subsets,
                dataset_info['preprocess_func'],
                dataset_info['split'],
                dataset_sample,
                random_state=random_state,
                dataset_test_ratio=dataset_test_ratio,
                remove_useless_columns=remove_useless_columns,
                use_hf=use_hf,
                revision=dataset_info.get('revision'),
                **kwargs)

            if dataset_name == 'self-cognition':
                assert model_name is not None and model_author is not None
                dataset = cls.preprocess_self_cognition_dataset(dataset, model_name, model_author)

            train_d: HfDataset
            if isinstance(dataset, (list, tuple)):
                train_d, val_d = dataset
            else:
                train_d, val_d = dataset, None

            assert train_d is not None or val_d is not None
            if train_d is not None:
                train_dataset_list.append(train_d)
            if val_d is not None:
                val_dataset_list.append(val_d)

        if len(train_dataset_list) > 1:
            train_dataset = concatenate_datasets(train_dataset_list) if not streaming else interleave_datasets(
                train_dataset_list)
        else:
            train_dataset = train_dataset_list[0] if train_dataset_list else None

        if len(val_dataset_list) > 1:
            val_dataset = concatenate_datasets(val_dataset_list) if not streaming else interleave_datasets(
                val_dataset_list)
        else:
            val_dataset = val_dataset_list[0] if val_dataset_list else None
        if check_dataset_strategy != 'none':
            logger.info('check dataset...')
            logger.info(f"check_dataset_strategy: '{check_dataset_strategy}'")

        return train_dataset, val_dataset

    @classmethod
    def dataset_id_to_name(cls, dataset_name_list: List[str]) -> List[str]:
        # register dataset_id (ms/hf). Convert dataset_id to dataset_name.
        ms_dataset_mapping = {}
        hf_dataset_mapping = {}
        for k_name, container in zip(['dataset_id_or_path', 'hf_dataset_id'], [ms_dataset_mapping, hf_dataset_mapping]):
            for k, v in DATASET_MAPPING.items():
                if v.get(k_name) is None or not v.get('is_main', True):
                    continue
                if v[k_name] not in container:
                    container[v[k_name]] = []
                container[v[k_name]].append(k)

        res_dataset = []
        dataset_list = []
        # Add dataset_id or dataset_path to dataset_list, and add dataset_name to res_dataset.
        for d in dataset_name_list:
            use_hf, d_name = parse_dataset_name(d)[:2]
            if d_name in DATASET_MAPPING:
                res_dataset.append(d)
            else:
                dataset_list.append((d, use_hf, d_name))

        extra_dataset = []
        for d, use_hf, d_id_or_path in dataset_list:
            dataset_mapping = hf_dataset_mapping if use_hf else ms_dataset_mapping
            if d_id_or_path in dataset_mapping:
                # Add the dataset_name corresponding to the dataset_id to res_dataset.
                for d_name in dataset_mapping[d_id_or_path]:
                    res_dataset.append(d.replace(d_id_or_path, d_name))
            else:
                # This dataset needs to be registered.
                extra_dataset.append((d, use_hf, d_id_or_path))

        for i, (d, use_hf, d_id_or_path) in enumerate(extra_dataset):
            d_info = {}
            d_name = f'_{i}'
            if os.path.isfile(d_id_or_path):
                d_info['dataset_path'] = d_id_or_path
            else:
                if d_id_or_path.startswith('/'):
                    raise ValueError(f"path: '{d_id_or_path}' not found")
                if use_hf:
                    d_info['hf_dataset_id'] = d_id_or_path
                else:
                    d_info['dataset_id'] = d_id_or_path
            from swift.llm.dataset.register import register_single_dataset
            register_single_dataset(d_name, d_info)
            res_dataset.append(d.replace(d_id_or_path, d_name))
        return res_dataset

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
                            'messages': [
                                {
                                    'role': 'user',
                                    'content': row['query'].replace('{{NAME}}', model_n).replace('{{AUTHOR}}', model_a)
                                },
                                {
                                    'role': 'assistant',
                                    'content': row['response'].replace('{{NAME}}', model_n).replace('{{AUTHOR}}', model_a)
                                }
                            ]
                        }

                dataset = HfIterableDataset.from_generator(generate_example, gen_kwargs={'dataset': dataset})
            else:
                messages = []
                for row in dataset:
                    if row['tag'] == 'zh':
                        model_n, model_a = model_name[0], model_author[0]
                    else:
                        model_n, model_a = model_name[1], model_author[1]

                    messages.append({'messages':
                        [{
                                    'role': 'user',
                                    'content': row['query'].replace('{{NAME}}', model_n).replace('{{AUTHOR}}', model_a)
                                },
                    {
                                    'role': 'assistant',
                                    'content': row['response'].replace('{{NAME}}', model_n).replace('{{AUTHOR}}', model_a)
                                }]})
                dataset = HfDataset.from_list(messages)
            res_d_list.append(dataset)
        return tuple(res_d_list)


class HubDatasetLoader(DatasetLoader):

    @classmethod
    def dataset_get_function(cls,
                             dataset_id: str,
                             subsets: Optional[List[str]],
                             preprocess_func: Union[PreprocessFunc, RowPreprocessor],
                             split: List[str],
                             dataset_sample: int = -1,
                             *,
                             random_state: Optional[RandomState] = None,
                             dataset_test_ratio: float = 0.,
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
            force_redownload=kwargs.get('force_redownload'))

        return cls.post_preprocess(dataset, dataset_sample, random_state, preprocess_func, dataset_test_ratio,
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
                             dataset_test_ratio: float = 0.,
                             remove_useless_columns: bool = True,
                             use_hf: bool = False,
                             **kwargs) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
        dataset = cls.load_dataset_from_local(split, preprocess_func, streaming=kwargs.get('streaming', False))
        return cls.post_preprocess(dataset, dataset_sample, random_state, None, dataset_test_ratio,
                                   remove_useless_columns,
                                   **kwargs)

    @classmethod
    def load_dataset_from_local(cls, dataset_path_list: Optional[Union[str, List[str]]],
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


def parse_dataset_name(dataset_name: str) -> Tuple[bool, str, List[str], int]:
    """Parse the dataset name from the command line

    Args:
        dataset_name: The dataset name

    Returns:
        Dataset infos, including:
        1. use hf or not
        2. dataset name
        3. subset list
        4. dataset sample number
    """
    # HF::dataset_name:subset1/subset2/subset3#dataset_sample
    use_hf, other = _safe_split(dataset_name, '::', False)
    if use_hf is None:
        use_hf = strtobool(os.environ.get('USE_HF', 'False'))
    elif isinstance(use_hf, str):
        use_hf = {'hf': 1, 'ms': 0}[use_hf.lower()]
    if os.path.isfile(other):
        part1, dataset_sample = other, None
    else:
        part1, dataset_sample = _safe_split(other, '#', True, 'right')
    if os.path.isfile(part1):
        dataset_name, subsets = part1, None
    else:
        dataset_name, subsets = _safe_split(part1, ':', True)

    if subsets is not None:
        subset_list = subsets.split('/')
        subset_list = [subset.strip() for subset in subset_list]
    else:
        subset_list = None
    if dataset_sample is None:
        dataset_sample = -1
    else:
        dataset_sample = int(dataset_sample)
    return tuple(t.strip() if isinstance(t, str) else t for t in [use_hf, dataset_name, subset_list, dataset_sample])


def dataset_name_exists(dataset_list: List[str], dataset_name: str) -> List[int]:
    """Check whether dataset name exists

    Args:
        dataset_list: The dataset list
        dataset_name: The dataset name

    Returns:
        TODO
    """
    dataset_name = parse_dataset_name(dataset_name)[1]
    cache_name_list = [parse_dataset_name(dataset)[1] for dataset in dataset_list]
    res = []
    for i, cache_name in enumerate(cache_name_list):
        if cache_name == dataset_name:
            res.append(i)
    return res
