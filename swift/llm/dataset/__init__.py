# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect

import datasets.fingerprint
from datasets import Dataset as HfDataset

from ..utils import get_temporary_cache_files_directory
from . import dataset
from .loader import DATASET_TYPE, load_dataset
from .media import MediaResource
from .preprocessor import (AlpacaPreprocessor, AutoPreprocessor, MessagesPreprocessor, ResponsePreprocessor,
                           RowPreprocessor)
from .register import DATASET_MAPPING, DatasetMeta, SubsetDataset, register_dataset, register_dataset_info
from .utils import (EncodePreprocessor, GetLengthPreprocessor, IterablePackingDataset, LazyLLMDataset, PackingDataset,
                    sample_dataset)

update_fingerprint_origin = datasets.fingerprint.update_fingerprint


def update_fingerprint(fingerprint, transform, transform_args):
    if 'function' in transform_args:
        # Calculate the hash using the source code.
        if hasattr(transform_args['function'], '__self__'):
            function = inspect.getsource(transform_args['function'].__self__.__class__)
        else:
            function = inspect.getsource(transform_args['function'])
        transform_args['function'] = (transform_args['function'], function)
    return update_fingerprint_origin(fingerprint, transform, transform_args)


datasets.fingerprint.update_fingerprint = update_fingerprint
datasets.arrow_dataset.update_fingerprint = update_fingerprint
datasets.fingerprint.get_temporary_cache_files_directory = get_temporary_cache_files_directory
datasets.arrow_dataset.get_temporary_cache_files_directory = get_temporary_cache_files_directory
register_dataset_info()
