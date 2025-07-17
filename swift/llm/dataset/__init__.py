# Copyright (c) Alibaba, Inc. and its affiliates.
import datasets.fingerprint
from datasets import Dataset as HfDataset

from ..utils import get_temporary_cache_files_directory
from . import dataset
from .loader import DATASET_TYPE, load_dataset
from .media import MediaResource
from .preprocessor import (AlpacaPreprocessor, AutoPreprocessor, MessagesPreprocessor, ResponsePreprocessor,
                           RowPreprocessor)
from .register import DATASET_MAPPING, DatasetMeta, SubsetDataset, register_dataset, register_dataset_info
from .utils import EncodePreprocessor, IterablePackingDataset, LazyLLMDataset, PackingDataset, sample_dataset

datasets.fingerprint.get_temporary_cache_files_directory = get_temporary_cache_files_directory
datasets.arrow_dataset.get_temporary_cache_files_directory = get_temporary_cache_files_directory
register_dataset_info()
