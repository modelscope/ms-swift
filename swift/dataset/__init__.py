# Copyright (c) Alibaba, Inc. and its affiliates.
import datasets.fingerprint
from datasets import Dataset as HfDataset

from . import dataset
from .loader import DATASET_TYPE, DatasetLoader, DatasetSyntax, load_dataset
from .media import MediaResource
from .packing import IterablePackingDataset, PackingDataset
from .preprocessor import (AlpacaPreprocessor, AutoPreprocessor, MessagesPreprocessor, ResponsePreprocessor,
                           RowPreprocessor)
from .register import (DATASET_MAPPING, DatasetMeta, SubsetDataset, get_dataset_list, register_dataset,
                       register_dataset_info)
from .utils import (AddLengthPreprocessor, EncodePreprocessor, LazyLLMDataset, get_temporary_cache_files_directory,
                    sample_dataset)

datasets.fingerprint.get_temporary_cache_files_directory = get_temporary_cache_files_directory
datasets.arrow_dataset.get_temporary_cache_files_directory = get_temporary_cache_files_directory
register_dataset_info()
