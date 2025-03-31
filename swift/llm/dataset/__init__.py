# Copyright (c) Alibaba, Inc. and its affiliates.
import datasets.fingerprint
from datasets import Dataset as HfDataset
from datasets import disable_caching

from swift.utils.torch_utils import _find_local_mac
from ..utils import get_temporary_cache_files_directory
from . import dataset
from .loader import DATASET_TYPE, load_dataset
from .media import MediaResource
from .preprocessor import (AlpacaPreprocessor, AutoPreprocessor, MessagesPreprocessor, ResponsePreprocessor,
                           RowPreprocessor)
from .register import DATASET_MAPPING, DatasetMeta, SubsetDataset, register_dataset, register_dataset_info
from .utils import (EncodePreprocessor, GetLengthPreprocessor, IterablePackingDataset, LazyLLMDataset, PackingDataset,
                    sample_dataset)

_update_fingerprint = datasets.fingerprint.update_fingerprint
_get_temporary_cache_files_directory = datasets.fingerprint.get_temporary_cache_files_directory


def _update_fingerprint_mac(*args, **kwargs):
    # Prevent different nodes use the same location in unique shared disk
    mac = _find_local_mac().replace(':', '')
    fp = _update_fingerprint(*args, **kwargs)
    fp += '-' + mac
    if len(fp) > 64:
        fp = fp[:64]
    return fp


datasets.fingerprint.update_fingerprint = _update_fingerprint_mac
datasets.arrow_dataset.update_fingerprint = _update_fingerprint_mac
datasets.fingerprint.get_temporary_cache_files_directory = get_temporary_cache_files_directory
datasets.arrow_dataset.get_temporary_cache_files_directory = get_temporary_cache_files_directory
register_dataset_info()
disable_caching()
