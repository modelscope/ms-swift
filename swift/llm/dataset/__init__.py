# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import tempfile

import datasets.config
import datasets.fingerprint
from datasets import disable_caching
from modelscope.hub.utils.utils import get_cache_dir

from swift.utils.torch_utils import _find_local_mac
from . import dataset
from .loader import DATASET_TYPE, load_dataset
from .media import MediaResource
from .preprocessor import (AlpacaPreprocessor, AutoPreprocessor, MessagesPreprocessor, ResponsePreprocessor,
                           RowPreprocessor, standard_keys)
from .register import DATASET_MAPPING, DatasetMeta, register_dataset, register_dataset_info
from .utils import (ConstantLengthDataset, EncodePreprocessor, GetLengthPreprocessor, LazyLLMDataset,
                    PackingPreprocessor, sample_dataset)

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


def _new_get_temporary_cache_files_directory(*args, **kwargs):
    global DATASET_TEMP_DIR
    if DATASET_TEMP_DIR is None:
        tmp_dir = os.path.join(get_cache_dir(), 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        DATASET_TEMP_DIR = tempfile.TemporaryDirectory(prefix=datasets.config.TEMP_CACHE_DIR_PREFIX, dir=tmp_dir)

    return DATASET_TEMP_DIR.name


datasets.fingerprint.update_fingerprint = _update_fingerprint_mac
datasets.arrow_dataset.update_fingerprint = _update_fingerprint_mac
datasets.fingerprint.get_temporary_cache_files_directory = _new_get_temporary_cache_files_directory
datasets.arrow_dataset.get_temporary_cache_files_directory = _new_get_temporary_cache_files_directory
DATASET_TEMP_DIR = None
register_dataset_info()
disable_caching()
