# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import datasets.fingerprint
from .preprocess import (AlpacaPreprocessor, ClsPreprocessor, ComposePreprocessor,
                                          ConversationsPreprocessor,
                                          ListPreprocessor, PreprocessFunc, RenameColumnsPreprocessor,
                                          SmartPreprocessor,
                                          TextGenerationPreprocessor, standard_keys, standard_tags)
from .loader import DatasetLoader, HubDatasetLoader, LocalDatasetLoader, dataset_name_exists, parse_dataset_name, DATASET_MAPPING
from .media import MediaResource
from .register import register_dataset, register_local_dataset, register_dataset_info_file, register_single_dataset
from .utils import dataset_map, stat_dataset, LLMDataset, LLMIterableDataset, LazyLLMDataset, ConstantLengthDataset, print_example, sort_by_max_length
from swift.utils.torch_utils import _find_local_mac


def _update_fingerprint_mac(*args, **kwargs):
    # Prevent different nodes use the same location in unique shared disk
    mac = _find_local_mac().replace(':', '')
    fp = datasets.fingerprint._update_fingerprint(*args, **kwargs)
    fp += '-' + mac
    if len(fp) > 64:
        fp = fp[:64]
    return fp


datasets.fingerprint._update_fingerprint = datasets.fingerprint.update_fingerprint
datasets.fingerprint.update_fingerprint = _update_fingerprint_mac
datasets.arrow_dataset.update_fingerprint = _update_fingerprint_mac
