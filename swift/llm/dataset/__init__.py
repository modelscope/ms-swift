# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import datasets.fingerprint

from swift.utils.torch_utils import _find_local_mac
from .loader import load_dataset
from .media import MediaResource
from .preprocess import (AlpacaPreprocessor, ClsPreprocessor, ComposePreprocessor, ConversationsPreprocessor,
                         ListPreprocessor, PreprocessFunc, RenameColumnsPreprocessor, SmartPreprocessor,
                         TextGenerationPreprocessor, multimodal_keys, multimodal_tags)
from .register import register_dataset, register_dataset_info, DATASET_MAPPING
from .utils import (ConstantLengthDataset, LazyLLMDataset, LLMDataset, LLMIterableDataset, dataset_map, print_example,
                    sort_by_max_length, stat_dataset)
from . import dataset


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
register_dataset_info()