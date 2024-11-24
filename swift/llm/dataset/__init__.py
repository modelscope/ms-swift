# Copyright (c) Alibaba, Inc. and its affiliates.

import datasets.fingerprint

from swift.utils.torch_utils import _find_local_mac
from . import dataset
from .loader import DATASET_TYPE, load_dataset, standard_keys
from .media import MediaResource
from .preprocessor import (AlpacaPreprocessor, AutoPreprocessor, MessagesPreprocessor, ResponsePreprocessor,
                           RowPreprocessor)
from .register import DATASET_MAPPING, register_dataset, register_dataset_info
from .utils import (ConstantLengthDataset, EncodePreprocessor, GetLengthPreprocessor, LazyLLMDataset,
                    LLMIterableDataset, sample_dataset)


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
