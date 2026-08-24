# Copyright (c) ModelScope Contributors. All rights reserved.
"""Dataset loading: name resolution + load orchestration (:mod:`base`), plus the built-in dataset
registrations (:mod:`llm`, :mod:`mllm`, and the pure-declaration entries in ``dataset_info.json``).

Importing this package registers every built-in dataset. The order matters: the code-declared
families (:mod:`llm`, :mod:`mllm`) go first, then :func:`register_dataset_info` reads the JSON. That
call is made by the parent package after these imports, and uses the default ``exist_ok=False`` so a
dataset accidentally declared in both places fails loudly rather than one silently shadowing the
other.
"""
from . import llm  # noqa: F401  (import for its registration side effects)
from . import mllm  # noqa: F401  (import for its registration side effects)
from .base import (DATASET_MAPPING, DATASET_TYPE, DatasetInfo, DatasetLoader, SubsetMeta, get_dataset_loader,
                   load_dataset, match_dataset_type, register_dataset, register_dataset_info)

__all__ = [
    'DATASET_MAPPING', 'DATASET_TYPE', 'DatasetInfo', 'DatasetLoader', 'SubsetMeta', 'get_dataset_loader',
    'load_dataset', 'match_dataset_type', 'register_dataset', 'register_dataset_info'
]
