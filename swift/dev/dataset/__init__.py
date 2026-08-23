# Copyright (c) ModelScope Contributors. All rights reserved.
"""The dev dataset stack: name resolution + loading (:mod:`base`), input-format conversion
(:mod:`format`), the row-transform execution layer (:mod:`preprocessor`), and multimodal resource
downloading (:mod:`mm_download`).

Importing this package registers the pilot dataset families (:mod:`llm`), so :func:`load_dataset`
can resolve them by name.
"""
from .base import (DATASET_MAPPING, DATASET_TYPE, DatasetInfo, DatasetLoader, SubsetMeta, get_dataset_loader,
                   load_dataset, match_dataset_type, register_dataset)
from .preprocessor import Preprocessor
from . import llm  # noqa: F401  (import for its registration side effects)

__all__ = [
    'DATASET_MAPPING', 'DATASET_TYPE', 'DatasetInfo', 'DatasetLoader', 'SubsetMeta', 'Preprocessor',
    'get_dataset_loader', 'load_dataset', 'match_dataset_type', 'register_dataset'
]
