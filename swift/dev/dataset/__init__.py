# Copyright (c) ModelScope Contributors. All rights reserved.
"""The dev dataset stack, five layers whose files carry the same names as the concepts:

- :mod:`format_converter` -- rewrite any raw row shape into standard ``messages``
- :mod:`preprocessor` -- run the per-row transform over the whole dataset, dropping bad rows
- :mod:`mm_download` -- fetch the media archives multimodal datasets reference by relative path
- :mod:`loader` -- name resolution + load orchestration + the built-in dataset registrations
- torch-facing wrappers directly under this package -- what the DataLoader consumes:
  :class:`LazyLLMDataset`, :class:`PackingDataset`, :class:`IterablePackingDataset`, and the
  encode / add-length preprocessors that feed them.

Importing this package registers every built-in dataset. :mod:`loader` handles the code-declared
families, then :func:`register_dataset_info` reads ``dataset_info.json``. The JSON pass runs from
this module's body rather than relying on import order, so it stays after the code-declared
families however the imports above are sorted -- and it uses the default ``exist_ok=False``, so a
dataset accidentally declared in both places fails loudly rather than one silently shadowing the
other.
"""
from .add_length_preprocessor import AddLengthPreprocessor
from .encode_preprocessor import EncodePreprocessor
from .lazy_dataset import LazyLLMDataset
from .loader import (DATASET_MAPPING, DATASET_TYPE, DatasetInfo, DatasetLoader, SubsetMeta, get_dataset_loader,
                     load_dataset, match_dataset_type, register_dataset, register_dataset_info)
from .packing import IterablePackingDataset, PackingDataset
from .preprocessor import Preprocessor

__all__ = [
    'DATASET_MAPPING', 'DATASET_TYPE', 'DatasetInfo', 'DatasetLoader', 'SubsetMeta', 'Preprocessor',
    'get_dataset_loader', 'load_dataset', 'match_dataset_type', 'register_dataset', 'register_dataset_info',
    'LazyLLMDataset', 'PackingDataset', 'IterablePackingDataset', 'EncodePreprocessor', 'AddLengthPreprocessor'
]

register_dataset_info()
