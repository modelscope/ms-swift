# Copyright (c) ModelScope Contributors. All rights reserved.
"""The dev dataset stack, four layers whose files carry the same names as the concepts:

- :mod:`format_converter` -- rewrite any raw row shape into standard ``messages``
- :mod:`preprocessor` -- run a per-row transform over the whole dataset via ``map``, dropping bad rows.
  Includes the two passes that run ``template.encode``: :class:`EncodePreprocessor` keeps what it
  produces, :class:`MeasurePreprocessor` keeps only the token count.
- :mod:`mm_download` -- fetch the media archives multimodal datasets reference by relative path
- :mod:`loader` -- name resolution + load orchestration + the built-in dataset registrations

Directly under this package is the torch-facing layer -- what the DataLoader consumes. Each class there
settles for itself when a row is encoded, so nothing outside has to arrange them:

- :class:`SwiftDataset` -- standard rows in, encoded on access. The base the others inherit.
- :class:`EncodedDataset` -- rows an earlier pass already encoded, so access just reads.
- :class:`PackingDataset` -- serves *groups* of rows that together fill one training sequence.
- :class:`IterablePackingDataset` -- the same for a stream, which cannot be planned ahead.
- :class:`LazyLLMDataset` -- the wrapper :class:`SwiftDataset` replaces, kept until the pipelines move.

Importing this package registers every built-in dataset. :mod:`loader` handles the code-declared
families, then :func:`register_dataset_info` reads ``dataset_info.json``. The JSON pass runs from
this module's body rather than relying on import order, so it stays after the code-declared
families however the imports above are sorted -- and it uses the default ``exist_ok=False``, so a
dataset accidentally declared in both places fails loudly rather than one silently shadowing the
other.
"""
from .encoded_dataset import EncodedDataset
from .lazy_dataset import LazyLLMDataset
from .loader import (
                     DATASET_MAPPING,
                     DATASET_TYPE,
                     DatasetInfo,
                     DatasetLoader,
                     SubsetMeta,
                     get_dataset_loader,
                     load_dataset,
                     match_dataset_type,
                     register_dataset,
                     register_dataset_info,
)
from .packing import IterablePackingDataset, PackingDataset
from .preprocessor import EncodePreprocessor, MeasurePreprocessor, Preprocessor
from .swift_dataset import SwiftDataset

__all__ = [
    'DATASET_MAPPING', 'DATASET_TYPE', 'DatasetInfo', 'DatasetLoader', 'SubsetMeta', 'Preprocessor',
    'get_dataset_loader', 'load_dataset', 'match_dataset_type', 'register_dataset', 'register_dataset_info',
    'LazyLLMDataset', 'PackingDataset', 'IterablePackingDataset', 'EncodePreprocessor', 'MeasurePreprocessor',
    'SwiftDataset', 'EncodedDataset'
]

register_dataset_info()
