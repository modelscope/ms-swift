# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import SwiftDataset
from .encoded_dataset import EncodedDataset
from .lazy_dataset import LazyLLMDataset
from .storage import (  # noqa: F401  (re-exported like the dev-side package init)
    CORE_STORE_FIELDS,
    KNOWN_STORE_FIELDS,
    STORAGE_MAPPING,
    DatasetStorage,
    detect_storage_format,
    load_dataset_store,
    register_storage,
    store_path,
    validate_store_fields,
    validate_store_format,
    write_dataset_store,
)
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
from .format_converter import (  # noqa: F401  (re-exported for callers of the migrated converter)
    AlpacaConverter,
    OpenAIConverter,
    PreferenceConverter,
    ResponseConverter,
    get_converter,
)
from .preprocessor import (  # noqa: F401  (re-exported like the dev-side package init)
    EncodePreprocessor,
    MeasurePreprocessor,
    Preprocessor,
)
from .packing import IterablePackingDataset, PackingDataset

# Register the dataset_info.json families at import, mirroring the dev-side package init.
register_dataset_info()

__all__ = [
    'DATASET_MAPPING', 'DATASET_TYPE', 'DatasetInfo', 'DatasetLoader', 'SubsetMeta', 'Preprocessor',
    'SwiftDataset', 'EncodedDataset', 'LazyLLMDataset', 'PackingDataset', 'IterablePackingDataset',
    'CORE_STORE_FIELDS', 'KNOWN_STORE_FIELDS', 'DatasetStorage', 'STORAGE_MAPPING',
    'register_storage', 'detect_storage_format', 'store_path', 'validate_store_fields',
    'validate_store_format', 'write_dataset_store', 'load_dataset_store',
    'get_dataset_loader', 'load_dataset', 'match_dataset_type', 'register_dataset',
    'register_dataset_info', 'EncodePreprocessor', 'MeasurePreprocessor', 'AlpacaConverter',
    'OpenAIConverter', 'PreferenceConverter', 'ResponseConverter', 'get_converter'
]
