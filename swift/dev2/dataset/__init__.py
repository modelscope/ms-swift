# Copyright (c) ModelScope Contributors. All rights reserved.
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

# Register the dataset_info.json families at import, mirroring the dev-side package init; the
# hand-written family modules (llm/mllm) arrive with the next batch.
register_dataset_info()

__all__ = [
    'DATASET_MAPPING', 'DATASET_TYPE', 'DatasetInfo', 'DatasetLoader', 'SubsetMeta', 'get_dataset_loader',
    'load_dataset', 'match_dataset_type', 'register_dataset', 'register_dataset_info', 'AlpacaConverter',
    'OpenAIConverter', 'PreferenceConverter', 'ResponseConverter', 'get_converter'
]
