# Copyright (c) ModelScope Contributors. All rights reserved.
"""The unified dataset storage layer: one format registry behind the encoded store and the data lake.

:mod:`.base` holds the mechanism -- the :class:`DatasetStorage` interface, the :obj:`STORAGE_MAPPING`
registry with :func:`register_storage`, the :func:`detect_storage_format` probe, and the public
:func:`write_dataset_store` / :func:`load_dataset_store`. :mod:`.formats` holds the concrete formats,
and :mod:`.writers` the write-side writer family a run streams rows through (:func:`open_store_writer`
/ :func:`store_output_complete`).
Importing this package pulls in :mod:`.formats` for its registration side effects, so every format is
in :obj:`STORAGE_MAPPING` before the first read or write; :mod:`.writers` is imported last, so its
choice reads a registry that already holds every format.
"""
from .base import (CORE_STORE_FIELDS, KNOWN_STORE_FIELDS, STORAGE_MAPPING, DatasetStorage,
                   detect_storage_format, load_dataset_store, register_storage, store_path,
                   validate_store_fields, validate_store_format, write_dataset_store)
# Imported for registration: each ``@register_storage`` subclass adds itself to STORAGE_MAPPING.
from .formats import (ArrowStorage, BinStorage, DatasetsBuilderStorage, JsonlStorage, LanceStorage,
                      OrcStorage)
# Imported after the registry and the formats: the writer family is chosen by a format's declared
# ``incremental`` capability bit.
from .writers import open_store_writer, store_output_complete

__all__ = [
    'CORE_STORE_FIELDS', 'KNOWN_STORE_FIELDS', 'DatasetStorage', 'STORAGE_MAPPING', 'register_storage',
    'detect_storage_format', 'store_path', 'validate_store_fields', 'validate_store_format',
    'write_dataset_store', 'load_dataset_store', 'open_store_writer', 'store_output_complete',
    'ArrowStorage', 'JsonlStorage', 'BinStorage', 'OrcStorage', 'LanceStorage', 'DatasetsBuilderStorage'
]
