# Copyright (c) ModelScope Contributors. All rights reserved.
"""The concrete storage formats registered into :obj:`~swift.dev.dataset.storage.base.STORAGE_MAPPING`.

Each class is one format: ``arrow``/``jsonl``/``bin`` are the store backends ``--store_format`` selects
(arrow and jsonl are writable, bin is the reserved pretraining layout), and ``orc``/``lance``/``builder``
are the read-only data-lake sources :meth:`DatasetLoader.build_dataset` routes local paths through. A
read-only format implements only ``read`` and inherits the base ``write`` refusal; adding a format is one
more ``@register_storage`` subclass here, with no change to the dispatch in :mod:`.base` or the loader.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional, Sequence

from swift.dev.utils import get_logger

from .base import DataType, DatasetStorage, register_storage, restrict_columns, validate_store_fields

logger = get_logger()

__all__ = [
    'ArrowStorage', 'JsonlStorage', 'BinStorage', 'OrcStorage', 'LanceStorage', 'DatasetsBuilderStorage'
]


@register_storage
class ArrowStorage(DatasetStorage):
    """The ``save_to_disk`` store: a directory holding an Arrow table plus ``state.json``/``dataset_info.json``.

    This is the default ``--store_format`` for ``swift export --to_cached_dataset`` and what ``swift infer``
    writes: a single table read back with ``load_from_disk``, directly trainable with no re-encoding.
    Detected by the ``state.json`` marker rather than an extension -- see :func:`detect_storage_format`.
    """

    name = 'arrow'
    writable = True

    def read(self, path: str, *, split: str = 'train', **kwargs) -> Any:
        from datasets import load_from_disk
        # A save_to_disk store is one table with no splits, so ``split`` is accepted for signature parity
        # with the other formats but not used; the loading kwargs (num_proc, streaming, ...) do not apply
        # to an in-place memory-map either.
        return load_from_disk(path)

    def write(self, data: DataType, path: str, *, fields: Optional[Sequence[str]] = None) -> str:
        from datasets import Dataset as HfDataset

        dataset = data if isinstance(data, HfDataset) else HfDataset.from_list(list(data))
        dataset = restrict_columns(dataset, fields)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        dataset.save_to_disk(path)
        logger.info(f'dataset_store: wrote arrow `{path}` ({len(dataset)} rows, columns={dataset.column_names})')
        return path


@register_storage
class JsonlStorage(DatasetStorage):
    """A plain ``.jsonl`` dump: one JSON object per line, portable and human-readable.

    The default ``--store_format`` for ``swift infer``, and resumable (unlike arrow, which writes the whole
    run once at the end). Both a ``.jsonl`` and a ``.json`` single file read back through the ``json``
    builder, so they share this format.
    """

    name = 'jsonl'
    extensions = ('jsonl', 'json')
    writable = True
    incremental = True
    store_suffix = '.jsonl'

    def read(self, path: str, *, split: str = 'train', **kwargs) -> Any:
        from datasets import load_dataset as hf_load_dataset
        return hf_load_dataset('json', data_files=path, split=split, **kwargs)

    def write(self, data: DataType, path: str, *, fields: Optional[Sequence[str]] = None) -> str:
        from datasets import Dataset as HfDataset

        if isinstance(data, HfDataset):
            # Iterating an HF Dataset yields one row dict at a time, so a large store is not held in memory.
            rows = restrict_columns(data, fields)
        else:
            rows = list(data)
            if fields is not None and rows:
                available = set().union(*[set(r.keys()) for r in rows])
                keep = validate_store_fields(fields, available)
                rows = [{k: r.get(k) for k in keep} for r in rows]
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        logger.info(f'dataset_store: wrote jsonl `{path}`')
        return path


@register_storage
class BinStorage(DatasetStorage):
    """The flat ``.bin``/``.idx`` memory-mapped pretraining layout -- reserved, not yet implemented.

    ``--store_format bin`` is accepted by the config vocabulary but refused here until the pretraining
    path lands, so a user who selects it gets a clear "later phase" rather than a silent no-op or a
    half-written store. Both directions raise: there is nothing to read back either.
    """

    name = 'bin'

    def read(self, path: str, *, split: str = 'train', **kwargs) -> Any:
        raise NotImplementedError("store_format='bin' (the flat .bin/.idx mmap pretraining layout) is a later "
                                  "phase; use 'arrow' for a directly-trainable store or 'jsonl' for a portable dump.")

    def write(self, data: DataType, path: str, *, fields: Optional[Sequence[str]] = None) -> str:
        raise NotImplementedError("store_format='bin' (the flat .bin/.idx mmap pretraining layout) is a later "
                                  "phase; use 'arrow' for a directly-trainable store or 'jsonl' for a portable dump.")


@register_storage
class OrcStorage(DatasetStorage):
    """ORC, the one common columnar format `datasets` has no packaged builder for.

    Read through ``pyarrow.orc`` directly: a single file and a directory of ``*.orc`` shards both become
    one table, shards read in sorted order so the row order is stable across runs. Write is refused by the
    base class -- ORC is a lake source, not a store target.
    """

    name = 'orc'
    extensions = ('orc', )
    detect_directory = True

    def read(self, path: str, *, split: str = 'train', **kwargs) -> Any:
        import glob

        import pyarrow as pa
        import pyarrow.orc
        from datasets import Dataset as HfDataset

        files = sorted(glob.glob(os.path.join(path, '*.orc'))) if os.path.isdir(path) else [path]
        if not files:
            raise FileNotFoundError(f'No .orc files found under {path}')
        tables = [pa.orc.read_table(file) for file in files]
        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
        dataset = HfDataset(table)
        # ``split`` is accepted for signature parity but an ORC table carries no splits, so the rows are
        # returned whole.
        return dataset.to_iterable_dataset() if kwargs.get('streaming', False) else dataset


@register_storage
class LanceStorage(DatasetStorage):
    """Lance, read through `datasets`' own lance builder -- which is reached only by name.

    The builder exists but the folder scanner never picks it for a ``.lance`` directory, so it is called
    explicitly: a single file via ``data_files``, a directory via ``data_dir``. It needs ``pylance``; a
    missing install is reported as such rather than surfacing later as an opaque builder error. Write is
    refused by the base class.
    """

    name = 'lance'
    extensions = ('lance', )
    detect_directory = True

    def read(self, path: str, *, split: str = 'train', **kwargs) -> Any:
        from datasets import load_dataset as hf_load_dataset
        try:
            import lance  # noqa: F401
        except ImportError as exc:
            raise ImportError('Loading a lance dataset needs the `pylance` package. '
                              'Install it with `pip install pylance`.') from exc
        if os.path.isdir(path):
            return hf_load_dataset('lance', data_dir=path, split=split, **kwargs)
        return hf_load_dataset('lance', data_files=path, split=split, **kwargs)


@register_storage
class DatasetsBuilderStorage(DatasetStorage):
    """The generic fallback: any single file whose extension `datasets` has a packaged builder for.

    Covers csv, parquet, text, a loose ``.arrow`` file, and so on -- the formats the folder scanner and the
    extension passthrough already handled, now behind the same registry so ``build_dataset`` has no
    per-format branch. The extension maps to the builder name (``jsonl``->``json``, ``txt``->``text``);
    ``csv`` needs ``na_filter=False`` so an empty cell stays an empty string instead of a float NaN that
    would reach the template where a string was meant. Write is refused by the base class.
    """

    name = 'builder'

    def read(self, path: str, *, split: str = 'train', **kwargs) -> Any:
        from datasets import load_dataset as hf_load_dataset

        extension = os.path.splitext(path)[1].lstrip('.') or 'json'
        extension = {'jsonl': 'json', 'txt': 'text'}.get(extension, extension)
        if extension == 'csv':
            kwargs['na_filter'] = False
        return hf_load_dataset(extension, data_files=path, split=split, **kwargs)
