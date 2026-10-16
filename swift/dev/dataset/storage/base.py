# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unified dataset storage: one format registry behind both the encoded store and the data lake.

Two concerns that used to live in separate files with separate dispatch share one mechanism here:

- the *store* (:func:`write_dataset_store` / :func:`load_dataset_store`) serialises encoded rows and
  reads them back -- ``arrow`` (a ``save_to_disk`` directory) and ``jsonl``, the two backends
  ``--store_format`` selects, plus the reserved ``bin`` pretraining layout;
- the *data lake* reads external columnar sources ``datasets`` cannot route on its own -- ``orc``
  (no packaged builder) and ``lance`` (a builder reached only by name, never by the folder scanner).

They are the read/write duals of the same "path <-> rows" idea, so both dispatch through one
:obj:`STORAGE_MAPPING` registry keyed by format name instead of two hardcoded if-chains (the store's
``if backend`` and the loader's ``if lake_format``). The interface is symmetric -- every format has
``read`` and ``write`` -- and a format that cannot be written (every lake source, plus ``bin``)
inherits the base :meth:`DatasetStorage.write`, which raises: an unsupported operation fails loudly
rather than being quietly absent from the interface.

Adding a format is one ``@register_storage`` subclass in :mod:`.formats`; nothing here changes.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from swift.dev.utils import get_logger

if TYPE_CHECKING:
    from datasets import Dataset as HfDataset

logger = get_logger()

__all__ = [
    'CORE_STORE_FIELDS', 'KNOWN_STORE_FIELDS', 'DatasetStorage', 'STORAGE_MAPPING', 'register_storage',
    'detect_storage_format', 'store_path', 'validate_store_fields', 'validate_store_format',
    'write_dataset_store', 'load_dataset_store'
]

#: Columns a materialised row is built around. ``lengths`` is the anchor every length-aware consumer
#: reads before the first step -- packing plans its groups from it, length-grouped sampling sorts by it,
#: and ``truncation_strategy='delete'`` filters on it. ``input_ids`` is what makes a row directly
#: trainable (the forward's ``_not_encoded`` guard keys on it); ``labels`` carries the loss mask (-100 on
#: the positions that are not trained).
CORE_STORE_FIELDS = ('input_ids', 'labels', 'lengths')

#: The documented field vocabulary: the core three plus the optional payload the two commands produce --
#: the rollout-token sidecar fields infer's recorder writes, the semantic row fields its emit builds, and
#: the media columns a multimodal row keeps so training can recompute vision. This is a reference for
#: diagnostics and defaults, NOT a closed set: an arbitrary dataset column may also be persisted, so
#: :func:`validate_store_fields` checks a request against the columns the data actually has rather than
#: rejecting anything outside this registry.
KNOWN_STORE_FIELDS = frozenset(CORE_STORE_FIELDS) | frozenset({
    'attention_mask',
    # rollout-token payload (infer recorder)
    'completion_mask', 'response_token_ids', 'response_loss_mask', 'rollout_logprobs',
    # semantic row fields (infer emit)
    'messages', 'all_messages', 'completions', 'responses', 'response',
    'rejected_messages', 'rejected_response', 'rejected_rollout_tokens', 'rejected_reward_score',
    'rollout_tokens', 'multi_turn', 'truncated', 'rollout_infos', 'rejected_truncated',
    'rejected_rollout_infos', 'id', 'num_generations', 'scores', 'reward_score', 'ground_truth',
    # media columns kept for train-time vision recompute
    'images', 'videos', 'audios', 'objects',
})

#: A stored dataset is either an HF table (the exporter's output) or the plain row dicts infer accumulates.
DataType = Union['HfDataset', Sequence[dict]]


class DatasetStorage(ABC):
    """One storage format: how to read a path back into rows and, when the format allows it, write them.

    A concrete subclass sets :attr:`name` -- the registry key, and the value ``--store_format`` and
    :func:`detect_storage_format` both speak -- and :attr:`extensions`, the file suffixes that identify
    the format on disk. :meth:`read` is mandatory; :meth:`write` defaults to raising, so a read-only
    lake source stays read-only without each subclass repeating the refusal.
    """

    #: Registry key and ``--store_format`` value. Every concrete subclass sets it.
    name: str = ''
    #: File suffixes (without the dot) that identify this format for a *single file*. Empty for formats
    #: recognised by directory shape rather than extension (the arrow ``save_to_disk`` store) and for the
    #: catch-all builder fallback.
    extensions: Tuple[str, ...] = ()
    #: Whether this format can be a ``--store_format`` write target. Only the writable store backends set
    #: it; a read-only lake source, the generic builder, and the reserved ``bin`` leave it False so
    #: :func:`validate_store_format` refuses them at config time rather than after a whole encode/generate
    #: run has produced rows only :meth:`write` would then reject.
    writable: bool = False
    #: Whether the format can be appended to row by row while a run progresses -- only ``jsonl`` sets
    #: it, the one line-appendable format -- and therefore the only family the writers of
    #: :mod:`.writers` can checkpoint or resume; every other writable format is serialised in one
    #: :meth:`write` at finish, so a crash loses the run. Drives the writer choice in
    #: :func:`swift.dev.dataset.storage.writers.open_store_writer` and the resume guard in
    #: :func:`swift.dev.config.validate.validate_infer_config`.
    incremental: bool = False
    #: Whether a *directory* holding this format's files is detected as the format. Set only by the
    #: columnar lake sources whose data is a folder of ``.<ext>`` shards (``orc``) or a named ``.<ext>``
    #: dataset dir (``lance``). Deliberately False for the extension-mapped file formats (``jsonl``/
    #: ``builder``): a downloaded hub snapshot is an ordinary directory that also contains ``.json``/
    #: ``.csv`` files, so matching those here would misread a snapshot as a store -- it must fall through
    #: to the loader's hub folder scanner instead.
    detect_directory: bool = False
    #: Suffix the store path carries relative to a base output name (see :func:`store_path`). A file store
    #: uses its real extension (``jsonl`` -> ``.jsonl``); a directory store detected by shape rather than
    #: name (arrow's ``save_to_disk`` dir) uses ``''`` so the base name is the directory as-is.
    store_suffix: str = ''

    @abstractmethod
    def read(self, path: str, *, split: str = 'train', **kwargs) -> Any:
        """Read ``path`` into an HF ``Dataset`` (or ``IterableDataset`` when ``streaming`` is passed)."""

    def write(self, data: DataType, path: str, *, fields: Optional[Sequence[str]] = None) -> str:
        """Serialise ``data`` to ``path`` and return ``path``. Read-only formats refuse here.

        The refusal is the base implementation on purpose: a lake source (``orc``/``lance``) or a plain
        ``datasets`` builder format is a *source*, never a store target, so it inherits this instead of
        spelling out the same ``NotImplementedError``. Only the writable store backends override it.
        """
        raise NotImplementedError(
            f"storage format {self.name!r} is read-only: it can load an existing dataset but cannot be "
            f"written. Use --store_format arrow (a directly-trainable store) or jsonl (a portable dump).")

    def iter_rows(self, path: str, **kwargs):
        """Iterate the store's rows as plain dicts -- the row-stream read a cache-style consumer uses.

        The default walks :meth:`read`, which needs the file to be complete and well-formed. An
        ``incremental`` (line-appendable) format overrides this with a line-wise parse that tolerates a
        producer still writing (or having crashed mid-write on) the file, so a run's partial output can
        already be read back row by row -- the read-side counterpart of incremental writes.
        """
        return iter(self.read(path, **kwargs))


#: Format name -> storage class. Populated by ``@register_storage`` in :mod:`.formats`, which the package
#: ``__init__`` imports so the registry is complete before any read/write call.
STORAGE_MAPPING: Dict[str, Type[DatasetStorage]] = {}


def register_storage(storage_cls: Type[DatasetStorage] = None, *, exist_ok: bool = False):
    """Register a storage format, keyed by its ``name``. Usable bare or with keywords.

    Mirrors :func:`~swift.dev.dataset.loader.register_dataset`: everything is read off the class, so a
    format is declared in one place, and a duplicate name is rejected rather than silently shadowing the
    format already registered under it.
    """

    def _register(cls: Type[DatasetStorage]) -> Type[DatasetStorage]:
        name = cls.name
        assert name, f'{cls.__name__} must set `name`.'
        if not exist_ok and name in STORAGE_MAPPING:
            raise ValueError(f'storage format `{name}` is already registered '
                             f'by {STORAGE_MAPPING[name].__name__}.')
        STORAGE_MAPPING[name] = cls
        return cls

    return _register if storage_cls is None else _register(storage_cls)


def detect_storage_format(path: str) -> Optional[str]:
    """The storage format ``path`` holds, or ``None`` when it is not one this layer reads directly.

    Unifies the three probes that used to be scattered: the store's ``os.path.isdir`` backend guess, the
    loader's ``resolve_lake_format``, and the single-file extension passthrough. ``None`` means an
    ordinary directory (a downloaded hub snapshot) that the loader routes through the hub folder scanner
    instead -- not a format this layer parses.

    A directory is classified by its *shape*, a single file by its extension:

    - an arrow store is a ``save_to_disk`` directory, marked by ``state.json``. That marker is specific
      to ``save_to_disk``; ``dataset_info.json`` is NOT used because a hub snapshot carries one too (the
      loader even hides it before folder-scanning), so keying on it would misread a download as a store.
    - a directory that opts into detection (see :attr:`~DatasetStorage.detect_directory`) is a columnar
      lake source: a dataset dir named ``something.lance`` (its contents are version/data subfolders, so
      the *name* identifies it) or a folder of ``.lance``/``.orc`` shards (the *entries* identify it). The
      extension-mapped file formats stay out of directory detection on purpose, so an ordinary hub
      snapshot -- which also holds ``.json``/``.csv`` files -- is not misread as a store.
    - a single file matches the format whose :attr:`~DatasetStorage.extensions` claim its suffix, else
      falls to the generic ``builder`` (any extension ``datasets`` has a packaged builder for).
    """
    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, 'state.json')):
            return 'arrow'
        # Only formats that opt into directory detection compete here: a columnar lake source is a folder
        # of ``.<ext>`` shards (orc) or a named ``.<ext>`` dataset dir (lance). Matching the directory NAME
        # extension covers ``something.lance``; matching an ENTRY extension covers a folder of shards.
        name_extension = os.path.splitext(path.rstrip(os.sep))[1].lstrip('.').lower()
        entry_extensions = {os.path.splitext(entry)[1].lstrip('.').lower() for entry in os.listdir(path)}
        for name, storage_cls in STORAGE_MAPPING.items():
            if not storage_cls.detect_directory:
                continue
            if name_extension in storage_cls.extensions or entry_extensions.intersection(storage_cls.extensions):
                return name
        return None
    extension = os.path.splitext(path)[1].lstrip('.').lower()
    if not extension:
        return None
    for name, storage_cls in STORAGE_MAPPING.items():
        if extension in storage_cls.extensions:
            return name
    return 'builder'


def store_path(base: str, store_format: str) -> str:
    """The path a store of ``store_format`` occupies, given a base output name with no format suffix.

    A file format appends its :attr:`~DatasetStorage.store_suffix` (``jsonl`` -> ``base.jsonl``); a
    directory format detected by shape (arrow) uses the base name as-is (``base`` -> the ``save_to_disk``
    dir), which is what keeps ``export --to_cached_dataset``'s ``train``/``val`` layout interchangeable
    with legacy's.
    """
    return base + STORAGE_MAPPING[store_format]().store_suffix


def validate_store_format(store_format: str, *, writable: bool = False) -> None:
    """Reject a ``--store_format`` that is not a registered format, or (when ``writable``) cannot be written.

    Called from the config validate phase -- AFTER ``--external_plugins`` has been imported, so a user's
    ``@register_storage`` format is already in :obj:`STORAGE_MAPPING` and passes. Failing here rather than
    at the write is the point: a run would otherwise encode a whole dataset (or generate for hours) and
    only then discover the chosen container cannot hold it.
    """
    if store_format not in STORAGE_MAPPING:
        raise ValueError(f'unknown store_format {store_format!r}; registered formats: {sorted(STORAGE_MAPPING)}. '
                         f'A custom format is added by a @register_storage subclass loaded via --external_plugins.')
    if writable and not STORAGE_MAPPING[store_format].writable:
        writable_names = sorted(name for name, cls in STORAGE_MAPPING.items() if cls.writable)
        raise ValueError(f'store_format {store_format!r} cannot be written (it is a read-only source or a '
                         f'reserved later phase); writable formats: {writable_names}.')


def validate_store_fields(requested: Sequence[str], available: Iterable[str]) -> List[str]:
    """Check a ``store_fields`` allow-list against the columns the data actually has; return the keep-list.

    A requested name that matches no column is rejected rather than ignored: a misspelt field would
    otherwise silently drop a column from the store, which is exactly the failure this guards against.
    The registry only sharpens the message -- a *known* field absent from this data means the command's
    encoding does not produce it (``rollout_logprobs`` on a plain cached dataset, say), while an
    *unknown* name is simply not a store field at all.
    """
    available = list(available)
    missing = [f for f in requested if f not in available]
    if missing:
        known_but_absent = [f for f in missing if f in KNOWN_STORE_FIELDS]
        unknown = [f for f in missing if f not in KNOWN_STORE_FIELDS]
        parts = []
        if known_but_absent:
            parts.append(f'known store field(s) {known_but_absent} are not produced by this encoding')
        if unknown:
            parts.append(f'unrecognised field(s) {unknown} (not a store field, and no such column)')
        raise ValueError(f'store_fields cannot be satisfied: {"; ".join(parts)}. '
                         f'Available columns: {sorted(available)}.')
    keep = list(dict.fromkeys(requested))
    # Dropping lengths is legal (a tokens-only store) but silently disables every length-aware consumer,
    # so say so when the data had it and the request did not keep it.
    if 'lengths' in available and 'lengths' not in keep:
        logger.warning("store_fields omits 'lengths' though the data has it; packing, group_by_length and "
                       "truncation_strategy='delete' all read lengths, so a store without it cannot drive them.")
    return keep


def restrict_columns(data: Any, fields: Optional[Sequence[str]]) -> Any:
    """Apply the ``fields`` allow-list to an HF Dataset, returning it unchanged when ``fields`` is None."""
    if fields is None:
        return data
    keep = validate_store_fields(fields, data.column_names)
    drop = [c for c in data.column_names if c not in keep]
    return data.remove_columns(drop) if drop else data


def write_dataset_store(data: DataType, path: str, *, backend: str = 'arrow',
                        fields: Optional[Sequence[str]] = None) -> str:
    """Serialise encoded rows to ``path`` through the one shared registry; return ``path``.

    ``data`` is an HF ``Dataset`` or a ``list[dict]``. ``backend`` names the storage format (see the
    module docstring); ``fields`` (None = keep every column) is an allow-list validated against the data.
    A read-only or unimplemented backend raises from the storage class rather than being special-cased
    here, so the dispatch stays a single registry lookup.
    """
    try:
        storage_cls = STORAGE_MAPPING[backend]
    except KeyError:
        writable_names = sorted(name for name, cls in STORAGE_MAPPING.items() if cls.writable)
        raise ValueError(f'unknown store backend {backend!r}; expected one of {sorted(STORAGE_MAPPING)} '
                         f'(writable: {writable_names}).') from None
    return storage_cls().write(data, path, fields=fields)


def load_dataset_store(path: str,
                       *,
                       max_length: Optional[int] = None,
                       truncation_strategy: Optional[str] = None,
                       data_seed: Optional[int] = None,
                       shuffle: bool = False) -> Any:
    """Read one stored dataset path back into an HF ``Dataset``, whichever backend wrote it.

    The backend is detected from the path by :func:`detect_storage_format` (an arrow store is a
    ``save_to_disk`` directory, a jsonl store is a file). A trailing ``#N`` row budget is honoured unless
    the path already exists on disk (an existing name is read as-is, matching the ``dataset#N`` syntax
    used everywhere else). Under ``truncation_strategy='delete'`` rows longer than ``max_length`` are
    dropped before the subsample.

    These post-read steps (the 3.x ``length``->``lengths`` rename, the delete filter, the ``#N`` sample)
    are specific to reading back an *encoded store*, so they live here rather than in the format's
    ``read``: a data-lake read of the same format returns the raw rows and applies its own row budget
    later, in the loader's ``post_process``.

    This is the per-path primitive; :meth:`DatasetLoader.load_cached_datasets` loops it over the train and
    val path lists and keeps the ``(train, val)`` contract the builder expects.
    """
    # Deferred: the loader package imports this module at call time, so importing DatasetLoader at module
    # load would be circular. By the time a store is read, both are fully initialised.
    from swift.dev.dataset.loader import DatasetLoader

    if os.path.exists(path):
        sample_count = None
    else:
        path, sample_count = DatasetLoader.split_sample_count(path)
    storage_format = detect_storage_format(path)
    if storage_format is None:
        raise ValueError(f'Cannot read a stored dataset from `{path}`: not a recognised storage format. '
                         f"An encoded store is an arrow `save_to_disk` directory (has `state.json`) or a "
                         f'`.jsonl` file; a plain directory of data files is loaded as a dataset, not a '
                         f'store, and should be passed to --dataset rather than --cached_dataset.')
    dataset = STORAGE_MAPPING[storage_format]().read(path)
    # ms-swift 3.x wrote the encoded token count as ``length``; dev reads ``lengths``.
    if 'length' in dataset.column_names and 'lengths' not in dataset.column_names:
        dataset = dataset.rename_column('length', 'lengths')
    if truncation_strategy == 'delete' and max_length is not None:
        dataset = filter_over_length(dataset, max_length)
    if sample_count is not None:
        dataset = DatasetLoader.sample_dataset(dataset, sample_count, shuffle, data_seed)
    return dataset


def filter_over_length(dataset: Any, max_length: int) -> Any:
    """Drop rows whose token count exceeds ``max_length`` (the ``truncation_strategy='delete'`` filter).

    ``lengths`` is a per-row token count, but a packed cache stores a list of the counts it packed -- take
    the longest so the filter is on the real sequence length. A row that could not be encoded carries an
    empty list and is treated as length 0 (kept, then substituted at access time).
    """
    if 'lengths' not in dataset.column_names:
        raise ValueError("truncation_strategy='delete' filters on a 'lengths' column, which only EAGER encoding "
                         'produces (Measure/EncodePreprocessor). Load with lazy_tokenize=False on a map-style '
                         "dataset, or drop truncation_strategy='delete'.")
    lengths = dataset['lengths']
    if lengths and isinstance(lengths[0], list):
        arr = np.fromiter((max(x) if x else 0 for x in lengths), dtype=np.int64, count=len(lengths))
    else:
        arr = np.asarray(lengths, dtype=np.int64)
    keep = arr <= max_length
    if not bool(keep.all()):
        dataset = dataset.select(np.flatnonzero(keep))
    return dataset
