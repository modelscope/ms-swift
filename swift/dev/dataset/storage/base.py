# Copyright (c) ModelScope Contributors. All rights reserved.
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

#: Columns a materialised row is built around: ``lengths`` is what every length-aware consumer reads
#: before the first step, ``input_ids`` is what makes a row directly trainable, ``labels`` the mask.
CORE_STORE_FIELDS = ('input_ids', 'labels', 'lengths')

#: The documented field vocabulary -- reference for diagnostics, NOT a closed set: an arbitrary
#: dataset column may also be persisted (validate_store_fields checks against the real columns).
KNOWN_STORE_FIELDS = frozenset(CORE_STORE_FIELDS) | frozenset({
    'attention_mask',
    # rollout-token payload (infer recorder) and semantic row fields (infer emit)
    'completion_mask', 'response_token_ids', 'response_loss_mask', 'rollout_logprobs',
    'messages', 'all_messages', 'completions', 'responses', 'response',
    'rejected_messages', 'rejected_response', 'rejected_rollout_tokens', 'rejected_reward_score',
    'rollout_tokens', 'multi_turn', 'truncated', 'rollout_infos', 'rejected_truncated',
    'rejected_rollout_infos', 'id', 'num_generations', 'scores', 'reward_score', 'ground_truth',
    # media columns kept for train-time vision recompute
    'images', 'videos', 'audios', 'objects',
})

#: A stored dataset is either an HF table (the exporter's output) or the plain row dicts a run emits.
DataType = Union['HfDataset', Sequence[dict]]


class DatasetStorage(ABC):
    """One storage format: how to read a path back into rows and, when the format allows, write them.

    A subclass sets ``name`` (the registry key ``--store_format`` and detection speak), ``extensions``,
    and the capability bits; ``read`` is mandatory, every write-side method defaults to raising so a
    read-only source refuses loudly instead of being quietly absent from the interface.
    """

    #: Registry key and ``--store_format`` value. Every concrete subclass sets it.
    name: str = ''
    #: File suffixes (no dot) identifying this format for a single file; empty for formats recognised
    #: by directory shape (the arrow ``save_to_disk`` store) and the catch-all builder fallback.
    extensions: Tuple[str, ...] = ()
    #: Whether this format can be a ``--store_format`` write target; only writable store backends set
    #: it, so validate_store_format refuses the rest at config time rather than after a whole run.
    writable: bool = False
    #: Whether the format can be appended to row by row while a run progresses. Only formats that
    #: implement ``append`` set it; they are the only family a run can flush mid-run or resume.
    incremental: bool = False
    #: Whether a *directory* holding this format's files is detected as the format. Set only by the
    #: columnar lake sources: a hub snapshot also holds .json/.csv files, which must not match.
    detect_directory: bool = False
    #: Suffix the store path carries relative to a base output name (see store_path): a file store
    #: uses its real extension; a directory store detected by shape (arrow) uses ''.
    store_suffix: str = ''

    @abstractmethod
    def read(self, path: str, *, split: str = 'train', **kwargs) -> Any:
        """Read ``path`` into an HF ``Dataset`` (or ``IterableDataset`` when ``streaming`` is passed)."""

    def write(self, data: DataType, path: str, *, fields: Optional[Sequence[str]] = None) -> str:
        """Serialise ``data`` to ``path`` in one go and return ``path``. Read-only formats refuse here."""
        raise NotImplementedError(
            f"storage format {self.name!r} is read-only: it can load an existing dataset but cannot be "
            f"written. Use --store_format arrow (a directly-trainable store) or jsonl (a portable dump).")

    def append(self, rows: Sequence[dict], path: str) -> None:
        """Append finished rows to a store a run is still writing. Only an incremental format may
        override this; the base refusal keeps the capability bit and the write action in one class."""
        raise NotImplementedError(
            f"storage format {self.name!r} is not incremental: a run can only write it once, at the "
            f'end, and cannot flush or resume mid-run. Use jsonl (or declare incremental=True and '
            f'implement append) for that.')

    def iter_rows(self, path: str, **kwargs):
        """Iterate the store's rows as plain dicts -- the row-stream read a cache-style consumer uses.

        The default walks ``read``, which needs a complete well-formed file. An incremental format
        overrides it with a tolerant line-wise parse, so a run's partial output is already readable.
        """
        return iter(self.read(path, **kwargs))


#: Format name -> storage class. Populated by ``@register_storage`` in :mod:`.formats`, which the
#: package ``__init__`` imports so the registry is complete before any read/write call.
STORAGE_MAPPING: Dict[str, Type[DatasetStorage]] = {}


def register_storage(storage_cls: Type[DatasetStorage] = None, *, exist_ok: bool = False):
    """Register a storage format, keyed by its ``name``. Usable bare or with keywords.

    Mirrors ``register_dataset``: everything is read off the class, so a format is declared in one
    place, and a duplicate name is rejected rather than silently shadowing the registered format.
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
    """The storage format ``path`` holds, or None when it is not one this layer reads directly.

    ``None`` means an ordinary directory (a downloaded hub snapshot) that the loader routes through
    the hub folder scanner instead. A directory is classified by shape: an arrow store is marked by
    ``state.json``; a ``detect_directory`` format matches the dir's own name or an entry's extension.
    A single file matches by extension, else 'builder'.
    """
    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, 'state.json')):
            return 'arrow'
        # Only formats that opt into directory detection compete here: a columnar lake source is a
        # folder of ``.<ext>`` shards (orc) or a named ``.<ext>`` dataset dir (lance).
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

    A file format appends its ``store_suffix``; a directory format detected by shape (arrow) uses the
    base name as-is, which keeps ``export --to_cached_dataset``'s ``train``/``val`` layout
    interchangeable with legacy's.
    """
    return base + STORAGE_MAPPING[store_format]().store_suffix


def validate_store_format(store_format: str, *, writable: bool = False) -> None:
    """Reject a ``--store_format`` that is not registered, or (when ``writable``) cannot be written.

    Called from the config validate phase -- AFTER ``--external_plugins`` is imported, so a user's
    ``@register_storage`` format already passes. Failing here rather than at the write is the point:
    a run would otherwise encode a whole dataset and only then discover the container cannot hold it.
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

    A requested name matching no column is rejected rather than ignored -- a misspelt field would
    otherwise silently drop a column. The registry only sharpens the message: a known field absent
    from this data means the encoding does not produce it, an unknown name is not a store field.
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
    # Dropping lengths is legal (a tokens-only store) but silently disables every length-aware
    # consumer, so say so when the data had it and the request did not keep it.
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

    ``data`` is an HF ``Dataset`` or a ``list[dict]``; ``fields`` (None = keep every column) is an
    allow-list validated against the data. A read-only or unimplemented backend raises from the
    storage class rather than being special-cased here, so dispatch stays a single registry lookup.
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

    The backend is detected from the path; a trailing ``#N`` row budget is honoured unless the path
    already exists on disk. The post-read steps (the 3.x ``length``->``lengths`` rename, the
    ``truncation_strategy='delete'`` filter, the ``#N`` sample) are specific to reading back an
    *encoded store*, so they live here rather than in the format's ``read``.
    """
    from ..loader import DatasetLoader

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

    A packed cache stores a list of the counts it packed -- take the longest so the filter is on the
    real sequence length. An empty list (a row that could not be encoded) reads as 0 and is kept.
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
