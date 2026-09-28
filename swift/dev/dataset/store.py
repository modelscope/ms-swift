# Copyright (c) ModelScope Contributors. All rights reserved.
"""One storage flow for precomputed dataset rows, shared by ``swift export --to_cached_dataset`` and ``swift infer``.

Two commands materialise dataset rows to disk: the cached-dataset exporter (preprocessing done once so
later training runs skip it) and the infer rollout dump (an offline RL corpus). They used to serialise
independently -- the exporter to an Arrow table, infer to jsonl rows plus a per-candidate NPZ sidecar --
so the same token payload had two schemas and no shared reader. This module is the single place that
defines what a stored dataset looks like and how it is written and read back:

- one field vocabulary (:data:`CORE_STORE_FIELDS` / :data:`KNOWN_STORE_FIELDS`) both commands validate
  against, so a misspelt ``store_fields`` entry fails loudly instead of silently dropping a column;
- one writer (:func:`write_dataset_store`) with a pluggable backend -- ``arrow`` (a single
  ``save_to_disk`` table, directly trainable) and ``jsonl`` (line-delimited rows, portable and
  append-friendly). ``bin`` (the flat mmap layout a pretraining corpus wants) is a named later phase and
  raises until it exists rather than silently falling back;
- one reader (:func:`load_dataset_store`) that reads either backend back into an HF dataset, applying the
  ``length``->``lengths`` rename, the ``truncation_strategy='delete'`` length filter and the ``#N`` row
  budget the cached-dataset loader has always applied.

Why an Arrow store trains with zero re-encoding: the model forward's ``_not_encoded`` guard
(twinkle ``TransformersModel``) keys on ``input_ids`` and only calls ``template.batch_encode`` when it is
absent, so a materialised row is consumed as-is.

What is deliberately NOT stored here: the vision tensors of a multimodal row (large, and recomputable
from the ``images`` column the row keeps), the group-relative advantage of an offline RL corpus (cheap,
and the training step recomputes it from the stored rewards + logprobs), and any packing layout (a
training-time choice that reads ``lengths``). The store holds tokenisation, not layout.
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Union

import numpy as np

from swift.dev.utils import get_logger

if TYPE_CHECKING:
    from datasets import Dataset as HfDataset

logger = get_logger()

__all__ = [
    'CORE_STORE_FIELDS', 'KNOWN_STORE_FIELDS', 'validate_store_fields', 'write_dataset_store', 'load_dataset_store'
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


def write_dataset_store(data: DataType, path: str, *, backend: str = 'arrow',
                        fields: Optional[Sequence[str]] = None) -> str:
    """Serialise encoded rows to ``path`` through the one shared backend dispatch; return ``path``.

    ``data`` is an HF ``Dataset`` or a ``list[dict]``. ``backend`` picks the container (see the module
    docstring); ``fields`` (None = keep every column) is an allow-list validated against the data.
    """
    if backend == 'bin':
        raise NotImplementedError("store_format='bin' (the flat .bin/.idx mmap pretraining layout) is a later "
                                  "phase; use 'arrow' for a directly-trainable store or 'jsonl' for a portable dump.")
    if backend == 'arrow':
        _write_arrow(data, path, fields)
        return path
    if backend == 'jsonl':
        _write_jsonl(data, path, fields)
        return path
    raise ValueError(f"unknown store backend {backend!r}; expected 'arrow', 'jsonl' or 'bin'.")


def _restrict_columns(data: Any, fields: Optional[Sequence[str]]) -> Any:
    """Apply the ``fields`` allow-list to an HF Dataset, returning it unchanged when ``fields`` is None."""
    if fields is None:
        return data
    keep = validate_store_fields(fields, data.column_names)
    drop = [c for c in data.column_names if c not in keep]
    return data.remove_columns(drop) if drop else data


def _write_arrow(data: DataType, path: str, fields: Optional[Sequence[str]]) -> None:
    from datasets import Dataset as HfDataset

    dataset = data if isinstance(data, HfDataset) else HfDataset.from_list(list(data))
    dataset = _restrict_columns(dataset, fields)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    dataset.save_to_disk(path)
    logger.info(f'dataset_store: wrote arrow `{path}` ({len(dataset)} rows, columns={dataset.column_names})')


def _write_jsonl(data: DataType, path: str, fields: Optional[Sequence[str]]) -> None:
    from datasets import Dataset as HfDataset

    if isinstance(data, HfDataset):
        # Iterating an HF Dataset yields one row dict at a time, so a large store is not held in memory.
        rows: Iterable[dict] = _restrict_columns(data, fields)
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


def load_dataset_store(path: str,
                       *,
                       max_length: Optional[int] = None,
                       truncation_strategy: Optional[str] = None,
                       data_seed: Optional[int] = None,
                       shuffle: bool = False) -> Any:
    """Read one stored dataset path back into an HF ``Dataset``, whichever backend wrote it.

    The backend is detected from the path: an Arrow store is a directory (``save_to_disk``), a jsonl store
    is a file. A trailing ``#N`` row budget is honoured unless the path already exists on disk (an
    existing name is read as-is, matching the ``dataset#N`` syntax used everywhere else). Under
    ``truncation_strategy='delete'`` rows longer than ``max_length`` are dropped before the subsample.

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
    dataset = _read_store(path)
    # ms-swift 3.x wrote the encoded token count as ``length``; dev reads ``lengths``.
    if 'length' in dataset.column_names and 'lengths' not in dataset.column_names:
        dataset = dataset.rename_column('length', 'lengths')
    if truncation_strategy == 'delete' and max_length is not None:
        dataset = _filter_over_length(dataset, max_length)
    if sample_count is not None:
        dataset = DatasetLoader.sample_dataset(dataset, sample_count, shuffle, data_seed)
    return dataset


def _read_store(path: str) -> Any:
    from datasets import load_dataset as hf_load_dataset
    from datasets import load_from_disk

    if os.path.isdir(path):
        return load_from_disk(path)
    # A jsonl file: infer's portable dump, or a cached dataset written with store_format='jsonl'.
    return hf_load_dataset('json', data_files=path, split='train')


def _filter_over_length(dataset: Any, max_length: int) -> Any:
    """Drop rows whose token count exceeds ``max_length`` (the ``truncation_strategy='delete'`` filter).

    ``lengths`` is a per-row token count, but a packed cache stores a list of the counts it packed -- take
    the longest so the filter is on the real sequence length. A row that could not be encoded carries an
    empty list and is treated as length 0 (kept, then substituted at access time).
    """
    lengths = dataset['lengths']
    if lengths and isinstance(lengths[0], list):
        arr = np.fromiter((max(x) if x else 0 for x in lengths), dtype=np.int64, count=len(lengths))
    else:
        arr = np.asarray(lengths, dtype=np.int64)
    keep = arr <= max_length
    if not bool(keep.all()):
        dataset = dataset.select(np.flatnonzero(keep))
    return dataset
