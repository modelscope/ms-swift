"""export_cached_dataset: precompute the dataset once and write it to disk.

The write half of ``DatasetConfig.cached_dataset``. Preprocessing (dataset load, column mapping,
filtering and the `lengths` pass) is pure CPU work, so doing it inside every training job wastes GPU
time and repeats itself on every rank and every rerun. This recipe runs that chain ONCE, saves the
result with ``save_to_disk``, and later runs point ``DatasetConfig.cached_dataset`` at the output --
``build_dataset`` then loads it and skips encoding (see builders/dataset.py::_load_cached_datasets).

Peer of legacy ``swift export --to_cached_dataset``
(swift/pipelines/export/cached_dataset.py::ExportCachedDataset), with one deliberate difference:
legacy subclasses SwiftSft (the training entry) and neutralizes the model with a meta device, whereas
this recipe reuses the dev builders directly, so NO model is constructed at all -- only a processor is
loaded, for the tokenizer the template needs.

What lands on disk mirrors the eager training path exactly, because it calls the SAME
``swift.dev.builders.dataset._encode``: for the default ``truncation_strategy`` that is
``AddLengthPreprocessor``, which keeps the raw row and adds only a ``lengths`` column (tokenization
itself still happens per-batch at train time); for ``truncation_strategy='split'`` it is a full
``EncodePreprocessor``, since splitting changes sample boundaries and must be materialized.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from swift.dev.config import DatasetConfig, ModelConfig, TemplateConfig

logger = logging.getLogger(__name__)


def export_cached_dataset(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    *,
    output_dir: str = 'output',
) -> Tuple[str, Optional[str]]:
    """Encode ``dataset_config`` once and save it under ``output_dir``.

    Returns ``(train_dir, val_dir)``; ``val_dir`` is None when there is no val split. The layout
    ('train' / 'val' subdirectories) matches legacy's exporter so a cache produced by either side is
    interchangeable, and each subdirectory is a standalone ``load_from_disk`` target -- which is what
    ``DatasetConfig.cached_dataset`` / ``cached_val_dataset`` expect (they take the SUBDIRECTORY, not
    ``output_dir``).

    No model is loaded and no distributed init happens: this is a single-process CPU job. Run it on a
    CPU box, then reuse the output across experiments.
    """
    from swift.dev.builders import build_template
    from swift.dev.builders.dataset import _encode_mode
    from swift.model import get_model_processor

    if not dataset_config.dataset and not dataset_config.val_dataset:
        raise ValueError('export_cached_dataset needs DatasetConfig.dataset (or val_dataset) to encode. '
                         'cached_dataset is the OUTPUT of this step, not its input.')
    if dataset_config.streaming:
        raise ValueError('export_cached_dataset does not support streaming=True: an IterableDataset has no '
                         'save_to_disk. Set DatasetConfig.streaming=False.')

    # Only the processor (tokenizer) is needed -- the template encodes with it. load_model=False keeps
    # this a CPU-only job; legacy instead builds a meta-device model to satisfy SwiftSft's __init__.
    _, processor = get_model_processor(model_config.model, load_model=False)
    template = build_template(template_config, processor)

    train_raw, val_raw = _load_raw(dataset_config)

    # Same mode resolution as training, so the cache matches what an eager run would have built.
    # Packing is NOT applied here: it is a training-time layout (it depends on packing_length and is
    # cheap given `lengths`), and baking it in would freeze that choice into the cache.
    encode_mode = _encode_mode(dataset_config, template)
    if encode_mode == 'lazy':
        # lazy would return a LazyLLMDataset wrapper (no save_to_disk) and defeat the purpose:
        # nothing would be precomputed. Force the caller to opt into eager encoding explicitly.
        raise ValueError('export_cached_dataset requires eager encoding, but the resolved mode is lazy '
                         '(DatasetConfig.lazy_tokenize=True, or the multimodal default). Set '
                         'DatasetConfig.lazy_tokenize=False to precompute and save.')

    os.makedirs(output_dir, exist_ok=True)
    train_dir = _encode_and_save(train_raw, template, dataset_config, encode_mode, output_dir, 'train')
    val_dir = _encode_and_save(val_raw, template, dataset_config, encode_mode, output_dir, 'val')
    if train_dir is None:
        raise ValueError('export_cached_dataset produced no train split; check DatasetConfig.dataset.')
    return train_dir, val_dir


def _load_raw(dataset_config: DatasetConfig) -> tuple:
    """Load train (+val) exactly as build_dataset does (same kwargs, same split semantics)."""
    from swift.dev.builders.dataset import _load_kwargs
    from swift.dev.dataset import load_dataset

    load_kwargs = _load_kwargs(dataset_config)
    train_raw, val_raw = (None, None)
    if dataset_config.dataset:
        train_raw, val_raw = load_dataset(
            dataset_config.dataset,
            split_dataset_ratio=dataset_config.split_dataset_ratio,
            shuffle=dataset_config.dataset_shuffle,
            **load_kwargs)
    if dataset_config.val_dataset:
        _, val_raw = load_dataset(
            dataset_config.val_dataset,
            split_dataset_ratio=1.0,
            shuffle=dataset_config.val_dataset_shuffle,
            **load_kwargs)
    return train_raw, val_raw


def _encode_and_save(raw, template, dataset_config: DatasetConfig, encode_mode: str, output_dir: str,
                     name: str) -> Optional[str]:
    """Encode one split and save it to ``output_dir/name``; returns the path (None if no split)."""
    from swift.dev.builders.dataset import _encode

    if raw is None:
        return None
    enc = _encode(
        raw,
        template,
        mode=encode_mode,
        num_proc=dataset_config.dataset_num_proc,
        strict=dataset_config.strict,
        data_seed=dataset_config.data_seed)
    path = os.path.join(output_dir, name)
    enc.save_to_disk(path)
    logger.info(f'cached_dataset: `{path}` ({len(enc)} rows)')
    return path
