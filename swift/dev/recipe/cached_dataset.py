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

What lands on disk is written through the shared store writer (``swift.dev.dataset.store``), the same
flow ``swift infer`` uses, so the container is configurable (``store_format``: arrow / jsonl) and the
persisted columns are an allow-list (``store_fields``). The encoding mirrors the eager training path,
because it calls the SAME ``swift.dev.builders.dataset._encode``: by default that is
``MeasurePreprocessor``, which keeps the raw row and adds only a ``lengths`` column (tokenization itself
still happens per-batch at train time); ``store_encoded=True`` materializes the tokens instead (a full
``EncodePreprocessor``, so rows carry ``input_ids``/``labels`` and train with no re-encoding), as does
``truncation_strategy='split'``, since splitting changes sample boundaries and must be materialized.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from swift.dev.config import DatasetConfig, ModelConfig, TemplateConfig

logger = logging.getLogger(__name__)


def export_cached_dataset(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    *,
    output_dir: str = 'output',
    template_mode: str = 'train',
    store_format: str = 'arrow',
    store_encoded: bool = False,
    store_fields: Optional[List[str]] = None,
) -> Tuple[str, Optional[str]]:
    """Encode ``dataset_config`` once and save it under ``output_dir``.

    Returns ``(train_dir, val_dir)``; ``val_dir`` is None when there is no val split. The layout
    ('train' / 'val' subdirectories) matches legacy's exporter so a cache produced by either side is
    interchangeable, and each subdirectory is a standalone ``load_from_disk`` target -- which is what
    ``DatasetConfig.cached_dataset`` / ``cached_val_dataset`` expect (they take the SUBDIRECTORY, not
    ``output_dir``).

    ``template_mode`` selects the encoding objective ('train' / 'rlhf' / 'kto'): the same rows encode
    differently per objective, so a cache built for one is wrong for the others. It mirrors legacy
    ``ExportCachedDataset`` calling ``template.set_mode(args.template_mode)``.

    No model is loaded and no distributed init happens: this is a single-process CPU job. Run it on a
    CPU box, then reuse the output across experiments.
    """
    from swift.dev.builders import build_template, load_model_processor
    from swift.dev.builders.dataset import _encode_mode

    if not dataset_config.dataset and not dataset_config.val_dataset:
        raise ValueError('export_cached_dataset needs DatasetConfig.dataset (or val_dataset) to encode. '
                         'cached_dataset is the OUTPUT of this step, not its input.')
    if dataset_config.streaming:
        raise ValueError('export_cached_dataset does not support streaming=True: an IterableDataset has no '
                         'save_to_disk. Set DatasetConfig.streaming=False.')

    # Only the processor (tokenizer) is needed -- the template encodes with it. load_model=False keeps
    # this a CPU-only job; legacy instead builds a meta-device model to satisfy SwiftSft's __init__.
    _, processor = load_model_processor(model_config)
    template = build_template(template_config, processor)
    # build_template fixes the mode to 'train' (the training default); override it here so an rlhf/kto
    # cache encodes with the right structure. Without this, --template_mode parsed but changed nothing.
    template.set_mode(template_mode)

    # store_encoded materialises TEXT tokens only; it does not persist the vision tensors (pixel_values /
    # image_grid_thw), which this phase treats as recomputable from `images` and leaves out of the store.
    # That is unsafe for a multimodal row, not because its input_ids cannot be encoded (template.encode
    # emits input_ids and the vision tensors together) but because the forward's `_not_encoded` guard is
    # all-or-nothing: seeing input_ids it skips the whole re-encode, vision included, so a token-only row
    # would reach the model as image placeholders with no pixel_values. Caching multimodal therefore means
    # storing tokens AND vision tensors together -- a later phase. Refuse the combo and point at the default
    # (raw row + lengths, re-encoded at train time) rather than write a row that trains broken.
    if store_encoded and template.model_meta.is_multimodal:
        raise ValueError('store_encoded=True materialises text tokens but not the vision tensors, and a multimodal '
                         'row needs both stored together: the forward\'s _not_encoded guard skips re-encoding once '
                         'input_ids is present, so a token-only row would reach the model without its pixel_values. '
                         'Multimodal materialisation is a later phase; leave store_encoded=False to keep the raw row '
                         '+ lengths and re-encode (tokens and vision together) at train time.')

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
    train_dir = _encode_and_save(
        train_raw, template, dataset_config, encode_mode, output_dir, 'train',
        store_format=store_format, store_encoded=store_encoded, store_fields=store_fields)
    val_dir = _encode_and_save(
        val_raw, template, dataset_config, encode_mode, output_dir, 'val',
        store_format=store_format, store_encoded=store_encoded, store_fields=store_fields)
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


def _encode_and_save(raw, template, dataset_config: DatasetConfig, encode_mode: str, output_dir: str, name: str, *,
                     store_format: str = 'arrow', store_encoded: bool = False,
                     store_fields: Optional[List[str]] = None) -> Optional[str]:
    """Encode one split and save it to ``output_dir/name``; returns the path (None if no split).

    The write goes through :func:`~swift.dev.dataset.store.write_dataset_store`, the one serialiser shared
    with ``swift infer``. ``arrow`` (the default) writes a ``save_to_disk`` directory; ``jsonl`` writes a
    single ``.jsonl`` file, so the returned path carries the extension a reader detects the backend from.
    """
    from swift.dev.builders.dataset import _encode
    from swift.dev.dataset import write_dataset_store

    if raw is None:
        return None
    enc = _encode(
        raw,
        template,
        mode=encode_mode,
        num_proc=dataset_config.dataset_num_proc,
        strict=dataset_config.strict,
        data_seed=dataset_config.data_seed,
        materialize=store_encoded)
    path = os.path.join(output_dir, name)
    if store_format == 'jsonl':
        path += '.jsonl'
    write_dataset_store(enc, path, backend=store_format, fields=store_fields)
    logger.info(f'cached_dataset: `{path}` ({len(enc)} rows)')
    return path
