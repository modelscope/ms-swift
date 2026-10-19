# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

from swift.dev.utils.logger import get_logger

from ..artifacts import DatasetArtifact
from ..base import RecipeComponent

if TYPE_CHECKING:
    from swift.dev.config import DatasetConfig, ModelConfig, TemplateConfig

logger = get_logger()


class CacheDatasetComponent(RecipeComponent):
    """Encode ``dataset_config`` once and save it under ``output_dir`` as a reusable cache.

    The 'train' / 'val' subdirectories match legacy's exporter (interchangeable caches) and are
    standalone ``load_from_disk`` targets -- what ``cached_dataset`` / ``cached_val_dataset``
    expect. A CPU-only job; ``template_mode`` picks the encoding objective ('train' / 'rlhf' /
    'kto') -- the same rows encode differently per objective, so a cache serves one only.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        template_config: TemplateConfig,
        dataset_config: DatasetConfig,
        *,
        output_dir: str = 'output',
        template_mode: str = 'train',
        store_format: str = 'arrow',
        store_encoded: bool = False,
        store_fields: Optional[List[str]] = None,
    ):
        self.model_config = model_config
        self.template_config = template_config
        self.dataset_config = dataset_config
        self.output_dir = output_dir
        self.template_mode = template_mode
        self.store_format = store_format
        self.store_encoded = store_encoded
        self.store_fields = store_fields

    def __call__(self) -> DatasetArtifact:
        self._validate_config()
        template = self._build_template()
        self._guard_multimodal(template)
        train_raw, val_raw = self._load_raw()

        # Same mode resolution as training, so the cache matches what an eager run builds. Packing
        # is a training-time layout (cheap given `lengths`); baking it in would freeze the choice.
        encode_mode = self._resolve_encode_mode(template)

        os.makedirs(self.output_dir, exist_ok=True)
        train_dir = self._encode_and_save(train_raw, template, encode_mode, 'train')
        val_dir = self._encode_and_save(val_raw, template, encode_mode, 'val')
        if train_dir is None:
            raise ValueError('export_cached_dataset produced no train split; check DatasetConfig.dataset.')
        return DatasetArtifact(train_dir, val_dir)

    def _validate_config(self) -> None:
        """Refuse an input that cannot be encoded, before the processor is even loaded."""
        dataset_config = self.dataset_config
        if not dataset_config.dataset and not dataset_config.val_dataset:
            raise ValueError('export_cached_dataset needs DatasetConfig.dataset (or val_dataset) to encode. '
                             'cached_dataset is the OUTPUT of this step, not its input.')
        if dataset_config.streaming:
            raise ValueError('export_cached_dataset does not support streaming=True: an IterableDataset has no '
                             'save_to_disk. Set DatasetConfig.streaming=False.')

    def _build_template(self):
        """Load the processor (tokenizer only) and build the template in the requested mode."""
        from swift.dev.builders import build_template, load_model_processor

        # Only the processor (tokenizer) is needed; load_model=False keeps this a CPU-only job
        # (legacy instead builds a meta-device model to satisfy SwiftSft's __init__).
        _, processor = load_model_processor(self.model_config)
        template = build_template(self.template_config, processor)
        # build_template fixes the mode to 'train'; override it so an rlhf/kto cache encodes with
        # the right structure. Without this, --template_mode parsed but changed nothing.
        template.set_mode(self.template_mode)
        return template

    def _guard_multimodal(self, template) -> None:
        # store_encoded materialises the full encode output (input_ids AND vision tensors), so
        # re-encoding is correctly skipped; only store_fields omitting vision columns is unsafe.
        if self.store_encoded and template.model_meta.is_multimodal and self.store_fields:
            logger.warning(
                'store_fields is set on a multimodal store_encoded run: if the allow-list omits vision '
                'columns (pixel_values, image_grid_thw, ...), the cached rows will reach the model without '
                'their vision tensors. Leave store_fields=None (keep all columns) for a safe cache.')

    def _load_raw(self) -> tuple:
        """Load train (+val) exactly as build_dataset does (same kwargs, same split semantics)."""
        from swift.dev.builders.dataset import _load_kwargs
        from swift.dev.dataset import load_dataset

        dataset_config = self.dataset_config
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

    def _resolve_encode_mode(self, template) -> str:
        """Resolve the encode mode as training would, refusing the lazy mode that cannot be saved."""
        from swift.dev.builders.dataset import _encode_mode

        encode_mode = _encode_mode(self.dataset_config, template)
        if encode_mode == 'lazy':
            # lazy would return a LazyLLMDataset wrapper (no save_to_disk) and precompute nothing;
            # force the caller to opt into eager encoding explicitly.
            raise ValueError('export_cached_dataset requires eager encoding, but the resolved mode is lazy '
                             '(DatasetConfig.lazy_tokenize=True, or the multimodal default). Set '
                             'DatasetConfig.lazy_tokenize=False to precompute and save.')
        return encode_mode

    def _encode_and_save(self, raw, template, encode_mode: str, name: str) -> Optional[str]:
        """Encode one split and save it to ``output_dir/name``; returns the path (None if no split).

        The write goes through ``write_dataset_store``, the one serialiser shared with ``swift
        infer``; the path comes from ``store_path`` and the format's ``store_suffix``: ``arrow``
        (default) writes a suffix-less ``save_to_disk`` directory, interchangeable with legacy's
        exporter; ``jsonl`` appends ``.jsonl``; a custom writable format brings its own suffix.
        """
        from swift.dev.builders.dataset import _encode
        from swift.dev.dataset import write_dataset_store
        from swift.dev.dataset.storage import store_path

        if raw is None:
            return None
        enc = _encode(
            raw,
            template,
            mode=encode_mode,
            num_proc=self.dataset_config.dataset_num_proc,
            strict=self.dataset_config.strict,
            materialize=self.store_encoded,
            data_seed=self.dataset_config.data_seed)
        path = store_path(os.path.join(self.output_dir, name), self.store_format)
        write_dataset_store(enc, path, backend=self.store_format, fields=self.store_fields)
        logger.info(f'cached_dataset: `{path}` ({len(enc)} rows)')
        return path
