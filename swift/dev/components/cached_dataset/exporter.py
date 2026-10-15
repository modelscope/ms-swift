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

    The layout ('train' / 'val' subdirectories) matches legacy's exporter so a cache produced by either
    side is interchangeable, and each subdirectory is a standalone ``load_from_disk`` target -- which is
    what ``DatasetConfig.cached_dataset`` / ``cached_val_dataset`` expect (they take the SUBDIRECTORY, not
    ``output_dir``). ``__call__`` returns a :class:`DatasetArtifact` naming both splits.

    ``template_mode`` selects the encoding objective ('train' / 'rlhf' / 'kto'): the same rows encode
    differently per objective, so a cache built for one is wrong for the others. It mirrors legacy
    ``ExportCachedDataset`` calling ``template.set_mode(args.template_mode)``.

    No model is loaded and no distributed init happens: this is a single-process CPU job. Run it on a
    CPU box, then reuse the output across experiments.
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

        # Same mode resolution as training, so the cache matches what an eager run would have built.
        # Packing is NOT applied here: it is a training-time layout (it depends on packing_length and is
        # cheap given `lengths`), and baking it in would freeze that choice into the cache.
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

        # Only the processor (tokenizer) is needed -- the template encodes with it. load_model=False keeps
        # this a CPU-only job; legacy instead builds a meta-device model to satisfy SwiftSft's __init__.
        _, processor = load_model_processor(self.model_config)
        template = build_template(self.template_config, processor)
        # build_template fixes the mode to 'train' (the training default); override it here so an rlhf/kto
        # cache encodes with the right structure. Without this, --template_mode parsed but changed nothing.
        template.set_mode(self.template_mode)
        return template

    def _guard_multimodal(self, template) -> None:
        # store_encoded materialises the FULL template.encode output -- for multimodal rows that includes
        # both input_ids AND vision tensors (pixel_values / image_grid_thw / ...) together, so the
        # forward's _not_encoded guard correctly skips re-encoding with all modalities present.
        # The only unsafe case is partial materialisation (tokens without vision), which cannot happen
        # here because EncodePreprocessor stores every key template.encode produces.
        # store_fields: if the user explicitly restricts columns and drops a vision key, the row would
        # arrive incomplete -- warn rather than raise, since the default (None = keep all) is safe.
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
            # lazy would return a LazyLLMDataset wrapper (no save_to_disk) and defeat the purpose:
            # nothing would be precomputed. Force the caller to opt into eager encoding explicitly.
            raise ValueError('export_cached_dataset requires eager encoding, but the resolved mode is lazy '
                             '(DatasetConfig.lazy_tokenize=True, or the multimodal default). Set '
                             'DatasetConfig.lazy_tokenize=False to precompute and save.')
        return encode_mode

    def _encode_and_save(self, raw, template, encode_mode: str, name: str) -> Optional[str]:
        """Encode one split and save it to ``output_dir/name``; returns the path (None if no split).

        The write goes through :func:`~swift.dev.dataset.storage.write_dataset_store`, the one serialiser
        shared with ``swift infer``. The store path is derived by
        :func:`~swift.dev.dataset.storage.store_path` from the format's ``store_suffix``: ``arrow`` (the
        default) writes a ``save_to_disk`` directory with no suffix, so a cache produced here stays
        interchangeable with legacy's exporter; ``jsonl`` appends ``.jsonl``, giving a single file a reader
        detects the backend from. A custom writable format contributes its own suffix.
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
            materialize=self.store_encoded)
        path = store_path(os.path.join(self.output_dir, name), self.store_format)
        write_dataset_store(enc, path, backend=self.store_format, fields=self.store_fields)
        logger.info(f'cached_dataset: `{path}` ({len(enc)} rows)')
        return path
