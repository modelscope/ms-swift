# Copyright (c) ModelScope Contributors. All rights reserved.
"""The ``export_cached_dataset`` entry point for encoding a dataset into a reusable store.

``export_cached_dataset`` is the name the CLI (``swift export --to_cached_dataset``) calls; the encoding
and store-writing logic lives in :class:`swift.dev.components.CacheDatasetComponent`. This module keeps
the procedural signature and returns a plain ``(train_dir, val_dir)`` tuple, so the CLI is unaffected by
the component's typed artifact.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from swift.dev.config import DatasetConfig, ModelConfig, TemplateConfig


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
    from swift.dev.components import CacheDatasetComponent

    artifact = CacheDatasetComponent(
        model_config,
        template_config,
        dataset_config,
        output_dir=output_dir,
        template_mode=template_mode,
        store_format=store_format,
        store_encoded=store_encoded,
        store_fields=store_fields,
    )()
    return artifact.train_dir, artifact.val_dir
