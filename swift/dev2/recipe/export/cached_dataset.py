# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from swift.dev2.config import DatasetConfig, ModelConfig, TemplateConfig


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
    """Encode ``dataset_config`` once and save it under ``output_dir`` -> ``(train_dir, val_dir)``.

    ``val_dir`` is None without a val split; the 'train' / 'val' subdirectories are the standalone
    ``load_from_disk`` targets ``cached_dataset`` / ``cached_val_dataset`` take (the SUBDIRECTORY,
    not ``output_dir``), matching legacy's layout. ``template_mode`` is the encoding objective
    ('train' / 'rlhf' / 'kto'); a cache built for one is wrong for the others. CPU-only, no init.
    """
    from swift.dev2.components import CacheDatasetComponent

    artifact = CacheDatasetComponent(
        model_config,
        template_config,
        dataset_config,
        output_dir=output_dir,
        template_mode=template_mode,
        store_format=store_format,
        store_encoded=store_encoded,
        store_fields=store_fields)()
    return artifact.train_dir, artifact.val_dir
