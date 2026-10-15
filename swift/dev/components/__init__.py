# Copyright (c) ModelScope Contributors. All rights reserved.
"""Components: the units a dev recipe is assembled from.

A component binds configuration in ``__init__`` and turns upstream artifacts into downstream ones in
``__call__``. The ``run_*`` recipes stay as the thin procedural entry points the CLI calls; they
construct the matching component and unwrap its artifact.

Layout: ``artifacts`` and ``base`` are the scaffolding every component uses; each category of
component then gets its own subpackage holding that category's base plus one module per member
(``weight_transform`` / ``cached_dataset`` today; more as they are migrated).
"""
from __future__ import annotations

from .artifacts import DatasetArtifact, ModelArtifact
from .base import RecipeComponent
from .cached_dataset import CacheDatasetComponent
from .weight_transform import BaseWeightTransform, ConvertComponent, MergeComponent, QuantizeComponent

__all__ = [
    'DatasetArtifact', 'ModelArtifact', 'RecipeComponent', 'CacheDatasetComponent', 'BaseWeightTransform',
    'ConvertComponent', 'MergeComponent', 'QuantizeComponent'
]
