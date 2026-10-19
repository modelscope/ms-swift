# Copyright (c) ModelScope Contributors. All rights reserved.
# Components bind config in __init__ and turn upstream artifacts into downstream ones in __call__.
from __future__ import annotations

from .artifacts import DatasetArtifact, ModelArtifact
from .base import RecipeComponent
from .cached_dataset import CacheDatasetComponent
from .infer import InferComponent
from .weight_transform import BaseWeightTransform, ConvertComponent, MergeComponent, QuantizeComponent

__all__ = [
    'DatasetArtifact', 'ModelArtifact', 'RecipeComponent', 'CacheDatasetComponent', 'InferComponent',
    'BaseWeightTransform', 'ConvertComponent', 'MergeComponent', 'QuantizeComponent'
]
