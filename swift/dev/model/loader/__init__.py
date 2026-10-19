# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import importlib
import pkgutil

from .base import (
    MODEL_ALIASES,
    MODEL_MAPPING,
    ModelArch,
    ModelInfo,
    ModelLoader,
    build_model_loader,
    get_model_loader,
    match_model_type,
    match_model_types_by_architectures,
    register_model,
    resolve_loader_cls,
    resolve_template,
)

# Import every family module for its @register_model side effects (mirroring the dataset loader
# package); the import is dynamic so a newly added family module needs no list edit here.
for _module in pkgutil.iter_modules(__path__):
    if not _module.name.startswith('_'):
        importlib.import_module(f'{__name__}.{_module.name}')
del _module

__all__ = [
    'MODEL_ALIASES', 'MODEL_MAPPING', 'ModelArch', 'ModelInfo', 'ModelLoader', 'build_model_loader',
    'get_model_loader', 'match_model_type', 'match_model_types_by_architectures', 'register_model',
    'resolve_loader_cls', 'resolve_template'
]
