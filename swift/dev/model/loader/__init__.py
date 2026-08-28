# Copyright (c) ModelScope Contributors. All rights reserved.
"""Model loading: checkpoint/architecture resolution and the :class:`ModelLoader` base (:mod:`base`),
plus the built-in family registrations (every other module in this package).

Importing this package registers every built-in model family. Each family module calls
``@register_model`` at import time, so they are imported here purely for that side effect -- mirroring
how the dataset ``loader`` package wires its registrations. The import is dynamic (``pkgutil``) so a
newly added family module is picked up without editing a list here.
"""
import importlib
import pkgutil

from .base import (
    MODEL_ALIASES,
    MODEL_MAPPING,
    ModelArch,
    ModelInfo,
    ModelLoader,
    get_model_loader,
    match_model_type,
    match_model_types_by_architectures,
    register_model,
    resolve_template,
)

# Import every family module for its @register_model side effects. ``base`` is already imported above
# (and carries no registrations); everything else in the package is a family declaration.
for _module in pkgutil.iter_modules(__path__):
    if _module.name != 'base':
        importlib.import_module(f'{__name__}.{_module.name}')
del _module

__all__ = [
    'MODEL_ALIASES', 'MODEL_MAPPING', 'ModelArch', 'ModelInfo', 'ModelLoader', 'get_model_loader',
    'match_model_type', 'match_model_types_by_architectures', 'register_model', 'resolve_template'
]
