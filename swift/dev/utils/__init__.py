# Copyright (c) ModelScope Contributors. All rights reserved.
"""dev's self-contained utility layer.

Two kinds of module live here. The first are copies of the helpers dev used to import from
``swift.utils`` / ``swift.hub``, so the dev stack no longer couples to legacy for them. This now
includes the previously-deferred complex utils -- ``parse_args``, ``safe_snapshot_download`` +
``get_hub``, ``HfConfigFactory`` and ``get_n_params_grads`` -- all internalized here (hub access is
trimmed to the download / load subset dev uses and serialized via ``twinkle.utils.processing_lock``).

The second are dev's own cross-cutting infrastructure modules, which belong to no single vertical and
so are imported submodule-qualified rather than re-exported below: :mod:`swift.dev.utils.naming`
(swift-style config name -> twinkle constructible), :mod:`swift.dev.utils.plugin` (the SwiftPlugin
registry / kind mechanism), :mod:`swift.dev.utils.tags` (the controlled model / dataset tag vocabulary)
and :mod:`swift.dev.utils.version` (dev's ``__version__``).
"""
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Static-analysis-only re-imports: hand IDEs / type checkers the real symbol bindings so
    # autocomplete, go-to-definition and ``from swift.dev.utils import X`` resolution all work,
    # without executing the heavy submodule imports at runtime. Runtime resolution stays lazy via
    # ``__getattr__`` below; this block never runs (``TYPE_CHECKING`` is False outside analyzers).
    from .env import get_dist_setting, get_hf_endpoint, is_dist, is_master, use_hf_hub
    from .hf_config import HfConfigFactory
    from .hub import get_hub, safe_snapshot_download
    from .logger import get_logger
    from .torch_utils import get_n_params_grads, to_device
    from .utils import (check_json_format, deep_getattr, deepspeed_zero_stage, get_env_args,
                        json_parse_to_dict, parse_args, resolve_deepspeed_config, split_list,
                        split_str_parts_by)

# Public name -> the submodule that defines it. Imports are deferred (PEP 562 ``__getattr__``) so
# that importing this package -- e.g. via the stdlib-only leaf ``swift.dev.utils.logger`` -- never
# eager-loads the torch / hub / transformers-heavy submodules (``torch_utils`` / ``hub`` /
# ``hf_config`` / ``utils``). ``swift.dev.config`` and ``swift.dev.rollout`` rely on this to keep
# their own import torch-free; ``get_logger`` stays a light leaf for the same reason (see logger.py,
# whose lazy ``env`` read is also stdlib-only).
_LAZY_ATTRS = {
    'get_dist_setting': 'env',
    'get_hf_endpoint': 'env',
    'is_dist': 'env',
    'is_master': 'env',
    'use_hf_hub': 'env',
    'get_logger': 'logger',
    'HfConfigFactory': 'hf_config',
    'get_hub': 'hub',
    'safe_snapshot_download': 'hub',
    'get_n_params_grads': 'torch_utils',
    'to_device': 'torch_utils',
    'check_json_format': 'utils',
    'deep_getattr': 'utils',
    'deepspeed_zero_stage': 'utils',
    'get_env_args': 'utils',
    'json_parse_to_dict': 'utils',
    'parse_args': 'utils',
    'resolve_deepspeed_config': 'utils',
    'split_list': 'utils',
    'split_str_parts_by': 'utils',
}

__all__ = [
    'get_logger',
    'get_dist_setting',
    'get_hf_endpoint',
    'is_dist',
    'is_master',
    'use_hf_hub',
    'to_device',
    'get_n_params_grads',
    'check_json_format',
    'deep_getattr',
    'deepspeed_zero_stage',
    'resolve_deepspeed_config',
    'get_env_args',
    'json_parse_to_dict',
    'parse_args',
    'split_list',
    'split_str_parts_by',
    'HfConfigFactory',
    'get_hub',
    'safe_snapshot_download',
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(importlib.import_module(f'.{module_name}', __name__), name)
    globals()[name] = value  # cache: resolve once, subsequent lookups skip __getattr__
    return value


def __dir__():
    return sorted(__all__)
