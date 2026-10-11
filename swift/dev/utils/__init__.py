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
from .env import get_dist_setting, get_hf_endpoint, is_dist, is_master, use_hf_hub
from .hf_config import HfConfigFactory
from .hub import get_hub, safe_snapshot_download
from .logger import get_logger
from .torch_utils import get_n_params_grads, to_device
from .utils import (check_json_format, deep_getattr, deepspeed_zero_stage, get_env_args, json_parse_to_dict,
                    parse_args, resolve_deepspeed_config, split_list, split_str_parts_by)

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
