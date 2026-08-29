# Copyright (c) ModelScope Contributors. All rights reserved.
"""dev's self-contained utility layer.

Copies of the helpers dev used to import from ``swift.utils`` / ``swift.hub``, so the dev stack no
longer couples to legacy for them. This now includes the previously-deferred complex utils --
``parse_args``, ``safe_snapshot_download`` + ``get_hub``, ``HfConfigFactory`` and
``get_n_params_grads`` -- all internalized here (hub access is trimmed to the download / load subset
dev uses and serialized via ``twinkle.utils.processing_lock``).
"""
from .env import get_dist_setting, get_hf_endpoint, is_dist, is_master, use_hf_hub
from .hf_config import HfConfigFactory
from .hub import get_hub, safe_snapshot_download
from .logger import get_logger
from .torch_utils import get_n_params_grads, to_device
from .utils import check_json_format, deep_getattr, get_env_args, json_parse_to_dict, parse_args, split_list

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
    'get_env_args',
    'json_parse_to_dict',
    'parse_args',
    'split_list',
    'HfConfigFactory',
    'get_hub',
    'safe_snapshot_download',
]
