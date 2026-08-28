# Copyright (c) ModelScope Contributors. All rights reserved.
"""dev's self-contained utility layer.

Copies of the simple, self-contained helpers dev used to import from ``swift.utils`` so the dev
stack no longer couples to legacy for them. The relatively complex utils (``parse_args``,
``safe_snapshot_download``, ``HfConfigFactory``, ``get_n_params_grads``, ``get_hub``) are
deliberately NOT copied here -- dev keeps importing those from ``swift.utils`` / ``swift.hub``.
"""
from .env import get_dist_setting, get_hf_endpoint, is_dist, is_master, use_hf_hub
from .logger import get_logger
from .torch_utils import to_device
from .utils import check_json_format, get_env_args, json_parse_to_dict, split_list

__all__ = [
    'get_logger',
    'get_dist_setting',
    'get_hf_endpoint',
    'is_dist',
    'is_master',
    'use_hf_hub',
    'to_device',
    'check_json_format',
    'get_env_args',
    'json_parse_to_dict',
    'split_list',
]
