# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import os

from swift.utils.logger import get_logger

logger = get_logger()

_DEFAULT_NPU_HCCL_CONNECT_TIMEOUT = '600'
_TORCH_NPU_GETENV_MODULE = 'torch_npu.utils.patch_getenv'


def _patch_torch_npu_getenv() -> None:
    try:
        from torch_npu.utils import patch_getenv
    except Exception:
        return

    orig_environ_get = getattr(patch_getenv, '_orig_environ_get', None)
    current_get = os.environ.get
    current_getenv = os.getenv
    getenv_module = getattr(current_getenv, '__module__', None)
    environ_get_module = getattr(current_get, '__module__', None)
    if not (getenv_module == _TORCH_NPU_GETENV_MODULE or environ_get_module == _TORCH_NPU_GETENV_MODULE):
        return
    if getattr(orig_environ_get, '__self__', None) is None:
        return

    log_once = getattr(patch_getenv, '_log_once', None)

    def _get_from_current_environ(key, default=None):
        hit = key in os.environ
        value = os.environ[key] if hit else default
        if hit and isinstance(value, str) and value != '' and log_once is not None:
            log_once(key, value)
        return value

    os.getenv = _get_from_current_environ
    os.environ.get = _get_from_current_environ
    logger.info('Patched torch_npu getenv to read from current os.environ.')


def apply_patch() -> None:
    _patch_torch_npu_getenv()

    if 'HCCL_CONNECT_TIMEOUT' in os.environ:
        return

    os.environ['HCCL_CONNECT_TIMEOUT'] = _DEFAULT_NPU_HCCL_CONNECT_TIMEOUT
    logger.info(f'Set HCCL_CONNECT_TIMEOUT={_DEFAULT_NPU_HCCL_CONNECT_TIMEOUT} by default for NPU.')
