# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import sys

from transformers.utils import strtobool

from .fsdp import NPUCastError
from .mindspeed import patch_mindspeed_te_cp_implementation

_APPLIED = False
_ENABLE_NPU_MODEL_PATCH_ARGS = ('--enable_npu_model_patch', '--enable-npu-model-patch')


def _parse_model_patch_enabled(value: str) -> bool:
    try:
        return bool(strtobool(value))
    except ValueError as exc:
        raise ValueError('--enable_npu_model_patch must be true or false.') from exc


def _is_model_patch_enabled_from_argv() -> bool:
    for i, arg in enumerate(sys.argv):
        if arg in _ENABLE_NPU_MODEL_PATCH_ARGS:
            if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith('--'):
                raise ValueError('--enable_npu_model_patch requires a value: true or false.')
            return _parse_model_patch_enabled(sys.argv[i + 1])
        if any(arg.startswith(f'{name}=') for name in _ENABLE_NPU_MODEL_PATCH_ARGS):
            value = arg.split('=', 1)[1]
            return _parse_model_patch_enabled(value)
    return True


def apply_all_patches() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from . import env, fsdp

    env.apply_patch()
    fsdp.apply_patch()
    # The model patch switch is checked only on the first import; monkey patches are not reversible.
    if _is_model_patch_enabled_from_argv():
        from . import model
        model.apply_patch()
    _APPLIED = True


__all__ = ['NPUCastError', 'apply_all_patches', 'patch_mindspeed_te_cp_implementation']
