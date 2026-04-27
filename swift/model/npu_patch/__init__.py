# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import sys

from transformers.utils import strtobool

from .fsdp import NPUCastError
from .mindspeed import patch_mindspeed_te_cp_implementation

_APPLIED = False


def _is_model_patch_enabled() -> bool:
    for i, arg in enumerate(sys.argv):
        if arg == '--enable_npu_patcher':
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                try:
                    return bool(strtobool(sys.argv[i + 1]))
                except ValueError as exc:
                    raise ValueError('--enable_npu_patcher must be true or false.') from exc
            return True
        if arg.startswith('--enable_npu_patcher='):
            value = arg.split('=', 1)[1]
            try:
                return bool(strtobool(value))
            except ValueError as exc:
                raise ValueError('--enable_npu_patcher must be true or false.') from exc
    return True


def apply_all_patches() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from . import env, fsdp

    env.apply_patch()
    fsdp.apply_patch()
    if _is_model_patch_enabled():
        from . import model
        model.apply_patch()
    _APPLIED = True


__all__ = ['NPUCastError', 'apply_all_patches', 'patch_mindspeed_te_cp_implementation']
