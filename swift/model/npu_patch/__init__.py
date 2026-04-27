# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from .fsdp import NPUCastError
from .mindspeed import patch_mindspeed_te_cp_implementation

_APPLIED = False


def apply_all_patches() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from . import env, fsdp, model

    env.apply_patch()
    fsdp.apply_patch()
    model.apply_patch()
    _APPLIED = True


__all__ = ['NPUCastError', 'apply_all_patches', 'patch_mindspeed_te_cp_implementation']
