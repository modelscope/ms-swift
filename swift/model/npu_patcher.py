# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from .npu_patch import NPUCastError, apply_all_patches, patch_mindspeed_te_cp_implementation

apply_all_patches()

__all__ = ['NPUCastError', 'apply_all_patches', 'patch_mindspeed_te_cp_implementation']
