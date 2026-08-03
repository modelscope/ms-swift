# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from .npu_patch import NPUCastError, apply_all_patches, apply_mindspeed_patches, patch_mindspeed_fla_gdn_implementation

apply_all_patches()

__all__ = ['NPUCastError', 'apply_all_patches', 'apply_mindspeed_patches', 'patch_mindspeed_fla_gdn_implementation']
