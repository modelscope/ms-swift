# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import os
import sys
from transformers.utils import strtobool

from .fsdp import NPUCastError
from .mindspeed import apply_mindspeed_patches, patch_mindspeed_fla_gdn_implementation

_APPLIED = False
_ENABLE_NPU_MODEL_PATCH_ARGS = ('--enable_npu_model_patch', '--enable-npu-model-patch')
#: Process-global switch exported by the v5 config layer (see swift.dev.config.process.process_configs)
#: from ``ModelConfig.enable_npu_model_patch``. It takes priority over argv so an explicit config value is
#: authoritative, and it reaches Ray workers through ``os.environ.copy()`` in create_workers -- which argv
#: cannot, since a worker's sys.argv is the raylet's, not the user's. Legacy never sets it and keeps argv.
_ENABLE_NPU_MODEL_PATCH_ENV = 'SWIFT_ENABLE_NPU_MODEL_PATCH'


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


def _is_model_patch_enabled() -> bool:
    """Resolve the model-patch switch: the v5 config env var wins, else the legacy argv flag, else True."""
    env_value = os.environ.get(_ENABLE_NPU_MODEL_PATCH_ENV)
    if env_value is not None:
        return _parse_model_patch_enabled(env_value)
    return _is_model_patch_enabled_from_argv()


def apply_all_patches() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from . import env, fsdp

    env.apply_patch()
    fsdp.apply_patch()
    # The model patch switch is checked only on the first import; monkey patches are not reversible.
    if _is_model_patch_enabled():
        from . import model
        model.apply_patch()
    _APPLIED = True


__all__ = ['NPUCastError', 'apply_all_patches', 'apply_mindspeed_patches', 'patch_mindspeed_fla_gdn_implementation']
