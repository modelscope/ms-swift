# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import accelerate.utils.fsdp_utils as fsdp_utils
import torch
from accelerate.accelerator import Accelerator
from functools import wraps


class NPUCastError(RuntimeError):
    """Raised when fp32 casting fails during NPU FSDP2 preparation."""


def _get_first_parameter(module: torch.nn.Module) -> torch.nn.Parameter | None:
    for param in module.parameters(recurse=True):
        return param
    return None


def _needs_fp32_cast_for_npu(
    module: torch.nn.Module,
    accelerator: Accelerator,
) -> bool:
    if accelerator.device.type != 'npu':
        return False

    param = _get_first_parameter(module)
    if param is None:
        return False

    return param.is_floating_point() and param.dtype != torch.float32


def _cast_to_fp32(module: torch.nn.Module) -> torch.nn.Module:
    """
    Cast module parameters to fp32.

    Assumes parameters are already on CPU or meta device.
    Only dtype is changed; device is preserved.
    """
    try:
        return module.to(torch.float32)
    except Exception as exc:
        raise NPUCastError(f'Failed to cast {module.__class__.__name__} to fp32.') from exc


_original_fsdp2_prepare_model = fsdp_utils.fsdp2_prepare_model


@wraps(_original_fsdp2_prepare_model)
def wrapped_fsdp2_prepare_model(
    accelerator: Accelerator,
    model: torch.nn.Module,
):
    if _needs_fp32_cast_for_npu(model, accelerator):
        model = _cast_to_fp32(model)

    return _original_fsdp2_prepare_model(accelerator, model)


_original_prepare_fsdp2 = Accelerator._prepare_fsdp2


@wraps(_original_prepare_fsdp2)
def wrapped_prepare_fsdp2(
    self: Accelerator,
    *args,
    **kwargs,
):
    patched_args = [
        _cast_to_fp32(obj) if isinstance(obj, torch.nn.Module) and _needs_fp32_cast_for_npu(obj, self) else obj
        for obj in args
    ]

    return _original_prepare_fsdp2(self, *patched_args, **kwargs)


_APPLIED = False


def apply_patch() -> None:
    global _APPLIED
    if _APPLIED:
        return

    fsdp_utils.fsdp2_prepare_model = wrapped_fsdp2_prepare_model
    Accelerator._prepare_fsdp2 = wrapped_prepare_fsdp2
    _APPLIED = True
