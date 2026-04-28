# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import accelerate.utils.fsdp_utils as fsdp_utils
import torch
from accelerate.accelerator import Accelerator
from functools import wraps


class NPUCastError(RuntimeError):
    """Raised when fp32 casting fails during NPU FSDP2 preparation."""


def _cast_module_to_fp32_for_npu_if_needed(module: torch.nn.Module, accelerator: Accelerator) -> torch.nn.Module:
    if accelerator.device.type != 'npu':
        return module

    param = next(module.parameters(recurse=True), None)
    if param is None:
        return module

    if not param.is_floating_point() or param.dtype == torch.float32:
        return module

    # Accelerate FSDP2 flattens and shards parameters during prepare. On NPU,
    # entering that path with bf16/fp16 parameters can fail before mixed
    # precision policy has a chance to manage runtime compute dtype. Cast early
    # while parameters are still on CPU or meta, so only dtype changes here.
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
    # Public utility entry used by some code paths before Accelerator.prepare.
    model = _cast_module_to_fp32_for_npu_if_needed(model, accelerator)
    return _original_fsdp2_prepare_model(accelerator, model)


_original_prepare_fsdp2 = Accelerator._prepare_fsdp2


@wraps(_original_prepare_fsdp2)
def wrapped_prepare_fsdp2(
    self: Accelerator,
    *args,
    **kwargs,
):
    # Accelerator.prepare may receive one or more modules directly; patch this
    # private entry too so all FSDP2 NPU preparation paths get the same fp32 cast.
    patched_args = [
        _cast_module_to_fp32_for_npu_if_needed(obj, self) if isinstance(obj, torch.nn.Module) else obj for obj in args
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
