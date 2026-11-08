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

    # GRPO with vLLM colocate mode may preload the model onto NPU before
    # Accelerator.prepare() is called. In that case, casting fp32 on NPU
    # would temporarily duplicate the full model (bf16 + fp32), causing OOM.
    # We move the model back to CPU first to free NPU memory, then cast.
    try:
        if param.device.type == 'npu':
            import torch_npu
            module = module.cpu()
            torch_npu.npu.synchronize()
            torch_npu.npu.empty_cache()
        return module.to(torch.float32)
    except Exception as exc:
        raise NPUCastError(f'Failed to cast {module.__class__.__name__} to fp32.') from exc


_original_fsdp2_prepare_model = fsdp_utils.fsdp2_prepare_model


@wraps(_original_fsdp2_prepare_model)
def _patched_fsdp2_prepare_model(accelerator: Accelerator, model: torch.nn.Module) -> torch.nn.Module:
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
    global _APPLIED, _original_fsdp2_prepare_model
    if _APPLIED:
        return

    import accelerate.accelerator as accelerate_accelerator

    from ..fsdp import apply_patch as apply_fsdp_patch

    apply_fsdp_patch()
    _original_fsdp2_prepare_model = fsdp_utils.fsdp2_prepare_model

    # accelerate/accelerator.py imports fsdp2_prepare_model at module load time;
    # patching fsdp_utils alone does not affect Accelerator._prepare_fsdp2.
    fsdp_utils.fsdp2_prepare_model = _patched_fsdp2_prepare_model
    accelerate_accelerator.fsdp2_prepare_model = _patched_fsdp2_prepare_model
    Accelerator._prepare_fsdp2 = wrapped_prepare_fsdp2
    _APPLIED = True
