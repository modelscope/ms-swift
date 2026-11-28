# Copyright (c) ModelScope Contributors. All rights reserved.
"""Direct NPU RMSNorm dispatch for frozen normalization weights."""

import inspect
import torch

_ORIGINAL_FORWARD = None
_TORCH_NPU = _FP8_MANAGER = _CPU_OFFLOAD = _DEBUG_STATE = _RMSNORM = None


def rmsnorm_forward(self, input, *extra_inputs, **kwargs):
    if (extra_inputs or kwargs or type(self) is not _RMSNORM or type(input) not in (torch.Tensor, torch.nn.Parameter)
            or input.device.type != 'npu' or type(self.weight) not in (torch.Tensor, torch.nn.Parameter)
            or self.weight.device != input.device or self.weight.requires_grad
            or input.dtype not in (torch.float16, torch.bfloat16, torch.float32) or torch.compiler.is_compiling()
            or torch.is_autocast_enabled('npu') or _FP8_MANAGER.is_fp8_enabled() or _FP8_MANAGER.is_fp8_calibration()
            or _CPU_OFFLOAD() or _DEBUG_STATE.debug_enabled):
        return _ORIGINAL_FORWARD(self, input, *extra_inputs, **kwargs)
    shape = tuple(self.weight.shape)
    if not input.numel() or input.ndim < len(shape) or tuple(input.shape[-len(shape):]) != shape:
        return _ORIGINAL_FORWARD(self, input, *extra_inputs, **kwargs)
    x = input.to(self.weight.dtype).contiguous().reshape(-1, self.weight.numel())
    weight = self.weight.reshape(-1)
    if self.zero_centered_gamma:
        weight = weight.float() + 1.0
    return _TORCH_NPU.npu_rms_norm(x, weight, epsilon=self.eps)[0].reshape(input.shape)


def patch_megatron_norm():
    global _ORIGINAL_FORWARD, _TORCH_NPU, _FP8_MANAGER, _CPU_OFFLOAD, _DEBUG_STATE, _RMSNORM
    if _ORIGINAL_FORWARD is not None:
        return
    import torch_npu
    from transformer_engine.debug.pytorch.debug_state import TEDebugState
    from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    from transformer_engine.pytorch.module.rmsnorm import RMSNorm

    if tuple(inspect.signature(RMSNorm.forward).parameters) != tuple(inspect.signature(rmsnorm_forward).parameters):
        raise RuntimeError('Unsupported TENPU RMSNorm signature for the Swift NPU performance patch.')
    _TORCH_NPU, _FP8_MANAGER = torch_npu, FP8GlobalStateManager
    _CPU_OFFLOAD, _DEBUG_STATE = is_cpu_offload_enabled, TEDebugState
    _RMSNORM = RMSNorm
    _ORIGINAL_FORWARD = RMSNorm.forward
    RMSNorm.forward = rmsnorm_forward
