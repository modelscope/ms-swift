# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.
"""NPU gradient norm and scaling optimizations."""

import importlib
import inspect
import torch
from typing import List, Optional, Tuple


def multi_tensor_scale(
    chunk_size: int,
    is_infinite: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    scale: float,
) -> None:
    """Scale tensors and flag non-finite inputs; chunk_size is ignored."""
    if len(tensor_lists) != 2:
        raise AssertionError(f"tensor_lists must have exactly 2 elements, got {len(tensor_lists)}")
    if len(tensor_lists[0]) != len(tensor_lists[1]):
        raise AssertionError('input and output lists must have the same number of tensors')

    with torch.no_grad():
        inputs, outputs = tensor_lists
        if inputs and _can_unscale_in_place(inputs, outputs):
            # Check overflow and scale gradients in one NPU call.
            found_inf = torch.zeros(1, device=inputs[0].device, dtype=torch.float32)
            factor = torch.as_tensor(scale, device=inputs[0].device, dtype=torch.float32)
            nonempty = [tensor for tensor in inputs if tensor.numel()]
            torch._amp_foreach_non_finite_check_and_unscale_(nonempty, found_inf, factor)
            is_infinite.copy_(
                torch.where(
                    found_inf.reshape(is_infinite.shape).ne(0),
                    torch.ones_like(is_infinite),
                    is_infinite,
                ))
            return
        finite_inputs = []
        for in_t, out_t in zip(tensor_lists[0], tensor_lists[1]):
            # Match NVTE: flag non-finite inputs, not scale-induced overflow.
            finite_inputs.append(torch.isfinite(in_t).all())
            out_t.copy_(in_t)
            out_t.mul_(scale)
        if finite_inputs:
            nonfinite = ~torch.stack(finite_inputs).all()
            is_infinite.copy_(torch.where(nonfinite, torch.ones_like(is_infinite), is_infinite))


def _can_unscale_in_place(inputs, outputs):
    """Check eligibility for fused in-place scaling."""
    device, dtype = inputs[0].device, inputs[0].dtype
    if device.type != 'npu' or dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    ranges = []
    for src, dst in zip(inputs, outputs):
        if src is not dst or src.device != device or src.dtype != dtype or not src.is_contiguous():
            return False
        if src.numel():
            start = src.data_ptr()
            ranges.append((start, start + src.numel() * src.element_size()))
    ranges.sort()
    return bool(ranges) and all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:]))


def multi_tensor_scale_tensor(
    chunk_size: int,
    is_infinite: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    scale: torch.Tensor,
) -> None:
    """Scale tensors using a scalar tensor."""
    multi_tensor_scale(chunk_size, is_infinite, tensor_lists, scale)


def multi_tensor_l2norm(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    per_tensor: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return the total L2 norm and optional per-tensor norms."""
    per_tensor = bool(per_tensor)

    norms: List[torch.Tensor] = []
    for grads_for_norm in tensor_lists:
        for grad in grads_for_norm:
            if grad.numel() == 0:
                norms.append(torch.tensor(0.0, dtype=torch.float32, device=grad.device))
            else:
                norm = grad.float().norm()
                norms.append(norm)

    if not norms:
        device = tensor_lists[0][0].device if tensor_lists and tensor_lists[0] else 'cpu'
        ret = torch.zeros(1, dtype=torch.float32, device=device)
        ret_per_tensor = (torch.stack(norms) if per_tensor else torch.empty(0, dtype=torch.float32, device=device))
        return ret, ret_per_tensor if per_tensor else None

    stacked = torch.stack(norms)
    # Reduce overflow flags on device.
    nonfinite = ~torch.isfinite(stacked).all()
    noop_flag.copy_(torch.where(nonfinite, torch.ones_like(noop_flag), noop_flag))
    total_norm = stacked.norm()

    if per_tensor:
        return total_norm, stacked
    return total_norm, None


def multi_tensor_unscale_l2norm(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    inv_scale: torch.Tensor,
    per_tensor: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute unscaled norms without modifying gradients."""
    per_tensor = bool(per_tensor)
    inv_scale_val = inv_scale.item()

    norms: List[torch.Tensor] = []
    for grads_for_norm in tensor_lists:
        for grad in grads_for_norm:
            if grad.numel() == 0:
                norms.append(torch.tensor(0.0, dtype=torch.float32, device=grad.device))
            else:
                norm = (grad.float() * inv_scale_val).norm()
                norms.append(norm)

    if not norms:
        device = tensor_lists[0][0].device if tensor_lists and tensor_lists[0] else 'cpu'
        ret = torch.zeros(1, dtype=torch.float32, device=device)
        ret_per_tensor = (torch.stack(norms) if per_tensor else torch.empty(0, dtype=torch.float32, device=device))
        return ret, ret_per_tensor if per_tensor else None

    stacked = torch.stack(norms)
    nonfinite = ~torch.isfinite(stacked).all()
    noop_flag.copy_(torch.where(nonfinite, torch.ones_like(noop_flag), noop_flag))
    total_norm = stacked.norm()

    if per_tensor:
        return total_norm, stacked
    return total_norm, None


_APPLIED = False


def patch_megatron_optimizer():
    """Patch TENPU exports and MCore gradient-clipping aliases."""
    global _APPLIED
    if _APPLIED:
        return
    from megatron.core.optimizer import clip_grads
    from transformer_engine.pytorch import optimizers

    functions = {
        'multi_tensor_l2norm': multi_tensor_l2norm,
        'multi_tensor_unscale_l2norm': multi_tensor_unscale_l2norm,
        'multi_tensor_scale': multi_tensor_scale,
        'multi_tensor_scale_tensor': multi_tensor_scale_tensor,
    }
    # Import consumers before replacing cached aliases.
    norm_module = importlib.import_module('transformer_engine.pytorch.optimizers.multi_tensor_l2norm')
    scale_module = importlib.import_module('transformer_engine.pytorch.optimizers.multi_tensor_scale')
    for name, fn in functions.items():
        original = getattr(optimizers, name)
        if tuple(inspect.signature(original).parameters) != tuple(inspect.signature(fn).parameters):
            raise RuntimeError(f'Unsupported TENPU {name} signature for the Swift NPU performance patch.')
    for name, fn in functions.items():
        module = norm_module if 'l2norm' in name else scale_module
        setattr(module, name, fn)
        setattr(optimizers, name, fn)
        if hasattr(clip_grads, name):
            setattr(clip_grads, name, fn)
    clip_grads.l2_norm_impl = multi_tensor_l2norm
    clip_grads.multi_tensor_scale_impl = multi_tensor_scale
    clip_grads.multi_tensor_scale_tensor_impl = multi_tensor_scale_tensor
    _APPLIED = True
