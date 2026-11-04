# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.
"""NPU gradient norm and scaling optimizations."""

import importlib
import inspect
import torch


def _set_overflow(flag, nonfinite):
    flag.copy_(torch.where(nonfinite.reshape(flag.shape), torch.ones_like(flag), flag))


def _can_scale_in_place(inputs, outputs):
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


def multi_tensor_scale(chunk_size, is_infinite, tensor_lists, scale):
    """Scale gradients and flag non-finite inputs; chunk_size is ignored."""
    if len(tensor_lists) != 2 or len(tensor_lists[0]) != len(tensor_lists[1]):
        raise AssertionError('Expected equally sized input and output lists')
    inputs, outputs = tensor_lists
    with torch.no_grad():
        if inputs and _can_scale_in_place(inputs, outputs):
            found_inf = torch.zeros(1, device=inputs[0].device, dtype=torch.float32)
            factor = torch.as_tensor(scale, device=inputs[0].device, dtype=torch.float32)
            torch._amp_foreach_non_finite_check_and_unscale_([tensor for tensor in inputs if tensor.numel()], found_inf,
                                                             factor)
            _set_overflow(is_infinite, found_inf.ne(0))
        else:
            finite = []
            for src, dst in zip(inputs, outputs):
                # Check each input before writing, including overlapping views.
                finite.append(torch.isfinite(src).all())
                dst.copy_(src)
                dst.mul_(scale)
            if finite:
                _set_overflow(is_infinite, ~torch.stack(finite).all())


multi_tensor_scale_tensor = multi_tensor_scale


def _l2norm(noop_flag, tensor_lists, per_tensor, inv_scale=None):
    grads = [grad for group in tensor_lists for grad in group]
    if not grads:
        return torch.zeros(
            1, dtype=torch.float32, device='cpu'), (
                torch.empty(0, dtype=torch.float32, device='cpu') if per_tensor else None)
    norms = torch.stack([(grad.float() if inv_scale is None else grad.float() * inv_scale).norm() for grad in grads])
    _set_overflow(noop_flag, ~torch.isfinite(norms).all())
    return norms.norm(), norms if per_tensor else None


def multi_tensor_l2norm(chunk_size, noop_flag, tensor_lists, per_tensor=False):
    """Return the total L2 norm and optional per-tensor norms."""
    return _l2norm(noop_flag, tensor_lists, per_tensor)


def multi_tensor_unscale_l2norm(chunk_size, noop_flag, tensor_lists, inv_scale, per_tensor=False):
    """Compute unscaled norms without modifying gradients or reading device scalars."""
    return _l2norm(noop_flag, tensor_lists, per_tensor,
                   inv_scale.detach().to(device=noop_flag.device, dtype=torch.float32))


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
    # Load consumers before replacing their cached aliases.
    modules = {
        kind: importlib.import_module(f'transformer_engine.pytorch.optimizers.multi_tensor_{kind}')
        for kind in ('l2norm', 'scale')
    }
    for name, fn in functions.items():
        if tuple(inspect.signature(getattr(optimizers, name)).parameters) != tuple(inspect.signature(fn).parameters):
            raise RuntimeError(f'Unsupported TENPU {name} signature for the Swift NPU performance patch.')
    for name, fn in functions.items():
        setattr(modules['l2norm' if 'l2norm' in name else 'scale'], name, fn)
        setattr(optimizers, name, fn)
        if hasattr(clip_grads, name):
            setattr(clip_grads, name, fn)
    clip_grads.l2_norm_impl = multi_tensor_l2norm
    clip_grads.multi_tensor_scale_impl = multi_tensor_scale
    clip_grads.multi_tensor_scale_tensor_impl = multi_tensor_scale_tensor
    _APPLIED = True
