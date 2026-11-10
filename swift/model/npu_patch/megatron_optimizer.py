# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.
"""NPU gradient norm and scaling optimizations."""

import importlib
import inspect
import sys
import torch


def _set_overflow(flag, nonfinite):
    flag.copy_(torch.where(nonfinite.reshape(flag.shape), torch.ones_like(flag), flag))


def _can_scale_in_place(inputs, outputs):
    device, dtype = inputs[0].device, inputs[0].dtype
    if device.type != 'npu' or dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    for src, dst in zip(inputs, outputs):
        if src is not dst or src.device != device or src.dtype != dtype or not src.is_contiguous():
            return False
    end, nonempty = -1, False
    for src in sorted(inputs, key=lambda tensor: tensor.data_ptr()):
        if not src.numel():
            continue
        start = src.data_ptr()
        if start < end:
            return False
        end, nonempty = start + src.numel() * src.element_size(), True
    return nonempty


def _coalesce_grads(grads):
    """Join adjacent views without copying or including buffer padding."""
    regions = []
    storage_id, end = None, -1
    for storage, offset, grad in sorted([(g.untyped_storage()._cdata, g.storage_offset(), g) for g in grads],
                                        key=lambda item: item[:2]):
        if storage != storage_id and grad.device.type == 'npu':
            import torch_npu
            if torch_npu.get_npu_format(grad) not in (0, 2):
                return grads
        size = grad.numel()
        if regions and storage == storage_id and offset == end and size and regions[-1][1]:
            regions[-1][1] += size
            end += size
        else:
            regions.append([grad, size])
            storage_id, end = storage, offset + size
    return [grad.as_strided((size, ), (1, ), grad.storage_offset()) for grad, size in regions]


def multi_tensor_scale(chunk_size, is_infinite, tensor_lists, scale):
    """Scale gradients and flag non-finite inputs; chunk_size is ignored."""
    if len(tensor_lists) != 2 or len(tensor_lists[0]) != len(tensor_lists[1]):
        raise AssertionError('Expected equally sized input and output lists')
    inputs, outputs = tensor_lists
    with torch.no_grad():
        if inputs and _can_scale_in_place(inputs, outputs):
            if not torch.compiler.is_compiling() and all(not tensor.requires_grad for tensor in inputs):
                inputs = _coalesce_grads(inputs)
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
                dst.copy_(src.to(torch.promote_types(src.dtype, dst.dtype)) * scale)
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


def get_grad_norm_fp32(grads_for_norm, norm_type=2, grad_stats_parallel_group=None):
    """Reduce contiguous gradient regions instead of individual parameter views."""
    if (float(norm_type) == 2 and isinstance(grads_for_norm, (list, tuple)) and len(grads_for_norm) > 1
            and not torch.compiler.is_compiling() and all(
                type(g) in (torch.Tensor, torch.nn.Parameter) and g.device.type == 'npu' and g.device ==
                grads_for_norm[0].device and g.dtype == torch.float32 and g.is_contiguous() and not g.requires_grad
                for g in grads_for_norm)):
        grads_for_norm = _coalesce_grads(grads_for_norm)
    return _ORIGINAL_GET_NORM(grads_for_norm, norm_type, grad_stats_parallel_group)


_APPLIED = False
_ORIGINAL_GET_NORM = None


def patch_megatron_optimizer():
    """Patch TENPU exports and MCore gradient-clipping aliases."""
    global _APPLIED, _ORIGINAL_GET_NORM
    if _APPLIED:
        return
    from megatron.core.optimizer import clip_grads
    from megatron.core.optimizer import optimizer as core_optimizer
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
    if tuple(inspect.signature(clip_grads.get_grad_norm_fp32).parameters) != tuple(
            inspect.signature(get_grad_norm_fp32).parameters):
        raise RuntimeError('Unsupported MCore gradient norm signature for the Swift NPU performance patch.')
    _ORIGINAL_GET_NORM = clip_grads.get_grad_norm_fp32
    clip_grads.get_grad_norm_fp32 = get_grad_norm_fp32
    core_optimizer.get_grad_norm_fp32 = get_grad_norm_fp32
    layer_wise_optimizer = sys.modules.get('megatron.core.optimizer.layer_wise_optimizer')
    if layer_wise_optimizer is not None:
        layer_wise_optimizer.get_grad_norm_fp32 = get_grad_norm_fp32
    for name, fn in functions.items():
        setattr(modules['l2norm' if 'l2norm' in name else 'scale'], name, fn)
        setattr(optimizers, name, fn)
        if hasattr(clip_grads, name):
            setattr(clip_grads, name, fn)
    clip_grads.l2_norm_impl = multi_tensor_l2norm
    clip_grads.multi_tensor_scale_impl = multi_tensor_scale
    clip_grads.multi_tensor_scale_tensor_impl = multi_tensor_scale_tensor
    _APPLIED = True
