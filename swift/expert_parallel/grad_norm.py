# Copyright (c) ModelScope Contributors. All rights reserved.
"""Multi-mesh gradient norm clipping for Expert-Parallel + FSDP2.

When EP is active, model parameters span multiple DTensor device meshes
(the ``ep`` mesh for expert shards, the global/FSDP mesh for everything
else).  Additionally, FSDP2 CPUOffload creates CPU-side DTensor gradients
that cannot go through the standard c10d all_reduce (CPU backend missing).

This module provides a unified ``clip_grad_norm_multi_mesh`` that:
  - separates CPU DTensor grads from the rest,
  - groups remaining params by device mesh,
  - computes per-group norms and merges them into a global total_norm,
  - clips each group with the combined total_norm,
  - guards against NaN (zeroes all grads and returns NaN).
"""
import torch
from torch.distributed.tensor import DTensor


def get_param_mesh(p):
    """Return the DTensor device_mesh of a parameter (or its grad), or None.

    Prefers ``p.grad`` as the mesh key when it is a DTensor (grads may be
    DTensors even when the param itself is a plain Tensor after FSDP2
    sharding).  Falls back to ``p`` itself.
    """
    target = p.grad if (p.grad is not None and isinstance(p.grad, DTensor)) else p
    if isinstance(target, DTensor):
        return target.device_mesh
    return None


@torch.no_grad()
def norm_of_cpu_dtensor_grad(grad, norm_type):
    """Compute the norm of a CPU DTensor grad without using DTensor all_reduce.

    DTensor's c10d_functional backend only supports CUDA, so we materialise
    the full tensor on the mesh device, compute the norm there, and return a
    plain CPU scalar.
    """
    full = grad.to(grad.device_mesh.device_type).full_tensor()
    return torch.linalg.vector_norm(full, norm_type).cpu()


@torch.no_grad()
def clip_cpu_dtensor_grad_with_norm_(param, max_norm, total_norm):
    """Clip a CPU DTensor grad in-place using a pre-computed *total_norm*.

    We cannot assign a CUDA grad to a CPU param, so we scale the local CPU
    shard directly.
    """
    if total_norm == 0 or not torch.isfinite(total_norm):
        return
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        local_grad = param.grad.to_local()
        local_grad.mul_(clip_coef)


def clip_grad_norm_multi_mesh(accelerator, parameters, *args, origin_fn=None, **kwargs):
    """Clip gradients across multiple DTensor meshes + CPU DTensor grads.

    This is a drop-in replacement for ``Accelerator.clip_grad_norm_`` that
    handles three parameter groups simultaneously:

    1. **CPU DTensor grads** — arise under FSDP2 CPUOffload; DTensor's
       ``all_reduce`` has no CPU backend, so we compute norms by
       materialising via ``.to(mesh_device).full_tensor()``.
    2. **Multi-mesh CUDA DTensor grads** — EP introduces a second device
       mesh; we group params by mesh and compute norms per group.
    3. **Single-mesh params** — delegated to the original clip function.

    Args:
        accelerator: The ``Accelerator`` instance (``self`` in the patched method).
        parameters: Iterable of parameters with gradients.
        *args: Positional args forwarded — ``args[0]`` is ``max_norm``,
            ``args[1]`` (optional) is ``norm_type``.
        origin_fn: The original ``Accelerator.clip_grad_norm_`` to delegate
            single-mesh fast-path to.
        **kwargs: Keyword args forwarded — ``max_norm``, ``norm_type``.
    """
    parameters = [p for p in parameters if p.grad is not None]

    # ---- Separate CPU DTensor grads from the rest ----
    cpu_dt_params = []  # params whose grad is a CPU DTensor (FSDP2 CPUOffload)
    other_params = []   # everything else (CUDA DTensors, plain tensors)
    for p in parameters:
        g = p.grad
        if isinstance(g, DTensor) and g.device.type == 'cpu':
            cpu_dt_params.append(p)
        else:
            other_params.append(p)

    max_norm = args[0] if len(args) > 0 else kwargs.get('max_norm')
    norm_type = float(args[1]) if len(args) > 1 else float(kwargs.get('norm_type', 2.0))

    # ---- Compute norms for the two groups separately ----
    other_norm = None
    if other_params:
        # Group other_params by mesh for the original logic
        mesh_to_params = {}
        for p in other_params:
            key = get_param_mesh(p)
            mesh_to_params.setdefault(key, []).append(p)
        if len(mesh_to_params) <= 1 and not cpu_dt_params:
            # Single mesh, no CPU DTensor grads — delegate entirely
            grad_norm = origin_fn(accelerator, other_params, *args, **kwargs)
            if isinstance(grad_norm, torch.Tensor) and grad_norm.isnan().item():
                for p in other_params:
                    p.grad = None
            return grad_norm
        # Multi-mesh path for other_params
        group_norms = []
        for group_params in mesh_to_params.values():
            grads = [p.grad for p in group_params]
            group_norm = torch.nn.utils.get_total_norm(
                grads, norm_type=norm_type, error_if_nonfinite=False
            )
            if isinstance(group_norm, DTensor):
                if group_norm.device.type == 'cpu':
                    group_norm = group_norm.to(group_norm.device_mesh.device_type)
                group_norm = group_norm.full_tensor()
            group_norms.append(group_norm)
        if group_norms:
            stacked = torch.stack([g.to(group_norms[0].device).reshape(()) for g in group_norms])
            other_norm = torch.linalg.vector_norm(stacked, norm_type)

    cpu_dt_norm = None
    if cpu_dt_params:
        cpu_norms = [norm_of_cpu_dtensor_grad(p.grad, norm_type) for p in cpu_dt_params]
        cpu_dt_norm = torch.linalg.vector_norm(torch.stack(cpu_norms), norm_type)

    # ---- Merge into a global total_norm ----
    norm_parts = [n for n in (other_norm, cpu_dt_norm) if n is not None]
    if len(norm_parts) == 1:
        total_norm = norm_parts[0]
    else:
        # Both are plain CPU scalars; combine on CPU
        total_norm = torch.linalg.vector_norm(torch.stack([n.cpu().reshape(()) for n in norm_parts]), norm_type)

    # ---- Clip each group using the combined total_norm ----
    if other_params:
        torch.nn.utils.clip_grads_with_norm_(other_params, max_norm, total_norm)
    for p in cpu_dt_params:
        clip_cpu_dtensor_grad_with_norm_(p, max_norm, total_norm)

    # NaN guard
    if isinstance(total_norm, torch.Tensor) and total_norm.isnan().item():
        for p in parameters:
            p.grad = None
    return total_norm