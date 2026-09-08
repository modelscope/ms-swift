# Copyright (c) ModelScope Contributors. All rights reserved.
"""NPU-only Megatron checkpoint compatibility helpers.

Megatron Core builds optimizer checkpoint state from scalar step values. On
Ascend, those values may be NPU tensors with object identity based hashing, so
normalize them temporarily while MCore builds the sharded state dict.
"""

from __future__ import annotations

import torch
from contextlib import contextmanager

from swift.utils import get_logger

logger = get_logger()


def _iter_optimizer_children(obj):
    """The optimizers nested inside `obj`, whichever way it holds them.

    `ChainedOptimizer` exposes `optimizer` as a backwards-compatible property that asserts it holds exactly
    one, so reading it blows up on a chain of several -- and `getattr`'s default only covers `AttributeError`.
    The chained members are reachable through their own list anyway, so leave that property alone whenever
    there is one.
    """
    chained_optimizers = getattr(obj, 'chained_optimizers', None) or []
    sub_optimizers = getattr(obj, 'sub_optimizers', None) or []
    if chained_optimizers or sub_optimizers:
        return [*chained_optimizers, *sub_optimizers]

    inner_optimizer = getattr(obj, 'optimizer', None)
    return [inner_optimizer] if inner_optimizer is not None and inner_optimizer is not obj else []


def _iter_optimizer_param_groups(optimizer):
    visited = set()

    def visit(obj):
        if obj is None or id(obj) in visited:
            return
        visited.add(id(obj))

        param_groups = getattr(obj, 'param_groups', None)
        if param_groups is not None:
            yield param_groups

        for child in _iter_optimizer_children(obj):
            yield from visit(child)

    yield from visit(optimizer)


def _step_to_int(step):
    if isinstance(step, torch.Tensor):
        if step.numel() != 1:
            raise RuntimeError(f'Optimizer step tensor must be scalar, got shape: {tuple(step.shape)}')
        return int(step.item())
    return int(step)


@contextmanager
def _canonicalize_optimizer_steps_for_checkpoint(optimizer):
    """Normalize NPU scalar step tensors while Megatron builds optimizer checkpoint state.

    Megatron-Core deduplicates param-group steps with set(). Equal NPU scalar
    tensors can still hash as distinct objects, so use their numeric value only
    while sharded_state_dict() is being built and restore the optimizer in place.
    """
    saved_steps = []
    numeric_steps = set()
    for param_groups in _iter_optimizer_param_groups(optimizer):
        for param_group in param_groups:
            if len(param_group.get('params', [])) == 0 or 'step' not in param_group:
                continue
            step = param_group['step']
            numeric_step = _step_to_int(step)
            saved_steps.append((param_group, step))
            numeric_steps.add(numeric_step)

    if len(numeric_steps) > 1:
        raise RuntimeError(f'Inconsistent optimizer steps before checkpoint save: {sorted(numeric_steps)}')

    canonical_step = next(iter(numeric_steps), None)
    try:
        if canonical_step is not None:
            for param_group, _step in saved_steps:
                param_group['step'] = canonical_step
            if any(isinstance(step, torch.Tensor) for _param_group, step in saved_steps):
                logger.warning(f'Canonicalized optimizer param-group step to {canonical_step} for checkpoint save.')
        yield
    finally:
        for param_group, step in saved_steps:
            param_group['step'] = step


def optimizer_sharded_state_dict(optimizer, state_dict, **optim_sd_kwargs):
    with _canonicalize_optimizer_steps_for_checkpoint(optimizer):
        return optimizer.sharded_state_dict(state_dict, **optim_sd_kwargs)


__all__ = ['optimizer_sharded_state_dict']
