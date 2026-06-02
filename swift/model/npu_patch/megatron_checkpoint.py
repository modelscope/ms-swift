# Copyright (c) ModelScope Contributors. All rights reserved.
"""NPU-only Megatron checkpoint compatibility helpers.

MindSpeed patches Megatron's distributed optimizer on NPU, but some Megatron-Core
checkpoint formats still need the native Megatron param_state loaders.
"""

from __future__ import annotations

import torch
from contextlib import contextmanager

from swift.utils import get_logger

logger = get_logger()


def _iter_optimizer_param_groups(optimizer):
    visited = set()

    def visit(obj):
        if obj is None or id(obj) in visited:
            return
        visited.add(id(obj))

        param_groups = getattr(obj, 'param_groups', None)
        if param_groups is not None:
            yield param_groups

        inner_optimizer = getattr(obj, 'optimizer', None)
        if inner_optimizer is not obj:
            yield from visit(inner_optimizer)

        for child in getattr(obj, 'chained_optimizers', []) or []:
            yield from visit(child)
        for child in getattr(obj, 'sub_optimizers', []) or []:
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


def _iter_distributed_optimizers(optimizer):
    visited = set()

    def visit(obj):
        if obj is None or id(obj) in visited:
            return
        visited.add(id(obj))

        if hasattr(obj, 'load_parameter_state_from_dp_reshardable') or hasattr(
                obj, 'load_parameter_state_from_fully_reshardable'):
            yield obj
            return

        for child in getattr(obj, 'chained_optimizers', []) or []:
            yield from visit(child)
        for child in getattr(obj, 'sub_optimizers', []) or []:
            yield from visit(child)

    yield from visit(optimizer)


def _has_mindspeed_patched_load_state_dict(distributed_optimizer):
    load_state_dict = getattr(type(distributed_optimizer), 'load_state_dict', None)
    return getattr(load_state_dict, '__module__', '').startswith('mindspeed.')


_MEGATRON_RESHARDABLE_PARAM_STATE_LOADERS = {
    'dp_reshardable': 'load_parameter_state_from_dp_reshardable',
    'fully_reshardable': 'load_parameter_state_from_fully_reshardable',
}


def _current_npu_device():
    if hasattr(torch, 'npu'):
        return torch.device('npu', torch.npu.current_device())
    return torch.cuda.current_device()


def _restore_mindspeed_optimizer_step_tensors(optimizer):
    restored_count = 0
    for param_groups in _iter_optimizer_param_groups(optimizer):
        for param_group in param_groups:
            step = param_group.get('step')
            if isinstance(step, torch.Tensor):
                continue
            if isinstance(step, (int, float)):
                param_group['step'] = torch.tensor(int(step), dtype=torch.int64, device=_current_npu_device())
                restored_count += 1
    if restored_count:
        logger.warning(f'Restored {restored_count} MindSpeed optimizer param-group step values to NPU tensors.')


def _split_chained_optimizer_state_dict(chained_optimizers, state_dict):
    if isinstance(state_dict, dict):
        state_dicts = [v for _k, v in sorted(state_dict.items())]
    else:
        state_dicts = list(state_dict)
    if len(chained_optimizers) != len(state_dicts):
        raise RuntimeError(
            f'Expected {len(chained_optimizers)} entries in optimizer state dict, but got {len(state_dicts)}.')
    return state_dicts


def _load_chained_optimizer_state_dict(optimizer, state_dict):
    chained_optimizers = getattr(optimizer, 'chained_optimizers', None)
    if not chained_optimizers or len(chained_optimizers) <= 1:
        return False

    state_dicts = _split_chained_optimizer_state_dict(chained_optimizers, state_dict)
    for child_optimizer, child_state_dict in zip(chained_optimizers, state_dicts):
        load_optimizer_state_dict(child_optimizer, child_state_dict)
    synchronize_steps = getattr(optimizer, '_synchronize_steps', None)
    if synchronize_steps is not None:
        synchronize_steps()
    return True


def load_optimizer_state_dict(optimizer, state_dict):
    if _load_chained_optimizer_state_dict(optimizer, state_dict):
        return

    distributed_optimizers = list(_iter_distributed_optimizers(optimizer))
    mindspeed_patched = any(
        _has_mindspeed_patched_load_state_dict(distributed_optimizer)
        for distributed_optimizer in distributed_optimizers)
    sharding_type = state_dict.get('param_state_sharding_type') if isinstance(state_dict, dict) else None
    native_loader_name = _MEGATRON_RESHARDABLE_PARAM_STATE_LOADERS.get(sharding_type)
    if native_loader_name is None:
        optimizer.load_state_dict(state_dict)
        if mindspeed_patched:
            _restore_mindspeed_optimizer_step_tensors(optimizer)
        return

    if not mindspeed_patched:
        optimizer.load_state_dict(state_dict)
        return

    if len(distributed_optimizers) != 1:
        raise RuntimeError(f'MindSpeed optimizer checkpoint compatibility supports exactly one distributed optimizer, '
                           f'got {len(distributed_optimizers)}.')
    distributed_optimizer = distributed_optimizers[0]
    if not hasattr(distributed_optimizer, native_loader_name):
        raise RuntimeError(f'Distributed optimizer does not support sharding type {sharding_type}.')

    state_dict_without_param_state = dict(state_dict)
    param_state = state_dict_without_param_state.pop('param_state', None)
    state_dict_without_param_state.pop('param_state_sharding_type', None)
    if param_state is None:
        raise RuntimeError(f'Optimizer checkpoint missing param_state for sharding type {sharding_type}.')

    logger.warning(f'Loading optimizer param_state with ms-swift compatibility path because MindSpeed '
                   f'DistributedOptimizer.load_state_dict does not support {sharding_type}.')
    # Let MindSpeed restore the generic optimizer state; load the missing
    # reshardable param_state with Megatron-Core's native implementation.
    optimizer.load_state_dict(state_dict_without_param_state)
    _restore_mindspeed_optimizer_step_tensors(optimizer)
    getattr(distributed_optimizer, native_loader_name)(param_state)


__all__ = ['load_optimizer_state_dict', 'optimizer_sharded_state_dict']
