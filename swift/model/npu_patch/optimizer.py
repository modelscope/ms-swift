# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import torch

from swift.utils.logger import get_logger

logger = get_logger()

_APPLIED = False


def _make_compat_npu_fused_adamw(base_cls):
    cache_attributes = (
        'params_all_group_combined',
        'params_all_group',
        'combined_params_indexed_by_group',
        'params_lists_indexed_by_group',
        'combined_param_states_indexed_by_group',
        'grads_all_group_combined',
        'combined_grads_indexed_by_group',
    )

    class SwiftNpuFusedAdamW(base_cls):
        """Make torch_npu's fused AdamW safe for changing optimizer param groups.

        torch_npu caches the combined parameter/state tensors on the first
        ``step``. DeepSpeed ZeRO-1/2 temporarily replaces ``param_groups`` with
        one group while stepping it. Keep parameter/state caches per group and
        only rebuild a group's gradient buffer when DeepSpeed supplies a new
        gradient tensor. The same cache invalidation handles a parameter that
        gets its first gradient in a later step.
        """

        _ms_swift_npu_fused_adamw_patch = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._ms_swift_combined_signature = None
            self._ms_swift_group_caches = {}
            self._ms_swift_param_state_rebuilds = 0
            self._ms_swift_grad_rebuilds = 0

        def _combined_signature(self):
            return tuple(
                tuple(id(param) for param in group.get('params', []) if param.grad is not None)
                for group in self.param_groups)

        @staticmethod
        def _group_param_signature(group):
            return tuple(id(param) for param in group.get('params', []) if param.grad is not None)

        @staticmethod
        def _group_grad_signature(group):
            return tuple((id(param), id(param.grad), param.grad.data_ptr())
                         for param in group.get('params', []) if param.grad is not None)

        def _reset_combined_caches(self):
            self.is_params_grads_combined = False
            self.is_states_combined = False
            self.is_grads_masks_combined = False
            self.combined_params_indexed_by_group = None
            self.combined_grads_indexed_by_group = None
            self.combined_param_states_indexed_by_group = None
            self.params_lists_indexed_by_group = []

            dtype_count = len(getattr(self, 'params_all_group_combined', (None, None)))
            self.params_all_group_combined = [None] * dtype_count
            self.grads_all_group_combined = [None] * dtype_count
            self._ms_swift_active_param_formats = None
            if hasattr(self, 'combined_grads_masks'):
                self.combined_grads_masks = None

        def _capture_group_cache(self, group, param_signature):
            cache = {name: getattr(self, name) for name in cache_attributes}
            import torch_npu

            cache['param_formats_by_dtype'] = [
                [torch_npu.get_npu_format(param) for param in params]
                for params in self.params_lists_indexed_by_group[0]
            ]
            cache.update({
                'group': group,
                'param_signature': param_signature,
                'grad_signature': self._group_grad_signature(group),
                'is_params_grads_combined': self.is_params_grads_combined,
                'is_states_combined': self.is_states_combined,
            })
            return cache

        def _load_group_cache(self, cache):
            for name in cache_attributes:
                setattr(self, name, cache[name])
            self._ms_swift_active_param_formats = cache['param_formats_by_dtype']
            self.is_params_grads_combined = cache['is_params_grads_combined']
            self.is_states_combined = cache['is_states_combined']
            self.is_grads_masks_combined = False
            if hasattr(self, 'combined_grads_masks'):
                self.combined_grads_masks = None

        def _refresh_group_grads(self):
            import torch_npu
            from torch_npu.utils import get_part_combined_tensor, npu_combine_tensors

            # The cached parameter lists are grouped by dtype in the same shape
            # as NpuFusedOptimizerBase.params_lists_indexed_by_group.
            params_by_dtype = self.params_lists_indexed_by_group[0]
            grads_by_dtype = [[], []]
            grads_size_by_dtype = [0, 0]
            for dtype_index, params in enumerate(params_by_dtype):
                param_formats = self._ms_swift_active_param_formats[dtype_index]
                for param, param_format in zip(params, param_formats):
                    grad = param.grad
                    if grad is None:
                        raise RuntimeError('A cached NPU fused optimizer group lost its gradient.')
                    if param_format != torch_npu.get_npu_format(grad):
                        param.grad = torch_npu.npu_format_cast(grad, param_format).contiguous()
                        grad = param.grad
                    grads_by_dtype[dtype_index].append(grad)
                    grads_size_by_dtype[dtype_index] += grad.storage().size()

            grads_all_group = [npu_combine_tensors(grads) for grads in grads_by_dtype]
            combined_group_grads = []
            for dtype_index, combined_grad in enumerate(grads_all_group):
                combined_group_grads.append(
                    get_part_combined_tensor(combined_grad, 0, grads_size_by_dtype[dtype_index]))

            self.grads_all_group_combined = grads_all_group
            self.combined_grads_indexed_by_group = [combined_group_grads]
            self.is_params_grads_combined = not all(value is None for value in self.params_all_group_combined)
            self.is_grads_masks_combined = False
            if hasattr(self, 'combined_grads_masks'):
                self.combined_grads_masks = None
            self._ms_swift_grad_rebuilds += 1

        def _step_single_group(self, closure):
            group = self.param_groups[0]
            group_key = id(group)
            param_signature = self._group_param_signature(group)
            cache = self._ms_swift_group_caches.get(group_key)

            if cache is None or cache['param_signature'] != param_signature:
                self._reset_combined_caches()
                self._ms_swift_param_state_rebuilds += 1
                loss = super().step(closure)
                self._ms_swift_group_caches[group_key] = self._capture_group_cache(group, param_signature)
                return loss

            self._load_group_cache(cache)
            grad_signature = self._group_grad_signature(group)
            if grad_signature != cache['grad_signature']:
                self._refresh_group_grads()
                cache = self._capture_group_cache(group, param_signature)
                self._ms_swift_group_caches[group_key] = cache

            return super().step(closure)

        def load_state_dict(self, state_dict):
            result = super().load_state_dict(state_dict)
            self._ms_swift_combined_signature = None
            self._ms_swift_group_caches.clear()
            self._reset_combined_caches()
            return result

        @torch.no_grad()
        def step(self, closure=None):
            # DeepSpeed ZeRO-1/2 calls step once per temporary one-group view.
            # Reusing the cache belonging to that group avoids rebuilding the
            # large combined parameter and Adam-state tensors every time.
            if len(self.param_groups) == 1:
                return self._step_single_group(closure)

            signature = self._combined_signature()
            if (self._ms_swift_combined_signature is not None
                    and signature != self._ms_swift_combined_signature):
                self._reset_combined_caches()

            loss = super().step(closure)
            self._ms_swift_combined_signature = signature
            return loss

    SwiftNpuFusedAdamW.__name__ = 'SwiftNpuFusedAdamW'
    SwiftNpuFusedAdamW.__qualname__ = 'SwiftNpuFusedAdamW'
    return SwiftNpuFusedAdamW


def apply_patch() -> None:
    global _APPLIED
    if _APPLIED:
        return

    import torch_npu.optim as npu_optim
    import torch_npu.optim.npu_fused_adamw as npu_fused_adamw

    original_cls = npu_fused_adamw.NpuFusedAdamW
    if getattr(original_cls, '_ms_swift_npu_fused_adamw_patch', False):
        _APPLIED = True
        return

    compat_cls = _make_compat_npu_fused_adamw(original_cls)
    # Keep the class name in torch_npu.optim.npu_fused_adamw unchanged. The
    # original implementation uses ``super(NpuFusedAdamW, self)`` in its
    # methods, so replacing that module global would make the original
    # initializer resolve the wrong class.
    npu_optim.NpuFusedAdamW = compat_cls
    logger.info('Applied ms-swift compatibility patch for torch_npu NpuFusedAdamW per-group cache reuse.')
    _APPLIED = True


__all__ = ['apply_patch']
