# Copyright (c) ModelScope Contributors. All rights reserved.
"""NPU-only Megatron-FSDP optimizer compatibility.

MCore's GPU implementation has two valid optimizer contracts for
Megatron-FSDP: Transformer Engine's precision-aware FusedAdam, or the native
PyTorch AdamW fallback when the fused optimizer is unavailable.  The
TransformerEngineNPU FusedAdam currently rejects DTensor parameters, so the
NPU path implements the latter contract locally.  The adapter below accepts
the small TE/Apex constructor surface that MCore uses, then delegates the
actual update and state handling to ``torch.optim.AdamW``.

The patch is requested only by NPU Megatron-FSDP, but changes process-global
MCore optimizer aliases. Mixing FSDP and non-FSDP optimizers in the same
process is not supported by this adapter. GPU-only processes are unchanged.
"""

from __future__ import annotations

import copy
import torch

from swift.utils import get_logger

logger = get_logger()

# 1. NPU Megatron-FSDP optimizer adaptation.

_APPLIED = False


class NPUFSDPAdamW(torch.optim.AdamW):
    """Native AdamW with MCore's FusedAdam-compatible constructor surface.

    MCore selects the optimizer class through module-level aliases.  When TE
    is installed, its normal FusedAdam alias receives arguments that native
    AdamW does not understand (``adam_w_mode``, ``bias_correction`` and the
    precision-aware state options).  This class makes that boundary explicit
    instead of relying on a bare alias to ``torch.optim.AdamW``.

    Only the standard, non-precision-aware FSDP contract is supported.  A
    precision-aware request must fail in ``patch_megatron_fsdp_optimizer``;
    silently dropping master weights or decoupled gradients would change the
    optimizer semantics.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        adam_w_mode=True,
        bias_correction=True,
        capturable=False,
        use_decoupled_grad=False,
        master_weights=False,
        master_weight_dtype=torch.float32,
        exp_avg_dtype=None,
        exp_avg_sq_dtype=None,
        store_param_remainders=False,
        fused=None,
        set_grad_none=None,
        **kwargs,
    ):
        if not adam_w_mode:
            raise ValueError('NPU Megatron-FSDP requires AdamW decoupled weight decay.')
        if not bias_correction:
            raise ValueError('NPU Megatron-FSDP native AdamW requires bias correction.')
        if use_decoupled_grad:
            raise ValueError('NPU Megatron-FSDP native AdamW does not support decoupled_grad; '
                             'disable precision-aware optimizer.')
        if master_weights or store_param_remainders:
            raise ValueError('NPU Megatron-FSDP native AdamW does not own TE master weights or parameter remainders.')
        if exp_avg_dtype is not None or exp_avg_sq_dtype is not None:
            raise ValueError('NPU Megatron-FSDP native AdamW does not support precision-aware optimizer state dtypes.')
        if set_grad_none is not None:
            raise ValueError('NPU Megatron-FSDP native AdamW uses zero_grad(set_to_none=...); '
                             'set_grad_none is not a constructor option.')
        if kwargs:
            unknown = ', '.join(sorted(kwargs))
            raise TypeError(f'Unsupported NPU Megatron-FSDP AdamW arguments: {unknown}')

        # ``fused`` is a TE/Apex hint.  The NPU implementation intentionally
        # uses torch's ordinary AdamW kernels; dropping this hint is safe for
        # the standard FSDP path and avoids asking torch_npu for CUDA fusion.
        del fused, master_weight_dtype
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            capturable=capturable,
        )


def patch_megatron_fsdp_optimizer(*, use_precision_aware_optimizer: bool = False) -> None:
    """Use native Torch AdamW for NPU Megatron-FSDP DTensor parameters.

    MCore keeps the optimizer class and backend-selection flags in both
    ``megatron.core.optimizer`` and ``megatron.core.optimizer.distrib_optimizer``
    module globals.  Updating both modules is necessary because the latter
    performs its own ``Adam`` import and type checks.

    Precision-aware Adam is intentionally not silently downgraded: the NPU
    FSDP compatibility path currently targets the standard BF16/FP32 AdamW
    contract.  Callers using precision-aware optimizer settings receive a
    clear error instead of a partially compatible optimizer.
    """

    global _APPLIED
    if use_precision_aware_optimizer:
        raise ValueError('NPU Megatron-FSDP currently requires use_precision_aware_optimizer=False; '
                         'the NPU FSDP compatibility path uses native torch.optim.AdamW for DTensor parameters.')
    if _APPLIED:
        return

    import megatron.core.optimizer as mcore_optimizer
    import megatron.core.optimizer.distrib_optimizer as distributed_optimizer

    # MCore's default import prefers TE whenever it is installed.  Switch only
    # the already-loaded optimizer aliases used to build the FSDP optimizer;
    # TE model/attention/linear callables remain untouched.  Mark the alias as
    # a non-TE optimizer so MCore passes the FusedAdam-compatible constructor
    # arguments to our explicit adapter and uses its native DTensor state
    # handling in DistributedOptimizer.
    mcore_optimizer.Adam = NPUFSDPAdamW
    mcore_optimizer.USING_PYTORCH_OPTIMIZER = False
    distributed_optimizer.Adam = NPUFSDPAdamW
    distributed_optimizer.HAVE_APEX_OR_TE = False
    distributed_optimizer.USING_TE_OPTIMIZER = False
    distributed_optimizer.USING_APEX_OPTIMIZER = False
    _APPLIED = True
    logger.info('NPU Megatron-FSDP: using Swift NPUFSDPAdamW (native torch.optim.AdamW) for DTensor parameters.')


# 2. Refresh stale optimizer steps on empty local shards.


def _optimizer_step_value(step, param_name):
    if isinstance(step, torch.Tensor):
        if step.numel() != 1:
            raise RuntimeError(
                f'Megatron-FSDP optimizer step for `{param_name}` must be scalar, got shape {tuple(step.shape)}.')
        return step.detach().item()
    if isinstance(step, (int, float)):
        return step
    raise RuntimeError(f'Megatron-FSDP optimizer step for `{param_name}` has unsupported type {type(step).__name__}.')


def _clone_optimizer_step(step, existing_step=None):
    if isinstance(step, torch.Tensor):
        cloned_step = step.detach().clone()
        if isinstance(existing_step, torch.Tensor):
            cloned_step = cloned_step.to(device=existing_step.device, dtype=existing_step.dtype)
        return cloned_step
    return copy.deepcopy(step)


def _refresh_empty_optimizer_step(param_state, param_name, canonical_step, canonical_step_value):
    existing_step = param_state.get('step')
    if existing_step is None or _optimizer_step_value(existing_step, param_name) != canonical_step_value:
        param_state['step'] = _clone_optimizer_step(canonical_step, existing_step)
        return 1
    return 0


# 3. Complete empty optimizer shards for DCP save/load.


def _get_fsdp_model_parameter(model, param_name):
    model_param_name = param_name[len('module.'):] if param_name.startswith('module.') else param_name
    num_experts = getattr(getattr(model, 'config', None), 'num_moe_experts', None)
    if num_experts:
        from megatron.core.transformer.fsdp_dtensor_checkpoint import expert_param_local_key
        model_param_name = expert_param_local_key(model_param_name, num_experts)
    return model.get_parameter(model_param_name)


def _build_empty_optimizer_state(state_template, template_param, dist_param, param_name):
    missing_state = {}
    for state_key, template_value in state_template.items():
        if not isinstance(template_value, torch.Tensor):
            missing_state[state_key] = copy.deepcopy(template_value)
            continue

        if state_key == 'step':
            missing_state[state_key] = template_value.detach().clone()
            continue

        if tuple(template_value.shape) != tuple(template_param.shape):
            raise RuntimeError(
                f'Cannot infer the empty Megatron-FSDP optimizer state `{state_key}` for `{param_name}`: '
                f'template state shape {tuple(template_value.shape)} does not match its parameter shape '
                f'{tuple(template_param.shape)}.')
        missing_state[state_key] = torch.zeros_like(dist_param, dtype=template_value.dtype)
    return missing_state


def complete_npu_fsdp_dtensor_optimizer_state(state_dict, model) -> None:
    """Complete and refresh empty AdamW shards for consistent DCP state.

    MCore skips the dummy optimizer step for empty local DTensor shards. Native
    ``torch.optim.AdamW`` therefore has no state entry for those parameters,
    while Torch DCP requires a consistent optimizer key set across DP ranks.

    An empty shard created during checkpoint load remains in the live optimizer
    state afterwards, but AdamW never visits it and therefore never increments
    its scalar ``step``. Refresh those placeholders from a non-empty local
    shard before every save so a save-resume-save chain does not persist a
    stale optimizer step for the corresponding global parameter.

    This assumes synchronized updates across all optimized parameters and a
    uniform AdamW state layout/dtype. Conditional updates with unequal parameter
    steps or heterogeneous optimizer state layouts are not supported here.
    """
    optimizer_state_dict = state_dict.get('optimizer')
    if not optimizer_state_dict:
        return
    optimizer_state = optimizer_state_dict.get('state', {})
    param_to_group_meta = optimizer_state_dict.get('param_to_group_meta', {})
    if not param_to_group_meta:
        return
    if not optimizer_state:
        raise RuntimeError('Cannot infer Megatron-FSDP optimizer state fields from an empty local state dict.')

    parameter_info = {}
    active_steps = []
    template_param_name = None
    for param_name in param_to_group_meta:
        dist_param = _get_fsdp_model_parameter(model, param_name)
        local_param = dist_param.to_local() if hasattr(dist_param, 'to_local') else dist_param
        parameter_info[param_name] = (dist_param, local_param)
        param_state = optimizer_state.get(param_name)
        if local_param.numel() == 0 or not param_state:
            continue
        if 'step' not in param_state:
            raise RuntimeError(f'Megatron-FSDP AdamW state for `{param_name}` is missing scalar `step`.')
        active_steps.append((param_name, param_state['step'], _optimizer_step_value(param_state['step'], param_name)))
        if template_param_name is None:
            template_param_name = param_name

    if template_param_name is None:
        raise RuntimeError('Cannot match a non-empty Megatron-FSDP optimizer state template to a local parameter.')

    canonical_step_name, canonical_step, canonical_step_value = active_steps[0]
    inconsistent_steps = [(name, value) for name, _step, value in active_steps if value != canonical_step_value]
    if inconsistent_steps:
        details = ', '.join(f'{name}={value}' for name, value in inconsistent_steps[:8])
        raise RuntimeError(f'Inconsistent Megatron-FSDP optimizer steps on non-empty local shards: '
                           f'{canonical_step_name}={canonical_step_value}; {details}.')

    state_template = optimizer_state[template_param_name]
    template_param = _get_fsdp_model_parameter(model, template_param_name)
    completed_state = {}
    added_count = 0
    refreshed_count = 0
    for param_name in param_to_group_meta:
        dist_param, local_param = parameter_info[param_name]
        if optimizer_state.get(param_name):
            param_state = optimizer_state[param_name]
            if local_param.numel() == 0:
                refreshed_count += _refresh_empty_optimizer_step(param_state, param_name, canonical_step,
                                                                 canonical_step_value)
            completed_state[param_name] = param_state
            continue

        if local_param.numel() != 0:
            raise RuntimeError(
                f'Megatron-FSDP optimizer state is missing or empty for `{param_name}`, but its local parameter shard '
                f'is not empty (numel={local_param.numel()}).')
        completed_state[param_name] = _build_empty_optimizer_state(
            state_template,
            template_param,
            dist_param,
            param_name,
        )
        added_count += 1

    completed_state.update({key: value for key, value in optimizer_state.items() if key not in completed_state})
    optimizer_state_dict['state'] = completed_state
    if added_count or refreshed_count:
        logger.info(
            'Completed NPU Megatron-FSDP optimizer state: added %d empty local shards and refreshed %d stale steps.',
            added_count,
            refreshed_count,
        )


__all__ = ['NPUFSDPAdamW', 'complete_npu_fsdp_dtensor_optimizer_state', 'patch_megatron_fsdp_optimizer']
