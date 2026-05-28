# Copyright (c) ModelScope Contributors. All rights reserved.
"""vLLM-Ascend group factory for colocated Megatron GRPO on NPU.

This module builds vLLM group specs, validates that every rank sees the same
plan, then either reuses a registry-approved Megatron HCCL group or precreates a
new vLLM HCCL/Gloo group.  Runtime patches should only look up the cache built
here; they should not call ``new_group`` themselves.
"""
from __future__ import annotations

import json
import torch
from dataclasses import dataclass
from typing import Any, Optional

from swift.model.npu_patch.vllm_ascend_group_registry import (
    _PREFERRED_MEGATRON_AXES_BY_VLLM_GROUP, _lookup_megatron_hccl_group_record, _normalize_backend_name,
    _select_megatron_reuse_group, _stable_int64_hash, create_npu_process_group)
from swift.utils.logger import get_logger

logger = get_logger()

_SWIFT_VLLM_GROUP_CACHE: dict[tuple[str, str, str, tuple[int, ...]], Any] = {}
_SWIFT_VLLM_GROUP_FACTORY_SIGNATURES: set[tuple[str, ...]] = set()

@dataclass(frozen=True)
class _VLLMGroupSpec:
    name: str
    index: int
    backend: str
    kind: str
    ranks: tuple[int, ...]
    required: bool = True


@dataclass(frozen=True)
class _VLLMGroupExecution:
    spec: _VLLMGroupSpec
    action: str
    reuse_source: Optional[str] = None


def _vllm_group_cache_key(spec: _VLLMGroupSpec) -> tuple[str, str, str, tuple[int, ...]]:
    return (spec.kind, spec.backend, spec.name, spec.ranks)


def _put_vllm_group_cache(spec: _VLLMGroupSpec, group) -> None:
    _SWIFT_VLLM_GROUP_CACHE[_vllm_group_cache_key(spec)] = group


def _lookup_vllm_group_cache(kind: str, group_name: str, ranks, backend):
    backend = _normalize_backend_name(backend)
    ranks_tuple = tuple(int(rank) for rank in ranks)
    return _SWIFT_VLLM_GROUP_CACHE.get((kind, backend, group_name, ranks_tuple))


def _lookup_vllm_device_group_cache(group_name: str, ranks, backend):
    return _lookup_vllm_group_cache('device', group_name, ranks, backend)


def _lookup_vllm_cpu_group_cache(group_name: str, ranks):
    return _lookup_vllm_group_cache('cpu', group_name, ranks, 'gloo')


def _cache_key_summary() -> list[tuple[str, str, str, tuple[int, ...]]]:
    return sorted(_SWIFT_VLLM_GROUP_CACHE)


def _validate_global_hash(name: str, payload) -> None:
    import torch.distributed as dist

    device = torch.device('npu', torch.npu.current_device())
    local_hash = _stable_int64_hash(payload)
    local_min = torch.tensor([local_hash], dtype=torch.int64, device=device)
    local_max = local_min.clone()
    dist.all_reduce(local_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
    if int(local_min.cpu().item()) != int(local_max.cpu().item()):
        raise RuntimeError(
            f'vLLM {name} mismatch across ranks before group creation: rank={dist.get_rank()}, '
            f'local_hash={local_hash}, min_hash={int(local_min.cpu().item())}, '
            f'max_hash={int(local_max.cpu().item())}, payload={payload}.')


def _serialize_spec(spec: _VLLMGroupSpec):
    return {
        'name': spec.name,
        'index': spec.index,
        'backend': spec.backend,
        'kind': spec.kind,
        'ranks': list(spec.ranks),
        'required': spec.required,
    }


def _serialize_specs(specs: list[_VLLMGroupSpec]):
    return [_serialize_spec(spec) for spec in specs]


def _serialize_execution_plan(executions: list[_VLLMGroupExecution]):
    return [
        {
            **_serialize_spec(execution.spec),
            'action': execution.action,
            'reuse_source': execution.reuse_source,
        } for execution in executions
    ]


def _sync_default_world_on_npu() -> None:
    import torch.distributed as dist

    marker = torch.ones((), dtype=torch.int32, device=torch.device('npu', torch.npu.current_device()))
    dist.all_reduce(marker)


def _warmup_vllm_device_group(spec: _VLLMGroupSpec, group) -> None:
    import torch.distributed as dist

    if dist.get_rank() not in spec.ranks:
        return
    device = torch.device('npu', torch.npu.current_device())
    rank_value = torch.tensor([dist.get_rank()], dtype=torch.int32, device=device)
    reduced = rank_value.clone()
    dist.all_reduce(reduced, group=group)
    expected_sum = sum(spec.ranks)
    if int(reduced.cpu().item()) != expected_sum:
        raise RuntimeError(
            f'vLLM device group warmup all_reduce mismatch: name={spec.name}, rank={dist.get_rank()}, '
            f'ranks={spec.ranks}, got={int(reduced.cpu().item())}, expected={expected_sum}.')

    gathered = torch.empty((len(spec.ranks), ), dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(gathered, rank_value, group=group)
    expected = torch.tensor(list(spec.ranks), dtype=torch.int32)
    if not torch.equal(gathered.cpu(), expected):
        raise RuntimeError(
            f'vLLM device group warmup all_gather mismatch: name={spec.name}, rank={dist.get_rank()}, '
            f'ranks={spec.ranks}, got={gathered.cpu().tolist()}, expected={expected.tolist()}.')


def _warmup_vllm_cpu_group(spec: _VLLMGroupSpec, group) -> None:
    import torch.distributed as dist

    if dist.get_rank() not in spec.ranks:
        return
    rank_value = torch.tensor([dist.get_rank()], dtype=torch.int32, device='cpu')
    reduced = rank_value.clone()
    dist.all_reduce(reduced, group=group)
    expected_sum = sum(spec.ranks)
    if int(reduced.item()) != expected_sum:
        raise RuntimeError(
            f'vLLM CPU group warmup all_reduce mismatch: name={spec.name}, rank={dist.get_rank()}, '
            f'ranks={spec.ranks}, got={int(reduced.item())}, expected={expected_sum}.')

    gathered = [torch.empty_like(rank_value) for _ in spec.ranks]
    dist.all_gather(gathered, rank_value, group=group)
    got = [int(t.item()) for t in gathered]
    expected = list(spec.ranks)
    if got != expected:
        raise RuntimeError(
            f'vLLM CPU group warmup all_gather mismatch: name={spec.name}, rank={dist.get_rank()}, '
            f'ranks={spec.ranks}, got={got}, expected={expected}.')


def _warmup_vllm_group(spec: _VLLMGroupSpec, group) -> None:
    if spec.kind == 'device':
        _warmup_vllm_device_group(spec, group)
    elif spec.kind == 'cpu':
        _warmup_vllm_cpu_group(spec, group)
    else:
        raise RuntimeError(f'Unsupported vLLM group kind for warmup: spec={spec}.')


def _get_vllm_parallel_kwarg(args, name: str, default: int) -> int:
    engine_kwargs = getattr(args, 'vllm_engine_kwargs', None) or {}
    if isinstance(engine_kwargs, str):
        engine_kwargs = json.loads(engine_kwargs)
    return int(engine_kwargs.get(name, default) or default)


def _build_vllm_device_group_specs(args, backend: str) -> list[_VLLMGroupSpec]:
    import torch.distributed as dist

    world_size = dist.get_world_size()
    data_parallel_size = _get_vllm_data_parallel_size_from_args(args)
    tensor_parallel_size = int(getattr(args, 'vllm_tensor_parallel_size', 1) or 1)
    pipeline_parallel_size = _get_vllm_parallel_kwarg(args, 'pipeline_parallel_size', 1)
    prefill_context_parallel_size = _get_vllm_parallel_kwarg(args, 'prefill_context_parallel_size', 1)
    group_unit = data_parallel_size * pipeline_parallel_size * prefill_context_parallel_size * tensor_parallel_size
    if world_size % group_unit != 0:
        raise RuntimeError(
            f'Cannot build vLLM device group specs: world_size={world_size}, dp={data_parallel_size}, '
            f'pp={pipeline_parallel_size}, pcp={prefill_context_parallel_size}, tp={tensor_parallel_size}.')

    all_ranks = torch.arange(world_size).reshape(
        -1,
        data_parallel_size,
        pipeline_parallel_size,
        prefill_context_parallel_size,
        tensor_parallel_size,
    )

    specs: list[_VLLMGroupSpec] = []

    def add_specs(name: str, group_ranks) -> None:
        for index, ranks in enumerate(group_ranks):
            specs.append(
                _VLLMGroupSpec(
                    name=name,
                    index=index,
                    backend=backend,
                    kind='device',
                    ranks=tuple(int(rank) for rank in ranks),
                ))

    add_specs('world', [list(range(world_size))])
    add_specs(
        'pcp',
        [x.tolist() for x in all_ranks.transpose(3, 4).reshape(-1, prefill_context_parallel_size).unbind(0)])
    add_specs('pp', [x.tolist() for x in all_ranks.transpose(2, 4).reshape(-1, pipeline_parallel_size).unbind(0)])
    # vLLM still constructs a DP GroupCoordinator when data_parallel_size == 1.
    # In cache-only mode those singleton DP groups must be present in the
    # factory cache; otherwise GroupCoordinator correctly fails with cache miss.
    add_specs('dp', [x.tolist() for x in all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)])

    ep_group_ranks = [
        x.tolist() for x in all_ranks.transpose(1, 2).reshape(
            -1, data_parallel_size * prefill_context_parallel_size * tensor_parallel_size).unbind(0)
    ]
    if getattr(getattr(args, 'model_info', None), 'is_moe_model', False):
        add_specs('ep', ep_group_ranks)
    add_specs('mc2', ep_group_ranks)

    group_order = {
        'world': 0,
        'dp': 2,
        'pp': 3,
        'pcp': 4,
        'ep': 5,
        'mc2': 6,
    }
    return sorted(specs, key=lambda spec: (group_order.get(spec.name, 100), spec.index, spec.ranks))


def _build_vllm_cpu_group_specs(args) -> list[_VLLMGroupSpec]:
    """Build Gloo CPU specs matching the NPU-only vLLM groups covered here.

    TP/DCP message-queue groups still stay on the upstream vLLM path or SWIFT's
    rollout TP Gloo control helper.  This P4 step only normalizes the CPU group
    counterpart for groups whose HCCL device group is already handled by this
    runtime patch.
    """
    specs = _build_vllm_device_group_specs(args, 'gloo')
    return [
        _VLLMGroupSpec(
            name=spec.name,
            index=spec.index,
            backend='gloo',
            kind='cpu',
            ranks=spec.ranks,
            required=spec.required,
        ) for spec in specs
    ]


def _build_vllm_group_execution_plan(specs: list[_VLLMGroupSpec]) -> list[_VLLMGroupExecution]:
    executions: list[_VLLMGroupExecution] = []
    for spec in specs:
        reuse_selected, reuse_record = _select_megatron_reuse_group(
            spec.name, spec.ranks, spec.backend, phase='vllm_preinit') if spec.kind == 'device' else (False, None)
        if reuse_selected:
            executions.append(
                _VLLMGroupExecution(spec=spec, action='reuse', reuse_source='megatron_exact_rank_order'))
        else:
            executions.append(_VLLMGroupExecution(spec=spec, action='precreate'))
    return executions


def _get_reused_megatron_group_for_spec(spec: _VLLMGroupSpec):
    import torch.distributed as dist

    if dist.get_rank() not in spec.ranks:
        return None
    record = _lookup_megatron_hccl_group_record(spec.ranks, _PREFERRED_MEGATRON_AXES_BY_VLLM_GROUP.get(spec.name))
    if record is None:
        raise RuntimeError(f'vLLM group execution plan selected Megatron reuse but no group was found: spec={spec}.')
    return record.group


def prepare_vllm_ascend_device_groups_before_megatron(args) -> None:
    """Pre-create/cache vLLM NPU-only groups before LLMEngine init.

    This group factory deliberately covers only the NPU-only groups handled by
    ``GroupCoordinatorPatch`` in this module.  Device HCCL groups and matching
    Gloo CPU groups are prepared together.  TP/DCP message-queue groups still
    use upstream CPU/control behavior or SWIFT's dedicated rollout control
    helper.
    """
    if not getattr(args, 'use_vllm', False) or getattr(args, 'vllm_mode', None) != 'colocate':
        return

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return

    backend = _normalize_backend_name(dist.get_backend())
    specs = _build_vllm_device_group_specs(args, backend)
    specs.extend(_build_vllm_cpu_group_specs(args))
    _validate_global_hash('group specs', _serialize_specs(specs))
    executions = _build_vllm_group_execution_plan(specs)
    serialized_execution_plan = _serialize_execution_plan(executions)
    _validate_global_hash('group execution plan', serialized_execution_plan)
    signature = tuple(json.dumps(item, sort_keys=True) for item in serialized_execution_plan)
    if signature in _SWIFT_VLLM_GROUP_FACTORY_SIGNATURES:
        return

    rank = dist.get_rank()
    for execution in executions:
        spec = execution.spec
        if execution.action == 'reuse':
            if rank in spec.ranks:
                group = _get_reused_megatron_group_for_spec(spec)
                _put_vllm_group_cache(spec, group)
        elif execution.action == 'precreate':
            if spec.backend == 'gloo':
                _clear_default_pg_bound_device_id_for_gloo()
            group = create_npu_process_group(
                list(spec.ranks),
                backend=spec.backend,
                kind=spec.kind,
                group_name=spec.name,
                source='vllm_factory',
                phase='vllm_preinit')
            if rank in spec.ranks:
                _put_vllm_group_cache(spec, group)
        else:
            raise RuntimeError(f'Unsupported vLLM group execution action: {execution.action}, spec={spec}.')

        _sync_default_world_on_npu()
        if rank in spec.ranks:
            cached_group = _SWIFT_VLLM_GROUP_CACHE.get(_vllm_group_cache_key(spec))
            if cached_group is None:
                raise RuntimeError(f'vLLM group was not cached after {execution.action}: spec={spec}.')
            _warmup_vllm_group(spec, cached_group)
        _sync_default_world_on_npu()

    _SWIFT_VLLM_GROUP_FACTORY_SIGNATURES.add(signature)
    logger.warning_once(f'Prepared vLLM NPU group factory cache before vLLM init: plan={serialized_execution_plan}')




def _canonical_group_ranks(group_ranks) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(rank) for rank in ranks) for ranks in group_ranks)


def _clear_default_pg_bound_device_id_for_gloo() -> None:
    import torch.distributed as dist

    try:
        default_pg = dist.distributed_c10d._get_default_group()
    except Exception:
        return
    if getattr(default_pg, 'bound_device_id', None) is not None:
        default_pg.bound_device_id = None



def _get_vllm_data_parallel_size_from_args(args) -> int:
    engine_kwargs = getattr(args, 'vllm_engine_kwargs', None) or {}
    if isinstance(engine_kwargs, str):
        engine_kwargs = json.loads(engine_kwargs)
    return int(engine_kwargs.get('data_parallel_size', 1) or 1)


def prepare_vllm_ascend_dp_groups_before_megatron(args) -> None:
    """Prepare vLLM HCCL/Gloo groups before vLLM builds its topology.

    The public name is kept for existing callers, but the implementation now
    delegates to the full group factory instead of the older DP-only path.
    """
    prepare_vllm_ascend_device_groups_before_megatron(args)


__all__ = [
    'prepare_vllm_ascend_device_groups_before_megatron',
    'prepare_vllm_ascend_dp_groups_before_megatron',
]
