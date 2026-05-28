# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import torch
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from packaging import version
from typing import Any, Optional

from swift.utils.logger import get_logger

logger = get_logger()

_SWIFT_VLLM_DP_GROUP_SPECS: set[tuple[tuple[int, ...], ...]] = set()
_SWIFT_VLLM_TP_GLOO_GROUPS: dict[tuple[tuple[int, ...], ...], Any] = {}
_SWIFT_MEGATRON_HCCL_GROUPS: dict[tuple[str, str, tuple[int, ...]], list['_MegatronGroupRecord']] = {}
_SWIFT_VLLM_GROUP_CACHE: dict[tuple[str, str, str, tuple[int, ...]], Any] = {}
_SWIFT_VLLM_GROUP_FACTORY_SIGNATURES: set[tuple[str, ...]] = set()


@dataclass(frozen=True)
class _MegatronGroupRecord:
    backend: str
    kind: str
    axis: str
    ranks: tuple[int, ...]
    group: Any
    source: str


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


def create_npu_process_group(ranks=None, backend=None, *, kind: str, group_name: str, source: str, phase: str, **kwargs):
    """Create a torch distributed group through SWIFT's NPU patch layer."""
    import torch.distributed as dist

    return dist.new_group(ranks=ranks, backend=backend, **kwargs)


def _normalize_backend_name(backend) -> str:
    backend_name = str(backend or '').lower()
    if 'hccl' in backend_name:
        return 'hccl'
    if 'gloo' in backend_name:
        return 'gloo'
    return backend_name or 'default'


def _get_process_group_backend(group) -> str:
    import torch.distributed as dist

    try:
        return _normalize_backend_name(dist.get_backend(group))
    except TypeError:
        try:
            return _normalize_backend_name(dist.get_backend(group=group))
        except Exception:
            return _normalize_backend_name(dist.get_backend())
    except Exception:
        return _normalize_backend_name(dist.get_backend())


def _get_process_group_ranks(group) -> tuple[int, ...]:
    import torch.distributed as dist

    try:
        return tuple(int(rank) for rank in dist.get_process_group_ranks(group))
    except Exception:
        if group is getattr(dist.group, 'WORLD', None):
            return tuple(range(dist.get_world_size()))
        raise


def _register_megatron_hccl_group(axis: str, group, source: str) -> None:
    if group is None:
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return

    backend = _get_process_group_backend(group)
    if backend != 'hccl':
        return
    try:
        ranks = _get_process_group_ranks(group)
    except Exception as e:
        logger.warning_once(f'Skipped registering Megatron {axis} HCCL group because ranks cannot be inspected: {e!r}')
        return

    key = (backend, 'device', ranks)
    records = _SWIFT_MEGATRON_HCCL_GROUPS.setdefault(key, [])
    if any(record.axis == axis and record.source == source for record in records):
        return
    records.append(
        _MegatronGroupRecord(backend=backend, kind='device', axis=axis, ranks=ranks, group=group, source=source))


def _call_megatron_group_getter(getter, **kwargs):
    try:
        return getter(check_initialized=False, **kwargs)
    except TypeError:
        try:
            return getter(**kwargs)
        except (TypeError, AssertionError, RuntimeError, ValueError):
            return None
    except (AssertionError, RuntimeError, ValueError):
        return None


def _register_megatron_group_from_getter(parallel_state, axis: str, getter_name: str, **kwargs) -> None:
    getter = getattr(parallel_state, getter_name, None)
    if getter is None:
        return
    group = _call_megatron_group_getter(getter, **kwargs)
    _register_megatron_hccl_group(axis, group, 'megatron_mpu')


def register_megatron_hccl_groups_for_vllm(args) -> None:
    """Register Megatron HCCL groups that vLLM may safely reuse.

    The registry is exact-rank-order based.  It registers the current rank's
    Megatron groups that can be inspected safely.  Reuse is still decided later
    per vLLM group spec and only succeeds when all member ranks have an exact
    rank-order match; otherwise the vLLM group factory precreates a fresh group.
    """
    if not getattr(args, 'use_vllm', False) or getattr(args, 'vllm_mode', None) != 'colocate':
        return

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return

    _register_megatron_hccl_group('world', dist.group.WORLD, 'default_world')

    try:
        from megatron.core import parallel_state
    except ImportError:
        return
    group_getters = [
        ('tp', 'get_tensor_model_parallel_group', {}),
        ('pp', 'get_pipeline_model_parallel_group', {}),
        ('cp', 'get_context_parallel_group', {}),
        ('dp', 'get_data_parallel_group', {}),
        ('dp_cp', 'get_data_parallel_group', {'with_context_parallel': True}),
        ('partial_dp', 'get_data_parallel_group', {'partial_data_parallel': True}),
        ('partial_dp_cp', 'get_data_parallel_group', {
            'with_context_parallel': True,
            'partial_data_parallel': True,
        }),
        ('ep', 'get_expert_model_parallel_group', {}),
        ('etp', 'get_expert_tensor_parallel_group', {}),
        ('edp', 'get_expert_data_parallel_group', {}),
        ('partial_edp', 'get_expert_data_parallel_group', {'partial_expert_data_parallel': True}),
        ('etp_ep', 'get_expert_tensor_and_model_parallel_group', {}),
    ]
    for axis, getter_name, kwargs in group_getters:
        _register_megatron_group_from_getter(parallel_state, axis, getter_name, **kwargs)


_PREFERRED_MEGATRON_AXES_BY_VLLM_GROUP = {
    'world': ('world', ),
    'tp': ('tp', ),
    'pp': ('pp', ),
    'pcp': ('cp', ),
    'dp': ('dp', 'dp_cp', 'partial_dp', 'partial_dp_cp'),
    'ep': ('ep', 'etp_ep'),
    'mc2': ('ep', 'etp_ep', 'edp', 'partial_edp'),
}


def _lookup_megatron_hccl_group_record(ranks: tuple[int, ...],
                                       preferred_axes: tuple[str, ...] | None = None) -> Optional[_MegatronGroupRecord]:
    records = _SWIFT_MEGATRON_HCCL_GROUPS.get(('hccl', 'device', ranks), [])
    if not records:
        return None
    for axis in preferred_axes or ():
        for record in records:
            if record.axis == axis:
                return record
    return records[0]


def _megatron_group_record_token(record: _MegatronGroupRecord) -> dict[str, str]:
    return {'axis': record.axis, 'source': record.source}


def _validate_reuse_record_consensus(group_name: str, ranks: tuple[int, ...],
                                     record: Optional[_MegatronGroupRecord]) -> bool:
    """Ensure all member ranks chose the same Megatron group provenance.

    A single rank tuple can appear in more than one Megatron ProcessGroup.  The
    collectives are only safe if all member ranks reuse the same communicator
    provenance, not just any group with the same ranks.  Non-member ranks vote
    neutral; if member ranks disagree, every rank falls back to precreate.
    """
    import torch.distributed as dist

    rank = dist.get_rank()
    if rank in ranks and record is not None:
        token_hash = _stable_int64_hash(_megatron_group_record_token(record))
        min_hash = torch.tensor([token_hash], dtype=torch.int64, device=torch.device('npu', torch.npu.current_device()))
        max_hash = min_hash.clone()
    else:
        min_hash = torch.tensor([2**62 - 1], dtype=torch.int64, device=torch.device('npu', torch.npu.current_device()))
        max_hash = torch.tensor([-1], dtype=torch.int64, device=torch.device('npu', torch.npu.current_device()))

    dist.all_reduce(min_hash, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_hash, op=dist.ReduceOp.MAX)
    if int(min_hash.cpu().item()) == int(max_hash.cpu().item()):
        return True

    if rank in ranks:
        logger.warning_once(
            f'Megatron group reuse provenance differs across member ranks; fall back to vLLM precreate: '
            f'group_name={group_name}, ranks={ranks}, local_record={record}.')
    return False


def _select_megatron_reuse_group(group_name: str,
                                 ranks,
                                 backend,
                                 phase: str = 'vllm_groupcoordinator') -> tuple[bool, Optional[_MegatronGroupRecord]]:
    """Return a globally consistent Megatron-group reuse decision.

    The decision is intentionally all-rank: non-member ranks vote neutral, and
    every member rank must have an exact-order Megatron group record.  If any
    member cannot reuse, all ranks fall back to the normal vLLM group creation
    path for this spec.  This prevents rank-local registry differences from
    splitting execution between reuse and ``new_group``.
    """
    if _normalize_backend_name(backend) != 'hccl':
        return False, None

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return False, None

    ranks_tuple = tuple(int(rank) for rank in ranks)
    rank = dist.get_rank()
    record = _lookup_megatron_hccl_group_record(ranks_tuple, _PREFERRED_MEGATRON_AXES_BY_VLLM_GROUP.get(group_name))
    local_reuse_ok = rank not in ranks_tuple or record is not None
    device = torch.device('npu', torch.npu.current_device())
    flag = torch.tensor([1 if local_reuse_ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    can_reuse = bool(int(flag.cpu().item()))
    if not can_reuse:
        return False, None
    if rank in ranks_tuple and record is None:
        raise RuntimeError(
            f'Megatron group reuse decision is inconsistent: rank={rank}, group_name={group_name}, '
            f'ranks={ranks_tuple}.')
    if not _validate_reuse_record_consensus(group_name, ranks_tuple, record):
        return False, None
    return True, record


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


def _stable_int64_hash(payload) -> int:
    text = json.dumps(payload, sort_keys=True)
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)


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


def _get_package_version(package_name: str) -> Optional[version.Version]:
    try:
        return version.parse(metadata.version(package_name))
    except (metadata.PackageNotFoundError, version.InvalidVersion):
        return None


def patch_vllm_ascend_runtime() -> None:
    """Apply SWIFT runtime compatibility patches for vLLM-Ascend on NPU.

    The caller is responsible for guarding this function by device type.  This
    module intentionally imports vLLM/vLLM-Ascend modules lazily so CUDA/GPU
    paths do not enter Ascend-only code.
    """
    # Read versions for diagnostics and future version-gated branches.  The
    # concrete patches below still use symbol checks because local source builds
    # may report dev versions while carrying fixes from adjacent releases.
    vllm_version = _get_package_version('vllm')
    vllm_ascend_version = _get_package_version('vllm-ascend')
    logger.debug('Applying vLLM-Ascend runtime patches: vllm=%s, vllm-ascend=%s', vllm_version, vllm_ascend_version)

    _patch_flash_attn_optional_import()
    _patch_vllm_ascend_external_launcher_groups()
    _patch_vllm_ascend_moe_comm_lazy_init()
    _patch_vllm_ascend_mem_get_info()
    _patch_vllm_ascend_memory_profiling()
    _patch_vllm_ascend_device_op_nonquant_routing()
    _patch_vllm_ascend_unquant_moe_layout()
    patch_vllm_ascend_colocate_runtime()


def patch_vllm_ascend_colocate_runtime() -> None:
    """Apply vLLM-Ascend patches needed by colocated training."""
    _patch_vllm_ascend_colocate_memory_profiling()


def _patch_flash_attn_optional_import() -> None:
    module = sys.modules.get('flash_attn')
    if module is None or hasattr(module, '__path__'):
        return
    for module_name in list(sys.modules):
        if module_name == 'flash_attn' or module_name.startswith('flash_attn.'):
            sys.modules.pop(module_name, None)


def _patch_vllm_ascend_moe_comm_lazy_init() -> None:
    try:
        from vllm_ascend.ascend_forward_context import MoECommType
        from vllm_ascend.ops.fused_moe import moe_comm_method
    except (ImportError, AttributeError):
        return
    if getattr(moe_comm_method, '_swift_lazy_moe_comm_patched', False):
        return

    required_classes = {
        MoECommType.ALLTOALL: 'AlltoAllCommImpl',
        MoECommType.ALLGATHER: 'AllGatherCommImpl',
        MoECommType.MC2: 'MC2CommImpl',
        MoECommType.FUSED_MC2: 'FusedMC2CommImpl',
    }
    if not all(hasattr(moe_comm_method, cls_name) for cls_name in required_classes.values()):
        return

    moe_comm_method._MoECommMethods = {}
    moe_comm_method._SwiftMoECommConfig = None

    def get_moe_comm_method(moe_comm_type):
        if moe_comm_type is None:
            return None
        moe_comm = moe_comm_method._MoECommMethods.get(moe_comm_type)
        if moe_comm is not None:
            return moe_comm
        assert moe_comm_method._SwiftMoECommConfig is not None, 'MoE communication method is not set up'
        moe_comm_cls = getattr(moe_comm_method, required_classes[moe_comm_type])
        moe_comm = moe_comm_cls(moe_comm_method._SwiftMoECommConfig)
        moe_comm_method._MoECommMethods[moe_comm_type] = moe_comm
        return moe_comm

    def setup_moe_comm_method(moe_config):
        moe_comm_method._SwiftMoECommConfig = moe_config
        moe_comm_method._MoECommMethods.clear()

    moe_comm_method.get_moe_comm_method = get_moe_comm_method
    moe_comm_method.setup_moe_comm_method = setup_moe_comm_method
    fused_moe = sys.modules.get('vllm_ascend.ops.fused_moe.fused_moe')
    if fused_moe is not None:
        fused_moe.setup_moe_comm_method = setup_moe_comm_method
    moe_comm_method._swift_lazy_moe_comm_patched = True


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


def get_or_create_vllm_tp_gloo_group(tensor_parallel_size: int):
    """Create a Gloo control-plane group matching vLLM TP ranks.

    vLLM-Ascend 0.18 external-launcher TP groups are NPU/HCCL device groups
    in this colocated Megatron path. They are correct for tensor communication,
    but PyTorch object collectives such as ``all_gather_object`` first exchange
    Python-object sizes as tiny control metadata. On NPU/HCCL that metadata can
    be corrupted, which was observed as ``all_gather_object`` trying to resize a
    tensor to more than 1EB.

    Keep the device group unchanged for vLLM tensor work and create a matching
    Gloo group only for SWIFT rollout-side object collectives.
    """
    from transformers.utils import is_torch_npu_available
    import torch.distributed as dist

    if not is_torch_npu_available() or tensor_parallel_size <= 1 or not dist.is_initialized():
        return None

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size % tensor_parallel_size != 0:
        raise RuntimeError(
            f'Cannot build vLLM TP Gloo control groups: world_size={world_size}, tp={tensor_parallel_size}.')

    group_ranks = [list(range(start, start + tensor_parallel_size)) for start in range(0, world_size, tensor_parallel_size)]
    spec = _canonical_group_ranks(group_ranks)
    if spec not in _SWIFT_VLLM_TP_GLOO_GROUPS:
        _clear_default_pg_bound_device_id_for_gloo()
        own_group = None
        for ranks in group_ranks:
            group = create_npu_process_group(
                ranks,
                backend='gloo',
                kind='control',
                group_name='tp',
                source='vllm_tp_control',
                phase='rollout')
            if rank in ranks:
                own_group = group
        if own_group is None:
            raise RuntimeError(f'Rank {rank} is not included in vLLM TP group spec: {group_ranks}.')
        _SWIFT_VLLM_TP_GLOO_GROUPS[spec] = own_group
        logger.warning_once(f'Created vLLM TP Gloo control groups for NPU rollout object collectives: ranks={spec}')

    return _SWIFT_VLLM_TP_GLOO_GROUPS[spec]


def _group_spec_hash(group_name: str, group_ranks) -> int:
    payload = json.dumps({
        'group_name': group_name,
        'group_ranks': _canonical_group_ranks(group_ranks),
    },
                         sort_keys=True)
    # Keep the hash inside signed int64 so HCCL can compare it through a tensor.
    return int(hashlib.sha256(payload.encode()).hexdigest()[:15], 16)


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


def _patch_vllm_ascend_external_launcher_groups() -> None:
    """Create NPU-only vLLM groups when Megatron already owns torch.distributed.

    In colocated Megatron GRPO, vLLM runs in ``external_launcher`` mode and
    reuses Megatron's already-initialized default HCCL world.  vLLM-Ascend 0.18
    then creates an extra Gloo CPU group for each ``GroupCoordinator``.  On this
    stack, creating those Gloo groups inside the Megatron process tree can hang
    before vLLM finishes initialization.

    Tensor-parallel and decode-context groups use vLLM's shared-memory message
    queue, so their upstream Gloo group is left intact.  For the other NPU-only
    groups, the group factory precreates/caches both HCCL device groups and
    matching Gloo CPU groups before vLLM initialization.  GroupCoordinator then
    only looks up cached groups and fails loudly on cache miss.
    """
    try:
        import torch.distributed as dist
        from vllm.config import parallel as parallel_config_module
        from vllm.distributed import parallel_state
        from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator
        from vllm_ascend.patch.worker import patch_distributed
        from vllm_ascend.utils import should_skip_allreduce_across_dp_group
        from vllm_ascend.worker import model_runner_v1
    except (ImportError, AttributeError, RuntimeError):
        return

    group_cls = getattr(patch_distributed, 'GroupCoordinatorPatch', None)
    if group_cls is not None and not getattr(group_cls, '_swift_external_launcher_group_patched', False):
        origin_init = group_cls.__init__

        def _should_use_npu_only_group(use_device_communicator, use_message_queue_broadcaster, group_name) -> bool:
            if use_message_queue_broadcaster:
                return False
            return group_name == 'world' or use_device_communicator

        def _validate_dp_group_spec(group_name, group_ranks) -> None:
            if group_name != 'dp':
                return
            spec = _canonical_group_ranks(group_ranks)
            spec_hash = _group_spec_hash(group_name, group_ranks)
            device = torch.device('npu', torch.npu.current_device())
            local_min = torch.tensor([spec_hash], dtype=torch.int64, device=device)
            local_max = local_min.clone()
            dist.all_reduce(local_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
            if int(local_min.cpu().item()) != int(local_max.cpu().item()):
                raise RuntimeError(
                    f'vLLM DP group spec mismatch across ranks: rank={dist.get_rank()}, group_ranks={group_ranks}, '
                    f'local_hash={spec_hash}, min_hash={int(local_min.cpu().item())}, '
                    f'max_hash={int(local_max.cpu().item())}.')
            if spec not in _SWIFT_VLLM_DP_GROUP_SPECS:
                logger.warning_once(f'Validated vLLM DP group spec on Megatron world group: ranks={spec}, '
                                    f'hash={spec_hash}')
                _SWIFT_VLLM_DP_GROUP_SPECS.add(spec)

        def patched_init(self,
                         group_ranks,
                         local_rank,
                         torch_distributed_backend,
                         use_device_communicator,
                         use_message_queue_broadcaster=False,
                         group_name=None):
            group_name = group_name or 'anonymous'
            if not _should_use_npu_only_group(use_device_communicator, use_message_queue_broadcaster, group_name):
                return origin_init(self, group_ranks, local_rank, torch_distributed_backend, use_device_communicator,
                                   use_message_queue_broadcaster, group_name)

            _validate_dp_group_spec(group_name, group_ranks)

            self.unique_name = parallel_state._get_unique_name(group_name)
            parallel_state._register_group(self)
            self.rank = dist.get_rank()
            self.local_rank = local_rank
            self_device_group = None
            self_cpu_group = None
            for ranks in group_ranks:
                if self.rank not in ranks:
                    continue
                cached_group = _lookup_vllm_device_group_cache(group_name, ranks, torch_distributed_backend)
                if cached_group is None:
                    raise RuntimeError(
                        f'vLLM device group cache miss in cache-only mode: rank={self.rank}, '
                        f'group_name={group_name}, backend={torch_distributed_backend}, ranks={ranks}, '
                        f'cache_keys={_cache_key_summary()}.')
                self_cpu_group = _lookup_vllm_cpu_group_cache(group_name, ranks)
                if self_cpu_group is None:
                    raise RuntimeError(
                        f'vLLM CPU group cache miss in cache-only mode: rank={self.rank}, '
                        f'group_name={group_name}, ranks={ranks}, cache_keys={_cache_key_summary()}.')
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self_device_group = cached_group
                break
            if self_device_group is None:
                raise RuntimeError(
                    f'Rank {self.rank} is not included in cached vLLM device group spec: '
                    f'group_name={group_name}, group_ranks={group_ranks}.')

            self.cpu_group = self_cpu_group
            self.device_group = self_device_group
            self.device = torch.npu.current_device()
            self.use_device_communicator = use_device_communicator
            self.device_communicator = None
            if use_device_communicator and self.world_size > 1:
                self.device_communicator = NPUCommunicator(
                    cpu_group=self.cpu_group,
                    device=self.device,
                    device_group=self.device_group,
                    unique_name=self.unique_name,
                )
            self.mq_broadcaster = None
            self.use_custom_op_call = True
            self.use_cpu_custom_send_recv = False
            logger.warning_once(
                f'Reused vLLM-Ascend factory-cached {group_name} HCCL device group and Gloo CPU group '
                'in colocated Megatron training.')

        group_cls.__init__ = patched_init
        group_cls._swift_external_launcher_group_patched = True

    _patch_vllm_external_launcher_dp_tensor_sync(parallel_config_module, model_runner_v1, should_skip_allreduce_across_dp_group)


def _patch_vllm_external_launcher_dp_tensor_sync(parallel_config_module, model_runner_v1,
                                                 should_skip_allreduce_across_dp_group) -> None:
    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_dp_group

    def _is_swift_device_group(group) -> bool:
        if group is None:
            return False
        try:
            return str(dist.get_backend(group)).lower() == 'hccl'
        except (RuntimeError, ValueError):
            return False

    def _npu_all_reduce_cpu_tensor(tensor: torch.Tensor, group, op=dist.ReduceOp.SUM) -> torch.Tensor:
        if not _is_swift_device_group(group):
            dist.all_reduce(tensor, group=group, op=op)
            return tensor
        device_tensor = tensor.to(torch.device('npu', torch.npu.current_device()))
        dist.all_reduce(device_tensor, group=group, op=op)
        return device_tensor.cpu()

    parallel_config_cls = getattr(parallel_config_module, 'ParallelConfig', None)
    if parallel_config_cls is not None:
        origin_has_unfinished = getattr(parallel_config_cls, 'has_unfinished_dp', None)
        if origin_has_unfinished is not None and not getattr(origin_has_unfinished, '_swift_npu_dp_sync_patched', False):

            @staticmethod
            def has_unfinished_dp(dp_group, has_unfinished: bool) -> bool:
                if not _is_swift_device_group(dp_group):
                    return origin_has_unfinished(dp_group, has_unfinished)
                tensor = torch.tensor([has_unfinished], dtype=torch.int32, device='cpu')
                tensor = _npu_all_reduce_cpu_tensor(tensor, dp_group, op=dist.ReduceOp.MAX)
                return bool(tensor.item())

            has_unfinished_dp._swift_npu_dp_sync_patched = True
            has_unfinished_dp._swift_origin = origin_has_unfinished
            parallel_config_cls.has_unfinished_dp = has_unfinished_dp

        origin_sync_kv = getattr(parallel_config_cls, 'sync_kv_cache_memory_size', None)
        if origin_sync_kv is not None and not getattr(origin_sync_kv, '_swift_npu_dp_sync_patched', False):

            @staticmethod
            def sync_kv_cache_memory_size(dp_group, kv_cache_memory: int) -> int:
                if not _is_swift_device_group(dp_group):
                    return origin_sync_kv(dp_group, kv_cache_memory)
                if kv_cache_memory == -1:
                    kv_cache_memory = torch.iinfo(torch.int64).max
                tensor = torch.tensor([kv_cache_memory], dtype=torch.int64, device='cpu')
                tensor = _npu_all_reduce_cpu_tensor(tensor, dp_group, op=dist.ReduceOp.MIN)
                return int(tensor.item())

            sync_kv_cache_memory_size._swift_npu_dp_sync_patched = True
            sync_kv_cache_memory_size._swift_origin = origin_sync_kv
            parallel_config_cls.sync_kv_cache_memory_size = sync_kv_cache_memory_size

    runner_cls = getattr(model_runner_v1, 'NPUModelRunner', None)
    if runner_cls is None:
        return

    origin_sync_metadata = getattr(runner_cls, '_sync_metadata_across_dp', None)
    if origin_sync_metadata is not None and not getattr(origin_sync_metadata, '_swift_npu_dp_sync_patched', False):

        def _sync_metadata_across_dp(self, num_tokens: int, with_prefill: bool = False,
                                     is_draft_model: bool = False):
            if self.dp_size == 1:
                return num_tokens, None, with_prefill
            if should_skip_allreduce_across_dp_group(self.vllm_config, is_draft_model):
                num_tokens_after_padding = torch.tensor([num_tokens] * self.dp_size, device='cpu', dtype=torch.int32)
                return num_tokens, num_tokens_after_padding, with_prefill

            num_tokens_tensor = torch.tensor(
                [num_tokens if i == self.dp_rank else 0 for i in range(self.dp_size)], dtype=torch.int32, device='cpu')
            flags_tensor = torch.tensor([int(with_prefill)], dtype=torch.int32, device='cpu')
            packed_tensor = torch.cat([num_tokens_tensor, flags_tensor])
            packed_tensor = _npu_all_reduce_cpu_tensor(packed_tensor, get_dp_group().cpu_group)
            num_tokens_across_dp = packed_tensor[:-1]
            synced_flags = packed_tensor[-1:]
            max_tokens_across_dp = torch.max(num_tokens_across_dp).item()
            global_with_prefill = bool(synced_flags[0])
            num_tokens_after_padding = torch.tensor([max_tokens_across_dp] * self.dp_size,
                                                    device='cpu',
                                                    dtype=torch.int32)
            return max_tokens_across_dp, num_tokens_after_padding, global_with_prefill

        _sync_metadata_across_dp._swift_npu_dp_sync_patched = True
        _sync_metadata_across_dp._swift_origin = origin_sync_metadata
        runner_cls._sync_metadata_across_dp = _sync_metadata_across_dp

    origin_sync_batch = getattr(runner_cls, '_sync_batch_across_dp', None)
    if origin_sync_batch is not None and not getattr(origin_sync_batch, '_swift_npu_dp_sync_patched', False):

        def _sync_batch_across_dp(self,
                                  num_tokens_padded: int | None = None,
                                  cudagraph_mode: int = 0,
                                  allow_dp_padding: bool = False):
            if self.dp_size == 1:
                return False, None, cudagraph_mode
            if should_skip_allreduce_across_dp_group(self.vllm_config):
                num_tokens_after_padding = torch.tensor([num_tokens_padded] * self.dp_size,
                                                        device='cpu',
                                                        dtype=torch.int32)
                return False, num_tokens_after_padding, cudagraph_mode

            tensor = torch.zeros(2, self.dp_size, device='cpu', dtype=torch.int32)
            tensor[0][self.dp_rank] = num_tokens_padded
            tensor[1][self.dp_rank] = cudagraph_mode
            tensor = _npu_all_reduce_cpu_tensor(tensor, get_dp_group().cpu_group)
            num_tokens_across_dp = tensor[0, :]
            max_num_tokens = int(num_tokens_across_dp.max().item())
            if allow_dp_padding:
                num_tokens_after_padding = torch.tensor([max_num_tokens] * len(num_tokens_across_dp),
                                                        device='cpu',
                                                        dtype=torch.int32)
            else:
                num_tokens_after_padding = num_tokens_across_dp.cpu()
            synced_cudagraph_mode = int(tensor[1, :].min().item())
            return False, num_tokens_after_padding, synced_cudagraph_mode

        _sync_batch_across_dp._swift_npu_dp_sync_patched = True
        _sync_batch_across_dp._swift_origin = origin_sync_batch
        runner_cls._sync_batch_across_dp = _sync_batch_across_dp


def _patch_vllm_ascend_mem_get_info() -> None:
    try:
        from vllm_ascend.platform import NPUPlatform
    except (ImportError, AttributeError):
        return
    if getattr(NPUPlatform, '_swift_mem_get_info_patched', False):
        return

    @classmethod
    def mem_get_info(cls, device=None):
        if device is None:
            return torch.npu.mem_get_info()
        try:
            return torch.npu.mem_get_info(device=device)
        except TypeError:
            return torch.npu.mem_get_info()

    NPUPlatform.mem_get_info = mem_get_info
    NPUPlatform._swift_mem_get_info_patched = True


def _patch_vllm_ascend_memory_profiling() -> None:
    try:
        from vllm.logger import logger as vllm_logger
        from vllm_ascend.worker import worker as ascend_worker
    except (ImportError, AttributeError, RuntimeError):
        return

    origin_memory_profiling = getattr(ascend_worker, 'memory_profiling', None)
    if origin_memory_profiling is None or getattr(origin_memory_profiling,
                                                 '_swift_ascend_memory_profiling_patched', False):
        return

    @contextmanager
    def memory_profiling(baseline_snapshot, weights_memory: int = 0):
        with origin_memory_profiling(baseline_snapshot, weights_memory=weights_memory) as profile_result:
            yield profile_result

        if profile_result.after_profile.free_memory < baseline_snapshot.free_memory:
            return

        # In colocated GRPO, the training side may release NPU memory while
        # vLLM-Ascend is profiling KV cache capacity. vLLM-Ascend assumes
        # external memory is stable and asserts when free memory increases.
        profile_result.after_profile.free_memory = max(baseline_snapshot.free_memory - 1, 0)
        profile_result.after_profile.cuda_memory = (
            profile_result.after_profile.total_memory - profile_result.after_profile.free_memory)
        profile_result.non_torch_increase = max(profile_result.non_torch_increase, 0)
        profile_result.torch_peak_increase = max(profile_result.torch_peak_increase, 0)
        profile_result.non_kv_cache_memory = (
            profile_result.non_torch_increase + profile_result.torch_peak_increase + profile_result.weights_memory)
        vllm_logger.warning_once(
            'Patched vLLM-Ascend memory profiling because free memory increased during profiling. '
            'This is expected in colocated training when another rank releases memory.')

    memory_profiling._swift_ascend_memory_profiling_patched = True
    memory_profiling._swift_origin = origin_memory_profiling
    ascend_worker.memory_profiling = memory_profiling


def _patch_vllm_ascend_device_op_nonquant_routing() -> None:
    """Use the stable torch-npu routing op for non-quantized MoE when needed.

    Some released vLLM-Ascend versions route the non-quantized MoE case
    (``scale is None`` and ``quant_mode == -1``) through
    ``npu_moe_init_routing_custom`` / ``aclnnMoeInitRoutingCustom``, which is
    not stable for the parameter combination used by Qwen-style MoE rollout.

    This is intentionally gated by implementation detection instead of a fixed
    version threshold: source builds or future/backported versions may already
    dispatch the non-quantized path to ``torch_npu.npu_moe_init_routing_v2``.
    When that fixed branch is present, skip patching and keep the upstream
    implementation intact.

    Do not probe the custom op by calling it first.  On Ascend, a missing custom
    binary can be reported asynchronously: even if Python catches the immediate
    RuntimeError and falls back, the failed launch can poison the stream and hang
    later at an unrelated event synchronization.  Therefore, when source
    inspection shows that the non-quantized branch still routes to the custom op,
    dispatch that branch directly to ``torch_npu.npu_moe_init_routing_v2``.
    """
    try:
        import torch_npu
        from vllm_ascend.device import device_op
    except (ImportError, AttributeError):
        return

    adaptor_cls = getattr(device_op, 'BaseDeviceAdaptor', None)
    if adaptor_cls is None:
        return
    origin_routing = getattr(adaptor_cls, 'npu_moe_init_routing', None)
    if origin_routing is None or getattr(origin_routing, '_swift_nonquant_routing_patched', False):
        return
    try:
        origin_source = inspect.getsource(origin_routing)
    except (OSError, TypeError):
        origin_source = ''
    if 'npu_moe_init_routing_v2' in origin_source and 'quant_mode == -1' in origin_source:
        return
    def is_nonquant_routing(routing_kwargs) -> bool:
        return routing_kwargs['scale'] is None and routing_kwargs['quant_mode'] == -1

    def npu_moe_init_routing_v2(hidden_states, topk_ids, routing_kwargs):
        active_num = routing_kwargs['active_num']
        expert_num = routing_kwargs['expert_num']
        active_expert_range = routing_kwargs['active_expert_range']
        return torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            scale=None,
            offset=None,
            active_num=0 if active_num is None else active_num,
            expert_capacity=-1,
            expert_num=expert_num,
            drop_pad_mode=0,
            expert_tokens_num_type=routing_kwargs['expert_tokens_num_type'],
            expert_tokens_num_flag=routing_kwargs['expert_tokens_num_flag'],
            active_expert_range=[0, expert_num] if active_expert_range is None else active_expert_range,
            quant_mode=routing_kwargs['quant_mode'],
            row_idx_type=0,
        )

    def patched_npu_moe_init_routing(hidden_states, topk_ids, *args, **kwargs):
        routing_kwargs = {
            'scale': None,
            'active_num': None,
            'expert_num': None,
            'expert_tokens_num_type': 1,
            'expert_tokens_num_flag': True,
            'active_expert_range': None,
            'quant_mode': -1,
        }
        param_names = list(routing_kwargs)
        for index, value in enumerate(args):
            if index < len(param_names):
                routing_kwargs[param_names[index]] = value
        routing_kwargs.update(kwargs)

        if not is_nonquant_routing(routing_kwargs):
            return origin_routing(hidden_states, topk_ids, *args, **kwargs)
        logger.warning_once(
            'Using torch_npu.npu_moe_init_routing_v2 for vLLM-Ascend non-quantized MoE routing. '
            'The installed vLLM-Ascend implementation still dispatches this branch to '
            'npu_moe_init_routing_custom, whose missing custom-op binary fails asynchronously on this stack.')
        return npu_moe_init_routing_v2(hidden_states, topk_ids, routing_kwargs)

    patched_npu_moe_init_routing._swift_nonquant_routing_patched = True
    patched_npu_moe_init_routing._swift_origin = origin_routing
    adaptor_cls.npu_moe_init_routing = staticmethod(patched_npu_moe_init_routing)


def _patch_vllm_ascend_unquant_moe_layout() -> None:
    """Patch vLLM-Ascend unquantized gated-MoE layout before grouped matmul.

    Some vLLM-Ascend versions pass already-processed unquantized MoE weights
    into ``unquant_apply_mlp`` with ``need_trans=False``.  For Qwen-style gated
    MoE, ``w1`` is the concatenated gate/up projection:

        expected  w1: [experts, hidden, 2 * intermediate]
        expected  w2: [experts, intermediate, hidden]
        reversed  w1: [experts, 2 * intermediate, hidden]
        reversed  w2: [experts, hidden, intermediate]

    ``torch_npu.npu_grouped_matmul`` needs the K dimension to match the input
    hidden size.  If the pair is clearly the reversed gated-MoE layout, transpose
    both weights before calling the original implementation.  If the shapes do
    not match this gated-MoE pattern, do not guess; leave the original
    vLLM-Ascend path untouched.
    """
    try:
        from vllm_ascend.ops.fused_moe import moe_mlp
    except (ImportError, AttributeError, RuntimeError):
        return
    origin_apply_mlp = getattr(moe_mlp, 'unquant_apply_mlp', None)
    if origin_apply_mlp is None or getattr(origin_apply_mlp, '_swift_ascend_moe_layout_patched', False):
        return

    def _is_expected_gated_unquant_moe_layout(w1, w2, input_k: int) -> bool:
        return (w1.shape[1] == input_k and w2.shape[2] == input_k and w1.shape[2] % 2 == 0
                and w1.shape[2] // 2 == w2.shape[1])

    def _is_reversed_gated_unquant_moe_layout(w1, w2, input_k: int) -> bool:
        return (w1.shape[2] == input_k and w2.shape[1] == input_k and w1.shape[1] % 2 == 0
                and w1.shape[1] // 2 == w2.shape[2])

    def patched_unquant_apply_mlp(hidden_states, w1, w2, *args, **kwargs):
        need_trans = kwargs.get('need_trans', args[6] if len(args) > 6 else True)
        if not need_trans and w1.dim() == 3 and w2.dim() == 3:
            input_k = hidden_states.shape[-1]
            # This compatibility branch is only for gated MoE layouts where
            # w1 stores concatenated gate/up projections:
            #   w1=[experts, hidden, 2*intermediate],
            #   w2=[experts, intermediate, hidden].
            # Other MoE implementations may not have the 2x relation, so leave
            # them to vLLM-Ascend's original implementation instead of guessing.
            if _is_expected_gated_unquant_moe_layout(w1, w2, input_k):
                pass
            elif _is_reversed_gated_unquant_moe_layout(w1, w2, input_k):
                w1 = w1.transpose(1, 2)
                w2 = w2.transpose(1, 2)
        return origin_apply_mlp(hidden_states, w1, w2, *args, **kwargs)

    patched_unquant_apply_mlp._swift_ascend_moe_layout_patched = True
    patched_unquant_apply_mlp._swift_origin = origin_apply_mlp
    moe_mlp.unquant_apply_mlp = patched_unquant_apply_mlp


def _patch_vllm_ascend_colocate_memory_profiling() -> None:
    try:
        from vllm.utils.mem_constants import GiB_bytes
        from vllm.utils.mem_utils import memory_profiling
        from vllm_ascend.worker import worker as ascend_worker
    except (ImportError, AttributeError, RuntimeError):
        return

    NPUWorker = getattr(ascend_worker, 'NPUWorker', None)
    if NPUWorker is None:
        return
    origin_determine = getattr(NPUWorker, 'determine_available_memory', None)
    if origin_determine is None or getattr(origin_determine, '_swift_colocate_memory_patched', False):
        return

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        GiB = lambda b: b / GiB_bytes
        with memory_profiling(
                self.init_snapshot,
                weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        free_gpu_memory = profile_result.after_profile.free_memory
        if self.init_snapshot.free_memory <= free_gpu_memory:
            ascend_worker.logger.warning(
                'vLLM-Ascend memory profiling observed increased free memory '
                'during colocate initialization: initial %.2f GiB, current %.2f GiB. '
                'Continuing with profiled non-KV memory instead of failing.',
                GiB(self.init_snapshot.free_memory), GiB(free_gpu_memory))

        self.available_kv_cache_memory_bytes = self.requested_memory - profile_result.non_kv_cache_memory
        ascend_worker.logger.debug(profile_result)
        ascend_worker.logger.info_once(
            'Available KV cache memory: %.2f GiB', GiB(self.available_kv_cache_memory_bytes), scope='local')
        return int(self.available_kv_cache_memory_bytes)

    determine_available_memory._swift_colocate_memory_patched = True
    determine_available_memory._swift_origin = origin_determine
    NPUWorker.determine_available_memory = determine_available_memory


def patch_vllm_ascend_moe_expert_weight_loader(experts, name: str, param) -> None:
    """Patch one vLLM-Ascend processed MoE expert parameter loader.

    vLLM-Ascend transposes unquantized MoE weights after the initial model load
    so grouped matmul can consume them efficiently.  During GRPO colocate weight
    sync, however, SWIFT still sends regular 2D HF/Megatron weights, for example:

        gate_proj/up_proj: [intermediate, hidden] -> w13_weight
        down_proj       : [hidden, intermediate] -> w2_weight

    The target vLLM-Ascend parameters are already processed 3D tensors, e.g.:

        w13_weight: [local_experts, hidden, 2 * intermediate_per_tp]
        w2_weight : [local_experts, intermediate_per_tp, hidden]

    This wrapper keeps the normal vLLM loader for ordinary cases, and only
    handles the processed 3D vLLM-Ascend layout when a 2D runtime-sync tensor is
    loaded into w13_weight or w2_weight.
    """
    if 'w13_weight' not in name and 'w2_weight' not in name:
        return
    quant_method = getattr(experts, 'quant_method', None)
    quant_method_module = type(quant_method).__module__ if quant_method is not None else ''
    if not quant_method_module.startswith('vllm_ascend'):
        return

    def make_ascend_moe_weight_loader(experts, origin_weight_loader):

        def load_processed_ascend_weight(param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
            quant_method = getattr(experts, 'quant_method', None)
            quant_method_module = type(quant_method).__module__ if quant_method is not None else ''
            # Only the GRPO runtime-sync path needs special handling here:
            # SWIFT provides normal 2D HF/Megatron tensors, while the target
            # vLLM-Ascend MoE parameter has already been converted to a 3D
            # per-local-expert layout.  Initial checkpoint load and other
            # layouts continue to use the original vLLM loader.
            is_runtime_sync_into_processed_param = (param.data.dim() == 3 and loaded_weight.dim() == 2
                                                    and quant_method_module.startswith('vllm_ascend'))
            if not is_runtime_sync_into_processed_param:
                return origin_weight_loader(param, loaded_weight, weight_name, shard_id, expert_id, return_success)

            is_w13_shard = shard_id in {'w1', 'w3'} and 'w13_weight' in weight_name
            is_w2_shard = shard_id == 'w2' and 'w2_weight' in weight_name

            # Runtime sync may see a Parameter whose data was restored to the
            # pre-processed orientation.  Rebuild the vLLM-Ascend orientation
            # before copying the incoming HF/Megatron 2D shard.
            if is_w13_shard:
                if param.data.shape[-1] == loaded_weight.shape[-1] and param.data.shape[-2] != loaded_weight.shape[-1]:
                    param.data = param.data.transpose(1, 2).contiguous()
            elif is_w2_shard:
                if param.data.shape[-2] == loaded_weight.shape[0] and param.data.shape[-1] != loaded_weight.shape[0]:
                    param.data = param.data.transpose(1, 2).contiguous()

            local_expert_id = experts._map_global_expert_id_to_local_expert_id(expert_id)
            if local_expert_id == -1:
                return False if return_success else None

            tp_rank = experts.tp_rank
            param_data = param.data[local_expert_id]

            if is_w13_shard:
                # Example:
                #   loaded gate/up shard: [intermediate, hidden]
                #   target w13 slot    : [hidden, 2 * intermediate_per_tp]
                #
                # TP slices the intermediate dimension.  w1 occupies the first
                # half of w13_weight and w3 occupies the second half, so copy a
                # transposed TP slice into the selected half.
                shard_size = param_data.shape[1] // 2
                loaded_weight = loaded_weight.narrow(0, shard_size * tp_rank, shard_size)
                offset = 0 if shard_id == 'w1' else shard_size
                param_data[:, offset:offset + shard_size].copy_(loaded_weight.transpose(0, 1).contiguous())
                return True if return_success else None

            if is_w2_shard:
                # Example:
                #   loaded down shard: [hidden, intermediate]
                #   target w2 slot  : [intermediate_per_tp, hidden]
                #
                # TP slices the intermediate dimension on loaded_weight dim 1;
                # vLLM-Ascend stores the processed local shard transposed.
                shard_size = param_data.shape[0]
                loaded_weight = loaded_weight.narrow(1, shard_size * tp_rank, shard_size)
                param_data.copy_(loaded_weight.transpose(0, 1).contiguous())
                return True if return_success else None

            return origin_weight_loader(param, loaded_weight, weight_name, shard_id, expert_id, return_success)

        load_processed_ascend_weight._swift_ascend_moe_weight_loader = True
        return load_processed_ascend_weight

    if not hasattr(experts, 'weight_loader'):
        return
    weight_loader = getattr(param, 'weight_loader', experts.weight_loader)
    if not getattr(weight_loader, '_swift_ascend_moe_weight_loader', False):
        param.weight_loader = make_ascend_moe_weight_loader(experts, weight_loader)


__all__ = [
    'create_npu_process_group',
    'patch_vllm_ascend_colocate_runtime',
    'patch_vllm_ascend_moe_expert_weight_loader',
    'patch_vllm_ascend_runtime',
    'prepare_vllm_ascend_device_groups_before_megatron',
    'prepare_vllm_ascend_dp_groups_before_megatron',
    'register_megatron_hccl_groups_for_vllm',
]
