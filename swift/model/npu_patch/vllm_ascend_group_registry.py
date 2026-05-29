# Copyright (c) ModelScope Contributors. All rights reserved.
"""Megatron HCCL group registry for vLLM-Ascend colocated rollout.

This module only answers one question: can a desired vLLM device group safely
reuse an existing Megatron HCCL ProcessGroup with the exact same rank order?  If
not, the group factory will precreate a dedicated vLLM group.
"""
from __future__ import annotations

import hashlib
import json
import torch
from dataclasses import dataclass
from typing import Any, Optional

from swift.utils.logger import get_logger

logger = get_logger()

_SWIFT_MEGATRON_HCCL_GROUPS: dict[tuple[str, str, tuple[int, ...]], list['_MegatronGroupRecord']] = {}


@dataclass(frozen=True)
class _MegatronGroupRecord:
    """A Megatron HCCL group candidate that vLLM may reuse.

    ``axis`` and ``source`` are provenance labels for diagnostics.  Reuse is
    keyed by exact ``ranks`` order because collective rank order affects group
    rank and tensor chunk ordering.
    """

    backend: str
    kind: str
    axis: str
    ranks: tuple[int, ...]
    group: Any
    source: str


def create_npu_process_group(ranks=None,
                             backend=None,
                             *,
                             kind: str,
                             group_name: str,
                             source: str,
                             phase: str,
                             **kwargs):
    """Create a torch distributed group through SWIFT's NPU patch layer.

    The extra keyword-only fields are intentionally informational today.  They
    keep every NPU group creation call self-describing, which makes later audit
    logging or error messages possible without changing all call sites again.
    """
    import torch.distributed as dist

    return dist.new_group(ranks=ranks, backend=backend, **kwargs)


def _normalize_backend_name(backend) -> str:
    """Normalize torch backend objects/strings to stable cache key names."""
    backend_name = str(backend or '').lower()
    if 'hccl' in backend_name:
        return 'hccl'
    if 'gloo' in backend_name:
        return 'gloo'
    return backend_name or 'default'


def _get_process_group_backend(group) -> str:
    """Inspect a process group's backend across torch API variants."""
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
    """Return the global ranks for ``group`` with a WORLD fallback."""
    import torch.distributed as dist

    try:
        return tuple(int(rank) for rank in dist.get_process_group_ranks(group))
    except Exception:
        if group is getattr(dist.group, 'WORLD', None):
            return tuple(range(dist.get_world_size()))
        raise


def _register_megatron_hccl_group(axis: str, group, source: str) -> None:
    """Register one inspectable Megatron HCCL group as a reuse candidate."""
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


def _is_uninitialized_group_error(error: Exception) -> bool:
    """Return whether a Megatron getter reports an optional uninitialized group."""
    message = str(error).lower()
    return any(
        pattern in message for pattern in ('not initialized', 'has not been initialized', 'is not initialized',
                                           'not set', 'is none', 'partial dp for optimizer needs to include cp'))


def _call_megatron_group_getter(axis: str, getter_name: str, getter, **kwargs):
    """Call Megatron group getters while surfacing real registration failures."""
    try:
        return getter(check_initialized=False, **kwargs)
    except TypeError as first_error:
        try:
            return getter(**kwargs)
        except TypeError as second_error:
            logger.warning_once(f'Skipped registering Megatron {axis} HCCL group because getter signature is '
                                f'incompatible: getter={getter_name}, kwargs={kwargs}, '
                                f'first_error={first_error!r}, second_error={second_error!r}')
            return None
        except (AssertionError, RuntimeError, ValueError) as e:
            if _is_uninitialized_group_error(e):
                return None
            raise RuntimeError(f'Failed to register Megatron {axis} HCCL group from getter {getter_name} '
                               f'with kwargs={kwargs}.') from e
    except (AssertionError, RuntimeError, ValueError) as e:
        if _is_uninitialized_group_error(e):
            return None
        raise RuntimeError(f'Failed to register Megatron {axis} HCCL group from getter {getter_name} '
                           f'with kwargs={kwargs}.') from e


def _register_megatron_group_from_getter(parallel_state, axis: str, getter_name: str, **kwargs) -> None:
    """Register a Megatron group if the requested getter exists and succeeds."""
    getter = getattr(parallel_state, getter_name, None)
    if getter is None:
        return None
    group = _call_megatron_group_getter(axis, getter_name, getter, **kwargs)
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
        ('dp_cp', 'get_data_parallel_group', {
            'with_context_parallel': True
        }),
        ('partial_dp', 'get_data_parallel_group', {
            'partial_data_parallel': True
        }),
        ('partial_dp_cp', 'get_data_parallel_group', {
            'with_context_parallel': True,
            'partial_data_parallel': True,
        }),
        ('ep', 'get_expert_model_parallel_group', {}),
        ('etp', 'get_expert_tensor_parallel_group', {}),
        ('edp', 'get_expert_data_parallel_group', {}),
        ('partial_edp', 'get_expert_data_parallel_group', {
            'partial_expert_data_parallel': True
        }),
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
    """Find a reuse candidate with exact rank order, preferring matching axes."""
    records = _SWIFT_MEGATRON_HCCL_GROUPS.get(('hccl', 'device', ranks), [])
    if not records:
        return None
    for axis in preferred_axes or ():
        for record in records:
            if record.axis == axis:
                return record
    return records[0]


def _megatron_group_record_token(record: _MegatronGroupRecord) -> dict[str, str]:
    """Serialize record provenance for cross-rank reuse consensus checks."""
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
        raise RuntimeError(f'Megatron group reuse decision is inconsistent: rank={rank}, group_name={group_name}, '
                           f'ranks={ranks_tuple}.')
    if not _validate_reuse_record_consensus(group_name, ranks_tuple, record):
        return False, None
    return True, record


def _stable_int64_hash(payload) -> int:
    """Hash JSON-serializable payloads into signed-int64-safe positive values."""
    text = json.dumps(payload, sort_keys=True)
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)


__all__ = [
    'create_npu_process_group',
    'register_megatron_hccl_groups_for_vllm',
]
