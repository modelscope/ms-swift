# Copyright (c) ModelScope Contributors. All rights reserved.
"""Runtime vLLM group patches for NPU external-launcher colocate mode.

The factory module prepares HCCL device groups and matching Gloo CPU groups
before vLLM initialization.  This module patches vLLM-Ascend so
``GroupCoordinator`` only performs cache lookup instead of creating process
groups dynamically inside the Megatron process tree.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import torch

from swift.model.npu_patch.vllm_ascend_group_factory import (_cache_key_summary, _canonical_group_ranks,
                                                             _lookup_vllm_cpu_group_cache,
                                                             _lookup_vllm_device_group_cache)
from swift.utils.logger import get_logger

logger = get_logger()

_SWIFT_VLLM_DP_GROUP_SPECS: set[tuple[tuple[int, ...], ...]] = set()


def _group_spec_hash(group_name: str, group_ranks) -> int:
    """Hash a runtime GroupCoordinator rank spec for cross-rank validation."""
    payload = json.dumps({
        'group_name': group_name,
        'group_ranks': _canonical_group_ranks(group_ranks),
    },
                         sort_keys=True)
    # Keep the hash inside signed int64 so HCCL can compare it through a tensor.
    return int(hashlib.sha256(payload.encode()).hexdigest()[:15], 16)


def patch_vllm_ascend_external_launcher_groups() -> None:
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
        from vllm.distributed import parallel_state
        from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator
        from vllm_ascend.patch.worker import patch_distributed
    except (ImportError, AttributeError, RuntimeError):
        return

    group_cls = getattr(patch_distributed, 'GroupCoordinatorPatch', None)
    if group_cls is not None and not getattr(group_cls, '_swift_external_launcher_group_patched', False):
        origin_init = group_cls.__init__
        origin_signature = inspect.signature(origin_init)
        expected_params = {
            'self',
            'group_ranks',
            'local_rank',
            'torch_distributed_backend',
            'use_device_communicator',
            'use_message_queue_broadcaster',
            'group_name',
        }
        missing_params = expected_params.difference(origin_signature.parameters)
        if missing_params:
            raise RuntimeError('Unsupported vLLM-Ascend GroupCoordinatorPatch.__init__ signature: '
                               f'signature={origin_signature}, missing={sorted(missing_params)}.')

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

        def patched_init(self, *args, **kwargs):
            try:
                bound = origin_signature.bind(self, *args, **kwargs)
            except TypeError as e:
                raise RuntimeError('Failed to bind vLLM-Ascend GroupCoordinatorPatch.__init__ arguments: '
                                   f'signature={origin_signature}, args={args}, kwargs={kwargs}.') from e
            bound.apply_defaults()
            group_ranks = bound.arguments['group_ranks']
            local_rank = bound.arguments['local_rank']
            torch_distributed_backend = bound.arguments['torch_distributed_backend']
            use_device_communicator = bound.arguments['use_device_communicator']
            use_message_queue_broadcaster = bound.arguments['use_message_queue_broadcaster']
            group_name = bound.arguments['group_name']
            group_name = group_name or 'anonymous'
            if not _should_use_npu_only_group(use_device_communicator, use_message_queue_broadcaster, group_name):
                return origin_init(*bound.args, **bound.kwargs)

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
                    raise RuntimeError(f'vLLM device group cache miss in cache-only mode: rank={self.rank}, '
                                       f'group_name={group_name}, backend={torch_distributed_backend}, ranks={ranks}, '
                                       f'cache_keys={_cache_key_summary()}.')
                self_cpu_group = _lookup_vllm_cpu_group_cache(group_name, ranks)
                if self_cpu_group is None:
                    raise RuntimeError(f'vLLM CPU group cache miss in cache-only mode: rank={self.rank}, '
                                       f'group_name={group_name}, ranks={ranks}, cache_keys={_cache_key_summary()}.')
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self_device_group = cached_group
                break
            if self_device_group is None:
                raise RuntimeError(f'Rank {self.rank} is not included in cached vLLM device group spec: '
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
            logger.warning_once(f'Reused vLLM-Ascend factory-cached {group_name} HCCL device group and Gloo CPU group '
                                'in colocated Megatron training.')

        group_cls.__init__ = patched_init
        group_cls._swift_external_launcher_group_patched = True


__all__ = ['patch_vllm_ascend_external_launcher_groups']
