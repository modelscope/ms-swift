# Copyright (c) ModelScope Contributors. All rights reserved.
"""Runtime vLLM group patches for NPU external-launcher colocate mode.

The factory module prepares HCCL/Gloo groups before vLLM initialization.  This
module patches vLLM-Ascend so GroupCoordinator only performs cache lookup and so
small DP metadata tensors use real NPU/HCCL DP groups without pretending those
HCCL groups are CPU collectives.
"""
from __future__ import annotations

import hashlib
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

    _patch_vllm_external_launcher_dp_tensor_sync(parallel_config_module, model_runner_v1,
                                                 should_skip_allreduce_across_dp_group)


def _patch_vllm_external_launcher_dp_tensor_sync(parallel_config_module, model_runner_v1,
                                                 should_skip_allreduce_across_dp_group) -> None:
    """Patch small DP metadata collectives to use NPU tensors on HCCL groups.

    vLLM's DP metadata helpers operate on CPU tensors because upstream expects a
    CPU/Gloo group.  In this NPU colocate path, the cached DP group is a real
    HCCL group, so the wrapper moves those tiny CPU tensors to the current NPU,
    runs the collective on the real DP HCCL group, and converts results back to
    CPU for the original call contract.
    """
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
        if origin_has_unfinished is not None and not getattr(origin_has_unfinished, '_swift_npu_dp_sync_patched',
                                                             False):

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

        def _sync_metadata_across_dp(self, num_tokens: int, with_prefill: bool = False, is_draft_model: bool = False):
            if self.dp_size == 1:
                return num_tokens, None, with_prefill
            if should_skip_allreduce_across_dp_group(self.vllm_config, is_draft_model):
                num_tokens_after_padding = torch.tensor([num_tokens] * self.dp_size, device='cpu', dtype=torch.int32)
                return num_tokens, num_tokens_after_padding, with_prefill

            num_tokens_tensor = torch.tensor([num_tokens if i == self.dp_rank else 0 for i in range(self.dp_size)],
                                             dtype=torch.int32,
                                             device='cpu')
            flags_tensor = torch.tensor([int(with_prefill)], dtype=torch.int32, device='cpu')
            packed_tensor = torch.cat([num_tokens_tensor, flags_tensor])
            packed_tensor = _npu_all_reduce_cpu_tensor(packed_tensor, get_dp_group().cpu_group)
            num_tokens_across_dp = packed_tensor[:-1]
            synced_flags = packed_tensor[-1:]
            max_tokens_across_dp = torch.max(num_tokens_across_dp).item()
            global_with_prefill = bool(synced_flags[0])
            num_tokens_after_padding = torch.tensor(
                [max_tokens_across_dp] * self.dp_size, device='cpu', dtype=torch.int32)
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
                num_tokens_after_padding = torch.tensor(
                    [num_tokens_padded] * self.dp_size, device='cpu', dtype=torch.int32)
                return False, num_tokens_after_padding, cudagraph_mode

            tensor = torch.zeros(2, self.dp_size, device='cpu', dtype=torch.int32)
            tensor[0][self.dp_rank] = num_tokens_padded
            tensor[1][self.dp_rank] = cudagraph_mode
            tensor = _npu_all_reduce_cpu_tensor(tensor, get_dp_group().cpu_group)
            num_tokens_across_dp = tensor[0, :]
            max_num_tokens = int(num_tokens_across_dp.max().item())
            if allow_dp_padding:
                num_tokens_after_padding = torch.tensor(
                    [max_num_tokens] * len(num_tokens_across_dp), device='cpu', dtype=torch.int32)
            else:
                num_tokens_after_padding = num_tokens_across_dp.cpu()
            synced_cudagraph_mode = int(tensor[1, :].min().item())
            return False, num_tokens_after_padding, synced_cudagraph_mode

        _sync_batch_across_dp._swift_npu_dp_sync_patched = True
        _sync_batch_across_dp._swift_origin = origin_sync_batch
        runner_cls._sync_batch_across_dp = _sync_batch_across_dp


__all__ = ['patch_vllm_ascend_external_launcher_groups']
