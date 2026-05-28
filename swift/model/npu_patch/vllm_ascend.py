# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import torch
from contextlib import contextmanager
from importlib import metadata
from packaging import version
from typing import Any, Optional

from swift.utils.logger import get_logger

logger = get_logger()

_SWIFT_VLLM_DP_GROUP_SPECS: set[tuple[tuple[int, ...], ...]] = set()
_SWIFT_VLLM_PRECREATED_DP_GROUPS: dict[tuple[int, ...], Any] = {}
_SWIFT_VLLM_PRECREATED_DP_GROUP_SPECS: set[tuple[tuple[int, ...], ...]] = set()
_SWIFT_VLLM_TP_GLOO_GROUPS: dict[tuple[tuple[int, ...], ...], Any] = {}


def _env_enabled(name: str, default: str = '1') -> bool:
    return os.environ.get(name, default) == '1'


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
    if not _env_enabled('SWIFT_PATCH_VLLM_ASCEND_RUNTIME'):
        return

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
            group = dist.new_group(ranks, backend='gloo')
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


def _build_vllm_dp_group_ranks(world_size: int, data_parallel_size: int, tensor_parallel_size: int):
    if data_parallel_size <= 1:
        return []
    group_unit = data_parallel_size * tensor_parallel_size
    if world_size % group_unit != 0:
        raise RuntimeError(
            f'Cannot build vLLM DP groups: world_size={world_size}, '
            f'data_parallel_size={data_parallel_size}, tensor_parallel_size={tensor_parallel_size}.')
    all_ranks = torch.arange(world_size).reshape(-1, data_parallel_size, 1, 1, tensor_parallel_size)
    group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)
    return [x.tolist() for x in group_ranks]


def _get_vllm_data_parallel_size_from_args(args) -> int:
    engine_kwargs = getattr(args, 'vllm_engine_kwargs', None) or {}
    if isinstance(engine_kwargs, str):
        engine_kwargs = json.loads(engine_kwargs)
    return int(engine_kwargs.get('data_parallel_size', 1) or 1)


def prepare_vllm_ascend_dp_groups_before_megatron(args) -> None:
    """Pre-create vLLM DP HCCL groups before vLLM builds its topology.

    In the vLLM-Ascend 0.18 colocated Megatron path, creating the vLLM DP HCCL
    subgroup late inside vLLM initialization can hang on this stack, even though
    the same strided DP groups work in a clean torchrun smoke.  The difference
    is ordering: by then the process has already entered vLLM worker
    initialization.  Create the exact vLLM DP groups right after Megatron has
    created its own TP/PP/CP/EP groups, but still before vLLM initializes; vLLM
    later reuses the stored ProcessGroup instead of creating it again.
    """
    if not getattr(args, 'use_vllm', False) or getattr(args, 'vllm_mode', None) != 'colocate':
        return

    data_parallel_size = _get_vllm_data_parallel_size_from_args(args)
    if data_parallel_size <= 1:
        return

    import torch.distributed as dist

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    tensor_parallel_size = int(getattr(args, 'vllm_tensor_parallel_size', 1) or 1)
    group_ranks = _build_vllm_dp_group_ranks(world_size, data_parallel_size, tensor_parallel_size)
    spec = _canonical_group_ranks(group_ranks)
    if spec in _SWIFT_VLLM_PRECREATED_DP_GROUP_SPECS:
        return

    backend = dist.get_backend()
    own_group = None
    own_ranks = None
    for ranks in group_ranks:
        group = dist.new_group(ranks, backend=backend)
        ranks_tuple = tuple(int(r) for r in ranks)
        if rank in ranks:
            own_group = group
            own_ranks = ranks_tuple
            _SWIFT_VLLM_PRECREATED_DP_GROUPS[ranks_tuple] = group

    if own_group is None or own_ranks is None:
        raise RuntimeError(f'Rank {rank} is not included in vLLM DP group spec: {group_ranks}.')

    device = torch.device('npu', torch.npu.current_device())
    for ranks in group_ranks:
        marker = torch.ones((), dtype=torch.int32, device=device)
        dist.all_reduce(marker)
        if rank in ranks:
            rank_value = torch.tensor([rank], dtype=torch.int32, device=device)
            reduced = rank_value.clone()
            dist.all_reduce(reduced, group=_SWIFT_VLLM_PRECREATED_DP_GROUPS[tuple(int(r) for r in ranks)])
            expected_sum = sum(int(r) for r in ranks)
            if int(reduced.cpu().item()) != expected_sum:
                raise RuntimeError(
                    f'Pre-created vLLM DP group all_reduce mismatch: rank={rank}, ranks={ranks}, '
                    f'got={int(reduced.cpu().item())}, expected={expected_sum}.')
        marker = torch.ones((), dtype=torch.int32, device=device)
        dist.all_reduce(marker)

    _SWIFT_VLLM_PRECREATED_DP_GROUP_SPECS.add(spec)
    logger.warning_once(f'Pre-created vLLM DP HCCL groups before vLLM init: ranks={spec}')


def _patch_vllm_ascend_external_launcher_groups() -> None:
    """Create NPU-only vLLM groups when Megatron already owns torch.distributed.

    In colocated Megatron GRPO, vLLM runs in ``external_launcher`` mode and
    reuses Megatron's already-initialized default HCCL world.  vLLM-Ascend 0.18
    then creates an extra Gloo CPU group for each ``GroupCoordinator``.  On this
    stack, creating those Gloo groups inside the Megatron process tree can hang
    before vLLM finishes initialization.

    Tensor-parallel and decode-context groups use vLLM's shared-memory message
    queue, so their upstream Gloo group is left intact.  For the other NPU-only
    groups, build only the HCCL device group and alias ``cpu_group`` to that
    HCCL group when vLLM later needs scalar DP synchronization.  Those scalar
    sync sites are patched below to move CPU tensors to NPU before calling
    collectives on the aliased group.
    """
    try:
        import torch.distributed as dist
        from vllm.config import parallel as parallel_config_module
        from vllm.distributed import parallel_state
        from vllm.v1.executor import uniproc_executor
        from vllm.v1.executor.abstract import Executor
        from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator
        from vllm_ascend.patch.worker import patch_distributed
        from vllm_ascend.utils import create_hccl_pg_options, should_skip_allreduce_across_dp_group
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

        def _sync_default_world(device) -> None:
            # Use the already-stable Megatron default HCCL world as a phase
            # barrier.  This keeps disjoint vLLM DP HCCL subgroups from warming
            # at the same time during initialization.
            marker = torch.ones((), dtype=torch.int32, device=device)
            dist.all_reduce(marker)

        def _warmup_dp_group(group_name, group_ranks, ranks, device_group) -> bool:
            if group_name != 'dp' or len(ranks) <= 1 or dist.get_rank() not in ranks:
                return False
            device = torch.device('npu', torch.npu.current_device())
            warmed = False
            for dp_ranks in group_ranks:
                _sync_default_world(device)
                if list(dp_ranks) != list(ranks):
                    _sync_default_world(device)
                    continue

                rank_value = torch.tensor([dist.get_rank()], dtype=torch.int32, device=device)
                reduced = rank_value.clone()
                dist.all_reduce(reduced, group=device_group)
                expected_sum = sum(int(rank) for rank in ranks)
                if int(reduced.cpu().item()) != expected_sum:
                    raise RuntimeError(
                        f'vLLM DP group warmup all_reduce mismatch: rank={dist.get_rank()}, ranks={ranks}, '
                        f'got={int(reduced.cpu().item())}, expected={expected_sum}.')

                gathered = torch.empty((len(ranks), ), dtype=torch.int32, device=device)
                dist.all_gather_into_tensor(gathered, rank_value, group=device_group)
                expected = torch.tensor([int(rank) for rank in ranks], dtype=torch.int32)
                if not torch.equal(gathered.cpu(), expected):
                    raise RuntimeError(
                        f'vLLM DP group warmup all_gather mismatch: rank={dist.get_rank()}, ranks={ranks}, '
                        f'got={gathered.cpu().tolist()}, expected={expected.tolist()}.')
                logger.warning_once(f'Warmed vLLM DP HCCL subgroup in Megatron colocate mode: ranks={tuple(ranks)}')
                warmed = True
                _sync_default_world(device)
            return warmed

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
            self._swift_dp_group_warmed = False
            own_precreated_group = None
            if group_name == 'dp':
                for ranks in group_ranks:
                    if self.rank in ranks:
                        own_precreated_group = _SWIFT_VLLM_PRECREATED_DP_GROUPS.get(tuple(int(r) for r in ranks))
                        if own_precreated_group is not None:
                            self.ranks = ranks
                            self.world_size = len(ranks)
                            self.rank_in_group = ranks.index(self.rank)
                            self_device_group = own_precreated_group
                            self._swift_dp_group_warmed = True
                        break

            if self_device_group is None:
                hccl_pg_options = create_hccl_pg_options(group_name)
                for ranks in group_ranks:
                    device_group = dist.new_group(ranks, backend=torch_distributed_backend, pg_options=hccl_pg_options)
                    if self.rank in ranks:
                        self.ranks = ranks
                        self.world_size = len(ranks)
                        self.rank_in_group = ranks.index(self.rank)
                        self_device_group = device_group

            assert self_device_group is not None
            if not self._swift_dp_group_warmed:
                self._swift_dp_group_warmed = _warmup_dp_group(group_name, group_ranks, self.ranks, self_device_group)
            self._swift_no_cpu_group = True
            self._swift_cpu_group_uses_device_group = use_device_communicator
            self.cpu_group = self_device_group if use_device_communicator else None
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
            if own_precreated_group is not None:
                logger.warning_once(
                    f'Reused pre-created vLLM-Ascend {group_name} HCCL group in colocated Megatron training; '
                    'the Gloo CPU group is skipped for this NPU-only group.')
            else:
                logger.warning_once(
                    f'Skipped vLLM-Ascend {group_name} Gloo CPU group in colocated Megatron training; '
                    'NPU/HCCL communication is used for this vLLM group.')

        group_cls.__init__ = patched_init
        group_cls._swift_external_launcher_group_patched = True

        origin_destroy = getattr(group_cls, 'destroy', None)
        if origin_destroy is not None and not getattr(origin_destroy, '_swift_external_launcher_group_patched', False):

            def destroy(self):
                if getattr(self, '_swift_no_cpu_group', False) and not getattr(self, '_swift_cpu_group_uses_device_group',
                                                                              False):
                    if hasattr(self, 'cpu_group'):
                        del self.cpu_group
                return origin_destroy(self)

            destroy._swift_external_launcher_group_patched = True
            destroy._swift_origin = origin_destroy
            group_cls.destroy = destroy

    origin_node_count = getattr(parallel_state, '_node_count', None)
    if origin_node_count is not None and not getattr(origin_node_count, '_swift_external_launcher_group_patched', False):

        def _node_count(pg):
            if pg is None:
                return 1
            return origin_node_count(pg)

        _node_count._swift_external_launcher_group_patched = True
        _node_count._swift_origin = origin_node_count
        parallel_state._node_count = _node_count

    executor_cls = getattr(uniproc_executor, 'ExecutorWithExternalLauncher', None)
    origin_determine_memory = getattr(executor_cls, 'determine_available_memory', None) if executor_cls else None
    if origin_determine_memory is not None and not getattr(origin_determine_memory,
                                                          '_swift_external_launcher_group_patched', False):

        def determine_available_memory(self):
            memory = Executor.determine_available_memory(self)
            world = parallel_state.get_world_group()
            if getattr(world, '_swift_no_cpu_group', False):
                return memory
            return origin_determine_memory(self)

        determine_available_memory._swift_external_launcher_group_patched = True
        determine_available_memory._swift_origin = origin_determine_memory
        executor_cls.determine_available_memory = determine_available_memory

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
    if not _env_enabled('SWIFT_PATCH_VLLM_ASCEND_MOE_LAYOUT'):
        return
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
    'patch_vllm_ascend_colocate_runtime',
    'patch_vllm_ascend_moe_expert_weight_loader',
    'patch_vllm_ascend_runtime',
    'prepare_vllm_ascend_dp_groups_before_megatron',
]
