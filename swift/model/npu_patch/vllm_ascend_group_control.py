# Copyright (c) ModelScope Contributors. All rights reserved.
"""Gloo control groups used by SWIFT rollout object collectives on NPU."""
from __future__ import annotations

from typing import Any

from swift.model.npu_patch.vllm_ascend_group_factory import (_canonical_group_ranks,
                                                             _clear_default_pg_bound_device_id_for_gloo)
from swift.utils.logger import get_logger

logger = get_logger()

_SWIFT_VLLM_TP_GLOO_GROUPS: dict[tuple[tuple[int, ...], ...], Any] = {}


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
    import torch.distributed as dist
    from transformers.utils import is_torch_npu_available

    if not is_torch_npu_available() or tensor_parallel_size <= 1 or not dist.is_initialized():
        return None

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size % tensor_parallel_size != 0:
        raise RuntimeError(
            f'Cannot build vLLM TP Gloo control groups: world_size={world_size}, tp={tensor_parallel_size}.')

    group_ranks = [
        list(range(start, start + tensor_parallel_size)) for start in range(0, world_size, tensor_parallel_size)
    ]
    spec = _canonical_group_ranks(group_ranks)
    if spec not in _SWIFT_VLLM_TP_GLOO_GROUPS:
        _clear_default_pg_bound_device_id_for_gloo()
        own_group = None
        for ranks in group_ranks:
            group = dist.new_group(ranks=ranks, backend='gloo')
            if rank in ranks:
                own_group = group
        if own_group is None:
            raise RuntimeError(f'Rank {rank} is not included in vLLM TP group spec: {group_ranks}.')
        _SWIFT_VLLM_TP_GLOO_GROUPS[spec] = own_group
        logger.warning_once(f'Created vLLM TP Gloo control groups for NPU rollout object collectives: ranks={spec}')

    return _SWIFT_VLLM_TP_GLOO_GROUPS[spec]


__all__ = ['get_or_create_vllm_tp_gloo_group']
