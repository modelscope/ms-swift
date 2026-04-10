# Copyright (c) ModelScope Contributors. All rights reserved.
"""Weight synchronization utilities for Ray-based RL training.

Provides high-level functions that bridge Megatron's weight export
with vLLM's weight loading, supporting both co-located and separated
deployment modes.

Reuses:
    - ``swift.rlhf_trainers.utils.FlattenedTensorBucket`` for bucket packing
    - ``swift.rlhf_trainers.utils.patch_vllm_moe_model_weight_loader``
"""
import os
import torch
from typing import Any, Dict, List, Optional, Tuple

from swift.utils import get_logger

logger = get_logger()


def sync_megatron_weights_to_vllm(
    bridge,
    models,
    replica,
    *,
    target_device: Optional[str] = None,
    use_ipc: bool = True,
    wake_up_first: bool = True,
    reset_cache: bool = True,
) -> Dict[str, str]:
    """Complete weight synchronization from Megatron training to vLLM rollout.

    High-level orchestration:
    1. Optionally wake up vLLM weights
    2. Export weights via bridge
    3. Transfer to vLLM via IPC or direct path
    4. Reset prefix cache

    Args:
        bridge: mcore_bridge instance.
        models: Megatron unwrapped_models list.
        replica: RolloutReplica managing the VllmServer.
        target_device: 'cpu' to offload bridge export, None for GPU.
        use_ipc: Use ZMQ+IPC (True) or direct collective_rpc (False).
        wake_up_first: Wake up vLLM weights before sync.
        reset_cache: Reset prefix cache after sync.

    Returns:
        Status dict from the update.
    """
    if wake_up_first:
        replica.wake_up(tags=['weights'])

    result = replica.update_weights_from_megatron(bridge, models, target_device=target_device, use_ipc=use_ipc)

    if reset_cache:
        replica.reset_prefix_cache()

    logger.info('Weight sync complete: %s', result.get('status', 'unknown'))
    return result


def export_weights_for_transfer(
    bridge,
    models,
    target_device: str = 'cpu',
) -> List[Tuple[str, torch.Tensor]]:
    """Export Megatron weights as a list of (name, cpu_tensor) pairs.

    Used in separated mode where the driver collects weights from
    training workers and sends them to rollout workers.
    """
    return [(n, t.cpu().clone()) for n, t in bridge.export_weights(models, target_device=target_device)]
