# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rollout package for Ray-based vLLM integration.

Architecture (three layers):

    1. VllmServer (vllm_server.py):
       Ray Actor hosting AsyncLLM engine. Handles sampling, sleep/wake_up,
       and coordinates weight sync via collective_rpc to worker extension.

    2. RolloutReplica (replica.py):
       Manages VllmServer lifecycle and deployment modes (HYBRID / SEPARATED).
       Provides unified API for generate / update_weights / sleep / wake_up.

    3. Weight sync utilities (weight_sync.py):
       Adapts Megatron bridge export_weights to VllmServer's IPC-based
       update_weights. Reuses swift's FlattenedTensorBucket for bucket
       packing on the sender side.

Reused from existing swift:
    - swift.pipelines.infer.rollout.WeightSyncWorkerExtension (NCCL path)
    - swift.rlhf_trainers.utils.FlattenedTensorBucket
    - swift.rlhf_trainers.utils.patch_vllm_moe_model_weight_loader
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .replica import RolloutMode, RolloutReplica
    from .vllm_server import VllmServer
    from .weight_sync import sync_megatron_weights_to_vllm


def __getattr__(name):
    _imports = {
        'RolloutMode': '.replica',
        'RolloutReplica': '.replica',
        'VllmServer': '.vllm_server',
        'sync_megatron_weights_to_vllm': '.weight_sync',
    }
    if name in _imports:
        import importlib
        return getattr(importlib.import_module(_imports[name], __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
