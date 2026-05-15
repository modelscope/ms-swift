# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .driver_utils import (RayConfig, build_dataset_from_dict, compute_iter_params, estimate_dp_size,
                               extract_iteration, merge_group_dict, parse_args_from_dict, parse_ray_yaml)
    from .grpo_trainer import GRPOTrainer
    from .loss import GRPOLoss, Loss
    from .megatron_worker import MegatronWorker
    from .pipeline import MegatronRayPipeline, register_ray_trainer
    from .resource_pool import ResourcePool, ResourcePoolManager
    from .rollout import RolloutAdapter, RolloutMode, RolloutReplica, VllmEngineConfig, VllmServer
    from .worker_group import CollectMode, DispatchMode, WorkerGroup, dispatch_collect


def __getattr__(name):
    _imports = {
        'RayConfig': '.driver_utils',
        'build_dataset_from_dict': '.driver_utils',
        'compute_iter_params': '.driver_utils',
        'estimate_dp_size': '.driver_utils',
        'extract_iteration': '.driver_utils',
        'merge_group_dict': '.driver_utils',
        'parse_args_from_dict': '.driver_utils',
        'parse_ray_yaml': '.driver_utils',
        'GRPOTrainer': '.grpo_trainer',
        'MegatronRayPipeline': '.pipeline',
        'register_ray_trainer': '.pipeline',
        'Loss': '.loss',
        'GRPOLoss': '.loss',
        'ResourcePool': '.resource_pool',
        'ResourcePoolManager': '.resource_pool',
        'RolloutMode': '.rollout',
        'RolloutReplica': '.rollout',
        'VllmEngineConfig': '.rollout',
        'VllmServer': '.rollout',
        'RolloutAdapter': '.rollout',
        'MegatronWorker': '.megatron_worker',
        'CollectMode': '.worker_group',
        'DispatchMode': '.worker_group',
        'WorkerGroup': '.worker_group',
        'dispatch_collect': '.worker_group',
    }
    if name in _imports:
        import importlib
        return getattr(importlib.import_module(_imports[name], __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
