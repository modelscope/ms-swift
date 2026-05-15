# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adapter import RolloutAdapter
    from .ray_vllm_engine import RayVllmEngine
    from .replica import RolloutMode, RolloutReplica, VllmEngineConfig
    from .vllm_server import VllmServer
    from .weight_transfer import BucketedWeightSender


def __getattr__(name):
    _imports = {
        'RolloutAdapter': '.adapter',
        'RayVllmEngine': '.ray_vllm_engine',
        'RolloutMode': '.replica',
        'RolloutReplica': '.replica',
        'VllmEngineConfig': '.replica',
        'VllmServer': '.vllm_server',
        'BucketedWeightSender': '.weight_transfer',
    }
    if name in _imports:
        import importlib
        return getattr(importlib.import_module(_imports[name], __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
