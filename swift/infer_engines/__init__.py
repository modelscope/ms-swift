# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .vllm_engine import VllmEngine
    from .grpo_vllm_engine import GRPOVllmEngine
    from .lmdeploy_engine import LmdeployEngine
    from .sglang_engine import SglangEngine
    from .pt_engine import PtEngine
    from .infer_client import InferClient
    from .infer_engine import InferEngine
    from .base import BaseInferEngine
    from .utils import prepare_generation_config, AdapterRequest, patch_vllm_memory_leak
else:
    _import_structure = {
        'vllm_engine': ['VllmEngine'],
        'grpo_vllm_engine': ['GRPOVllmEngine'],
        'lmdeploy_engine': ['LmdeployEngine'],
        'sglang_engine': ['SglangEngine'],
        'pt_engine': ['PtEngine'],
        'infer_client': ['InferClient'],
        'infer_engine': ['InferEngine'],
        'base': ['BaseInferEngine'],
        'utils': ['prepare_generation_config', 'AdapterRequest', 'patch_vllm_memory_leak'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
