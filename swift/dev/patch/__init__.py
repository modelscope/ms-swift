# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from .accelerate_offload_hook import AccelerateSkipOffloadHookPatch
from .torch_conv3d import TorchConv3dPatch
from .torch_dist_save import TorchDistSaveThreadCountPatch
from .transformers_check_imports import TransformersIgnoreCheckImportsPatch
from .transformers_dynamic_module import TransformersDynamicModulePatch
from .transformers_hub_kernels import TransformersHubKernelPatch
from .transformers_qwen3_tts import Qwen3TTSTrainingPatch
from .unsloth_distributed import UnslothDistributedFunctionPatch

__all__ = [
    'AccelerateSkipOffloadHookPatch',
    'Qwen3TTSTrainingPatch',
    'TorchConv3dPatch',
    'TorchDistSaveThreadCountPatch',
    'TransformersDynamicModulePatch',
    'TransformersHubKernelPatch',
    'TransformersIgnoreCheckImportsPatch',
    'UnslothDistributedFunctionPatch',
]
