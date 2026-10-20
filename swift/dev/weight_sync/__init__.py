# Copyright (c) ModelScope Contributors. All rights reserved.
"""Quantization-aware weight synchronization for RL rollout engines.

The rollout engine may serve a quantized checkpoint (block-FP8 dense layers,
MXFP4 routed experts on DeepSeek-V4) while the trainer holds BF16 weights. A
plain ``load_weights`` then either fails on the kernel layouts the engine
rewrote at startup or silently drops the update. This package quantizes the
outgoing stream on the fly and brackets the bucketed transfer with the
stage/process pair each quantization scheme needs.

Package layout (all vLLM imports are lazy, so importing this package costs
nothing when vLLM is absent or the model is not quantized):

* :mod:`swift.dev.weight_sync.fp8_kernel` -- blockwise BF16 -> FP8 cast.
* :mod:`swift.dev.weight_sync.mxfp4_kernel` -- BF16 -> MXFP4 (E2M1) packing.
* :mod:`swift.dev.weight_sync.vllm_fp8` -- ``Fp8LinearMethod`` / ``Fp8MoEMethod`` refit.
* :mod:`swift.dev.weight_sync.vllm_mxfp4` -- ``Mxfp4MoEMethod`` (DeepSeek-V4) refit.
* :mod:`swift.dev.weight_sync.manager` -- the entry points the weight-sync paths call.
* :mod:`swift.dev.weight_sync.worker_extension` -- twinkle's vLLM worker extension
  subclass that routes the receive path through the quantizer.
* :mod:`swift.dev.weight_sync.checks` -- the engine/training combinations that
  cannot be synced, refused before any GPU work is spent.

Ported from the legacy ``swift.rlhf_trainers.quant_sync`` package (see the
DeepSeek-V4 BestPractices doc); the implementations are unchanged apart from
dev-local imports.
"""
from .checks import check_quantized_weight_sync
from .fp8_kernel import FP8_MAX, FP8_MIN, is_triton_available, scaled_fp8_blockwise
from .manager import (
                      apply_vllm_quant_patches,
                      get_module_from_param_name,
                      is_fp8_weight,
                      is_quantized_sync_model,
                      load_quanted_weights,
                      prepare_quanted_weights_for_loading,
                      process_quanted_weights_after_loading,
                      quant_weights,
)
from .mxfp4_kernel import dequantize_mxfp4_e2m1, quantize_mxfp4_e2m1
from .vllm_fp8 import copy_param_subclass_attrs, process_fp8_weights_after_loading, stage_fp8_params_for_loading
from .vllm_mxfp4 import (
                      check_mxfp4_experts_are_syncable,
                      has_mxfp4_experts,
                      is_deepseek_v4_model,
                      iter_deepseek_v4_weights,
                      process_mxfp4_moe_weights_after_loading,
                      stage_mxfp4_moe_params_for_loading,
                      trainable_mxfp4_expert_modules,
)

#: The engine-arg value that installs :class:`QuantSyncWorkerExtension`; kept as a
#: string so building a sampler never imports twinkle's worker extension eagerly.
QUANT_SYNC_WORKER_EXTENSION = 'swift.dev.weight_sync.worker_extension.QuantSyncWorkerExtension'

__all__ = [
    'FP8_MAX',
    'FP8_MIN',
    'QUANT_SYNC_WORKER_EXTENSION',
    'apply_vllm_quant_patches',
    'check_mxfp4_experts_are_syncable',
    'check_quantized_weight_sync',
    'copy_param_subclass_attrs',
    'dequantize_mxfp4_e2m1',
    'get_module_from_param_name',
    'has_mxfp4_experts',
    'is_deepseek_v4_model',
    'is_fp8_weight',
    'is_quantized_sync_model',
    'is_triton_available',
    'iter_deepseek_v4_weights',
    'load_quanted_weights',
    'prepare_quanted_weights_for_loading',
    'process_fp8_weights_after_loading',
    'process_mxfp4_moe_weights_after_loading',
    'process_quanted_weights_after_loading',
    'quant_weights',
    'quantize_mxfp4_e2m1',
    'scaled_fp8_blockwise',
    'stage_fp8_params_for_loading',
    'stage_mxfp4_moe_params_for_loading',
    'trainable_mxfp4_expert_modules',
]
