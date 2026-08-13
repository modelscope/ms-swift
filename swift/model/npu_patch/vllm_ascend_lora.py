# Copyright (c) ModelScope Contributors. All rights reserved.
"""vLLM-Ascend LoRA compatibility patches."""
from __future__ import annotations

import torch

from swift.utils.logger import get_logger

_QWEN3_5_QKVZ_SUFFIX = '.linear_attn.in_proj_qkvz'
logger = get_logger()


def validate_vllm_ascend_lora_training(model, args) -> None:
    """Reject training configurations whose expert LoRA cannot run in vLLM-Ascend."""
    if args.tuner_type != 'lora' or not args.vllm_enable_lora or not model.model_info.is_moe_model:
        return

    tuner = getattr(model, 'base_model', None)
    targeted_parameter_names = getattr(tuner, 'targeted_parameter_names',
                                       getattr(args, 'target_parameters', None) or [])
    routed_expert_parameters = sorted(name for name in targeted_parameter_names if 'experts' in name.split('.'))
    if not routed_expert_parameters:
        return

    raise ValueError('vLLM-Ascend does not support LoRA on fused routed experts, but the training model targets '
                     f'these expert parameters: {routed_expert_parameters}. With `vllm_enable_lora=true`, rollout '
                     'would omit their LoRA updates and diverge from training. Set `vllm_enable_lora=false` or '
                     'remove the routed-expert entries from `target_parameters`.')


def validate_vllm_ascend_megatron_lora_training(models, args) -> None:
    """Reject Megatron expert LoRA that vLLM-Ascend cannot apply during rollout."""
    if args.tuner_type != 'lora' or not args.vllm_enable_lora or not args.model_info.is_moe_model:
        return

    routed_expert_modules = sorted({
        name.split('.lora_', 1)[0]
        for model in models
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and 'experts' in name.split('.') and '.lora_' in name
    })
    if not routed_expert_modules:
        return

    raise ValueError('vLLM-Ascend does not support LoRA on fused routed experts, but the Megatron training model has '
                     f'trainable expert LoRA modules: {routed_expert_modules}. With `vllm_enable_lora=true`, rollout '
                     'would omit their LoRA updates and diverge from training. Set `vllm_enable_lora=false` or choose '
                     '`target_modules` that do not match routed-expert layers.')


def _exclude_unsupported_fused_moe_lora_modules(
    model,
    supported_modules: list[str],
    *,
    fused_moe_cls=None,
) -> list[str]:
    """Exclude routed experts whose LoRA runtime is CUDA-only in vLLM.

    vLLM adds every ``FusedMoE`` suffix to the LoRA manager's supported module
    list. Its v0.18 wrapper builds a TritonExperts kernel during manager
    initialization, but that kernel is unavailable on NPU. The Transformers
    LoRA path still supports ordinary linear modules around the MoE block
    (for example attention and shared experts), so only the fused routed
    expert suffixes must be removed.
    """
    if fused_moe_cls is None:
        from vllm.model_executor.layers.fused_moe import FusedMoE
        fused_moe_cls = FusedMoE

    fused_moe_suffixes = {
        name.rsplit('.', 1)[-1]
        for name, module in model.named_modules() if isinstance(module, fused_moe_cls)
    }
    if not fused_moe_suffixes:
        return supported_modules
    return [module for module in supported_modules if module not in fused_moe_suffixes]


def _patch_vllm_ascend_fused_moe_lora_modules() -> None:
    """Keep vLLM's CUDA-only FusedMoE LoRA wrapper out of the NPU path."""
    try:
        import vllm.lora.model_manager as model_manager
        import vllm.lora.utils as lora_utils
    except ImportError:
        return

    origin_get_supported = model_manager.get_supported_lora_modules
    if getattr(origin_get_supported, '_swift_npu_fused_moe_lora_patched', False):
        return

    def get_supported_lora_modules(model):
        supported_modules = origin_get_supported(model)
        filtered_modules = _exclude_unsupported_fused_moe_lora_modules(model, supported_modules)
        if len(filtered_modules) != len(supported_modules):
            logger.warning_once(
                'vLLM-Ascend does not support LoRA on fused routed experts; those modules will be skipped. '
                'LoRA on attention and other supported linear modules remains enabled.')
        return filtered_modules

    get_supported_lora_modules._swift_origin = origin_get_supported
    get_supported_lora_modules._swift_npu_fused_moe_lora_patched = True
    model_manager.get_supported_lora_modules = get_supported_lora_modules
    lora_utils.get_supported_lora_modules = get_supported_lora_modules


def _expand_qwen3_5_qkvz_lora(
    lora_a: list[torch.Tensor | None],
    lora_b: list[torch.Tensor | None],
    output_sizes: list[int],
    *,
    tp_size: int = 1,
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
    """Expand Qwen3.5's logical ``qkv + z`` LoRAs into ``q + k + v + z``.

    Qwen3.5 checkpoints expose two logical modules, ``in_proj_qkv`` and
    ``in_proj_z``, while vLLM's merged runtime layer has four physical output
    slices. The fused qkv LoRA shares one A matrix and concatenates q/k/v in
    its B matrix, so A is reused and B is split along its output dimension.
    """
    if len(lora_a) != 2 or len(lora_b) != 2 or len(output_sizes) != 4:
        raise RuntimeError('Qwen3.5 in_proj_qkvz LoRA expects 2 logical adapters and 4 output slices, '
                           f'got len(lora_a)={len(lora_a)}, len(lora_b)={len(lora_b)}, '
                           f'output_sizes={output_sizes}.')

    qkv_a, z_a = lora_a
    qkv_b, z_b = lora_b
    if (qkv_a is None) != (qkv_b is None) or (z_a is None) != (z_b is None):
        raise RuntimeError('Qwen3.5 in_proj_qkvz LoRA A/B presence does not match.')

    # vLLM creates its profiling adapter from the first two physical buffers,
    # although the packed-module mapping contains the two logical qkv/z names.
    # Those tensors are all-zero and have partition-local q/k shapes. Replace
    # them with correctly shaped zero q/k/v/z tensors for warmup.
    output_slices = [size // tp_size for size in output_sizes]
    is_profile_adapter = (
        qkv_a is not None and z_a is not None and qkv_b is not None and z_b is not None
        and qkv_b.shape[0] == output_slices[0] and z_b.shape[0] == output_slices[1]
        and not torch.count_nonzero(qkv_b).item() and not torch.count_nonzero(z_b).item())
    if is_profile_adapter:
        expanded_b = [qkv_b.new_zeros((size, qkv_b.shape[1])) for size in output_sizes[:3]]
        expanded_b.append(z_b.new_zeros((output_sizes[3], z_b.shape[1])))
        return [qkv_a, qkv_a, qkv_a, z_a], expanded_b

    expanded_a: list[torch.Tensor | None]
    expanded_b: list[torch.Tensor | None]
    if qkv_b is None:
        expanded_a = [None, None, None]
        expanded_b = [None, None, None]
    else:
        expected_qkv_size = sum(output_sizes[:3])
        if qkv_b.shape[0] != expected_qkv_size:
            raise RuntimeError('Qwen3.5 in_proj_qkv LoRA B has an unexpected output dimension: '
                               f'expected {expected_qkv_size}, got {qkv_b.shape[0]}.')
        expanded_a = [qkv_a, qkv_a, qkv_a]
        expanded_b = list(qkv_b.split(output_sizes[:3], dim=0))

    if z_b is not None and z_b.shape[0] != output_sizes[3]:
        raise RuntimeError('Qwen3.5 in_proj_z LoRA B has an unexpected output dimension: '
                           f'expected {output_sizes[3]}, got {z_b.shape[0]}.')
    expanded_a.append(z_a)
    expanded_b.append(z_b)
    return expanded_a, expanded_b


def patch_vllm_ascend_lora_runtime() -> None:
    """Patch vLLM-Ascend's merged LoRA wrapper for Qwen3.5 GDN projections."""
    _patch_vllm_ascend_fused_moe_lora_modules()
    try:
        from vllm_ascend.lora.utils import AscendMergedColumnParallelLinearWithLoRA
    except (ImportError, AttributeError):
        return

    wrapper_cls = AscendMergedColumnParallelLinearWithLoRA
    if getattr(wrapper_cls, '_swift_qwen3_5_qkvz_lora_patched', False):
        return
    origin_set_lora = wrapper_cls.set_lora

    def set_lora(self, index, lora_a, lora_b):
        prefix = getattr(self.base_layer, 'prefix', '')
        if prefix.endswith(_QWEN3_5_QKVZ_SUFFIX):
            lora_a, lora_b = _expand_qwen3_5_qkvz_lora(
                lora_a,
                lora_b,
                self.base_layer.output_sizes,
                tp_size=self.tp_size,
            )
        return origin_set_lora(self, index, lora_a, lora_b)

    set_lora._swift_origin = origin_set_lora
    wrapper_cls.set_lora = set_lora
    wrapper_cls._swift_qwen3_5_qkvz_lora_patched = True


__all__ = [
    'patch_vllm_ascend_lora_runtime',
    'validate_vllm_ascend_lora_training',
    'validate_vllm_ascend_megatron_lora_training',
]
