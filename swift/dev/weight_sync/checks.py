# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pre-flight guards for quantization-aware weight synchronization.

The refit itself refuses to guess, but by the time it runs the run has already
paid for the model build. These checks run at cluster assembly, where the HF
config, the tuner and the rollout config are all in hand, and reject the two
combinations the sync provably cannot serve:

* ``sleep_level=2`` on a DeepSeek-V4 whose routed experts are MXFP4. A level-2
  sleep discards the engine parameters (vLLM saves only named_buffers across
  it); the experts would come back as uninitialized memory and the rollout
  would emit garbage from the first token.
* A LoRA adapter attached to the routed experts. A merged delta is ~1e-4
  relative while the E2M1 grid step is ~40% of the mean weight magnitude, so
  re-packing replaces the update with quantization noise and the rollout
  silently stays on the unmodified experts.

Both are decided from the checkpoint config before any GPU work: the guard is
about the architecture and the adapter targets, not about what a particular
engine happened to build.
"""
from typing import Any, Optional

#: DeepSeek-V4 target names that only exist on the MLP / routed-expert stack.
#: The attention projections (`linear_q_down_proj`, `linear_q_up_proj`,
#: `linear_kv_proj`, `linear_proj`) are deliberately absent: an attention-only
#: adapter is the supported LoRA shape on this model.
_EXPERT_TARGET_NAMES = frozenset({'all-linear', 'linear_fc1', 'linear_fc2'})


def _config_attr(hf_config: Any, name: str) -> Any:
    if hf_config is None:
        return None
    if isinstance(hf_config, dict):
        return hf_config.get(name)
    return getattr(hf_config, name, None)


def _quant_method(hf_config: Any) -> Optional[str]:
    quant_config = _config_attr(hf_config, 'quantization_config')
    if quant_config is None:
        return None
    if isinstance(quant_config, dict):
        return quant_config.get('quant_method')
    return getattr(quant_config, 'quant_method', None)


def _targets_routed_experts(target_modules: Any) -> bool:
    """Whether the adapter's target set covers linears that live on routed experts.

    ``all-linear`` is the default and adapts everything, including the experts.
    ``linear_fc1`` / ``linear_fc2`` are Megatron's expert MLP names; the
    attention projections have distinct names on this architecture.
    """
    if target_modules is None:
        return True
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    return any(str(name).rsplit('.', 1)[-1] in _EXPERT_TARGET_NAMES for name in target_modules)


def check_quantized_weight_sync(hf_config: Any,
                                model_type: Optional[str],
                                tuner_config: Any,
                                rollout_config: Any) -> None:
    """Refuse the engine/training combinations the quantized refit cannot serve.

    No-op unless the rollout backend is vLLM and the policy is DeepSeek-V4, so
    it is safe to call unconditionally from cluster assembly. ``hf_config`` is
    the checkpoint's HF config (the staged template already carries it);
    ``model_type`` falls back to the same object when the caller has no explicit
    value.
    """
    backend = getattr(rollout_config, 'rollout_sampler', 'vllm')
    if backend != 'vllm':
        return
    model_type = model_type or _config_attr(hf_config, 'model_type')
    if model_type != 'deepseek_v4':
        return

    expert_dtype = _config_attr(hf_config, 'expert_dtype')
    sleep_level = getattr(rollout_config, 'sleep_level', None)
    if expert_dtype == 'fp4' and sleep_level == 2:
        raise NotImplementedError(
            'sleep_level=2 discards the rollout engine parameters (vLLM only saves named_buffers across a '
            'level-2 sleep). DeepSeek-V4 serves its routed experts as MXFP4, which the level-2 wake brings '
            'back as uninitialized memory, so the rollout emits garbage from the first token. Use '
            'sleep_level=1, which offloads the engine weights to host memory and restores them on wake.')

    tuner = getattr(tuner_config, 'tuner', None)
    if tuner != 'lora':
        return
    # A full-precision engine loads the BF16 stream as-is; an adapter on the
    # experts is fine there. Only a quantized engine is refused.
    if _quant_method(hf_config) not in ('fp8', 'mxfp4') and expert_dtype not in ('fp4', 'fp8'):
        return
    if _targets_routed_experts(getattr(tuner_config, 'target_modules', None)):
        raise NotImplementedError(
            'LoRA on DeepSeek-V4 routed experts cannot be synchronized into a quantized rollout engine: a '
            'merged delta is far below the MXFP4 grid step, so re-packing replaces the update with '
            'quantization noise and the rollout silently stays on the unmodified experts. Restrict the '
            'adapter to the attention projections, e.g. '
            'target_modules=[linear_q_down_proj, linear_q_up_proj, linear_kv_proj, linear_proj].')


__all__ = ['check_quantized_weight_sync']
