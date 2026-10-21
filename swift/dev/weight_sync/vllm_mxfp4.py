# Copyright (c) ModelScope Contributors. All rights reserved.
"""Refit support for vLLM's ``Mxfp4MoEMethod`` (DeepSeek-V4 routed experts).

refer: verl (``verl/utils/vllm/vllm_fp4_utils.py``).

Reloading these experts means undoing the kernel's weight conversion: the refit
stream carries checkpoint-layout tensors while the live parameters hold whatever
layout the selected backend rewrote them into. Each expert param is handed a
checkpoint-layout buffer to absorb the reload, then the backend's conversion is
replayed and folded back into the live storage, so the addresses the CUDA graph
captured stay put.

Backends differ only in how much of that buffer has to be a real allocation. A
param whose kernel layout spans the same bytes as the checkpoint one -- the
expert weights under both DeepGEMM and Marlin -- stages as a reinterpret of its
own storage and costs nothing. Only the block scales, which change element count,
need an allocation, and those are a sixteenth of the weights.

``manager.py`` is the entry point that drives these.
"""
from typing import Iterable, List, Optional, Tuple
from unittest.mock import patch

import torch

from swift.dev.utils.logger import get_logger

logger = get_logger()

_MXFP4_SF_BLOCK = 32
_MXFP4_LIVE_ATTR = '_swift_mxfp4_live_params'


def is_deepseek_v4_model(model) -> bool:
    """Whether ``model`` is a DeepSeek-V4, checked through the usual config chain."""
    if model is None:
        return False

    for obj in (model, getattr(model, 'config', None), getattr(model, 'hf_config', None)):
        if obj is not None and getattr(obj, 'model_type', None) is not None:
            return obj.model_type == 'deepseek_v4'

    text_config = getattr(getattr(model, 'config', None), 'text_config', None)
    return getattr(text_config, 'model_type', None) == 'deepseek_v4'


def iter_deepseek_v4_weights(weights: Iterable[Tuple[str, torch.Tensor]]) -> Iterable[Tuple[str, torch.Tensor]]:
    """Pass the refit stream through untouched apart from a dtype reinterpret.

    A DSv4 checkpoint already ships quantized experts, so unlike the BF16 path
    there is nothing to quantize here. The expert tensors only need to be seen as
    the raw ``uint8`` byte layout that ``Mxfp4MoEMethod`` allocated.
    """
    for name, weight in weights:
        if '.experts.' in name and weight.dtype in (torch.int8, torch.float8_e8m0fnu):
            weight = weight.view(torch.uint8)
        yield name, weight


def _is_mxfp4_fused_moe_module(module) -> bool:
    try:
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
    except ImportError:
        return False

    return isinstance(module, RoutedExperts) and isinstance(getattr(module, 'quant_method', None), Mxfp4MoEMethod)


def trainable_mxfp4_expert_modules(model) -> List[str]:
    """LoRA-adapted routed-expert modules on the training side.

    Matches on the module path rather than a class so it covers both the mcore
    grouped-expert modules and an HF expert list. ``shared_experts`` are excluded:
    DeepSeek-V4 serves those as block-FP8, which a sync can requantize exactly.
    """
    adapted = []
    for name, module in model.named_modules():
        if 'experts' not in name or 'shared_experts' in name:
            continue
        if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
            adapted.append(name)
    return adapted


def check_mxfp4_experts_are_syncable(model,
                                     tuner_type: Optional[str],
                                     engine_model=None,
                                     sleep_level: Optional[int] = None) -> None:
    """Raise on the two engine/training combinations the MXFP4 expert sync cannot serve.

    The sync now always re-quantizes and carries the routed experts (see
    ``_quant_deepseek_v4_weights``), so full-parameter training is supported and needs no
    guard here. Two cases still fail:

    * ``sleep_level=2`` discards the engine parameters outright (vLLM saves only
      named_buffers across it). This is refused conservatively: it has not been validated
      against the always-carry sync, and it historically brought the experts back as
      uninitialized memory.
    * a LoRA adapter attached to the experts, detected per module below. A merged delta
      cannot be transmitted: it falls below the FP4 grid step, so the rollout would
      silently stay on the unmodified experts.

    ``engine_model`` is the rollout model, and it is what decides whether any of this
    applies: ``model`` (the training side) holds dequantized BF16 experts whichever
    dtype the engine chose, so it cannot tell MXFP4 experts from the block-FP8 ones a
    Flash-Base checkpoint builds -- and those sync exactly.
    """
    if not has_mxfp4_experts(engine_model):
        return

    if sleep_level == 2:
        raise NotImplementedError(
            'sleep_level=2 discards the rollout engine parameters (vLLM only saves named_buffers across a level-2 '
            'sleep). This is not validated with the DeepSeek-V4 MXFP4 expert sync and historically brought the '
            'experts back as uninitialized memory, so the rollout emitted garbage from the first token. Use '
            '--sleep_level 1, which offloads the engine weights to host memory and restores them on wake.')

    if tuner_type in ('lora', 'lora_llm'):
        # A merged LoRA delta cannot ride the MXFP4 grid: E2M1's step is ~40% of the mean
        # weight magnitude while a merged delta is ~1e-4 relative, so re-packing replaces
        # the update with quantization noise and the rollout silently stays on the
        # unmodified experts (rollout_correction/kl never comes down, with no error).
        adapted = trainable_mxfp4_expert_modules(model)
        if adapted:
            shown = ', '.join(adapted[:3]) + (' ...' if len(adapted) > 3 else '')
            raise NotImplementedError(
                f'LoRA is attached to {len(adapted)} routed-expert module(s) ({shown}), but the rollout engine '
                'serves these routed experts as MXFP4, where a merged delta cannot be synchronized. Restrict the '
                'adapter to the attention projections, e.g. --target_modules linear_q_down_proj linear_q_up_proj '
                'linear_kv_proj linear_proj, rather than all-linear.')
        return

    # Full-parameter training updates the routed experts, and the sync carries them:
    # ``mxfp4_kernel.quantize_mxfp4_e2m1`` (a port of Megatron-Bridge's
    # ``quantize_mxfp4_e2m1_like_scale``, verified bit-identical against it on real expert
    # tensors) re-packs each expert onto the grid its checkpoint scale describes, and
    # ``prepare_quanted_weights_for_loading`` stages the MXFP4 modules so ``process`` writes
    # them back into the engine's kernel layout. The cost is a full re-quantization of every
    # expert on every step (~1.5s/step/rank measured, <0.2% of a GRPO step).
    logger.info_once(f'Full-parameter training (tuner_type={tuner_type!r}) will sync the DeepSeek-V4 routed experts '
                     'by re-quantizing them to MXFP4 each step. Watch rollout_correction/kl to confirm the rollout '
                     'policy tracks training.')


def has_mxfp4_experts(model) -> bool:
    """Whether the built rollout engine serves its routed experts as MXFP4.

    Asked of the engine rather than inferred from names, because DeepSeek-V4 ships two
    expert dtypes and only one of them is a problem: ``expert_dtype=fp4`` (Flash) builds
    ``Mxfp4MoEMethod`` experts whose grid cannot carry an update, while
    ``expert_dtype=fp8`` (Flash-Base) builds block-FP8 experts that a UE8M0 requantize
    reproduces exactly and can therefore be synced. A training-side module-path match
    cannot distinguish the two and would refuse Flash-Base for no reason. The dtype is a
    whole-model property, so one boolean answers it for every layer.
    """
    if model is None:
        return False
    return any(_is_mxfp4_fused_moe_module(module) for module in model.modules())


def _refittable_mxfp4_backends() -> tuple:
    """Backends whose weight conversion can be replayed onto live storage.

    Two properties qualify a backend. Its conversion must be a pure function of
    the checkpoint layout, so replaying it after a reload reproduces the same
    inference layout. And it must publish the result through
    ``replace_parameter``, which is the hook ``_replace_parameter_in_place``
    intercepts to redirect the write into the captured storage.

    ``Mxfp4MoEMethod._setup_kernel`` gates that publication on
    ``mxfp4_backend not in TRITON_BACKENDS``, so every non-Triton backend routes
    through the hook. Triton is the exclusion the upstream branch exists for: it
    assigns the weights as non-Parameter wrappers and parks the scales on the
    quant method, so the buffers its kernel reads are not reachable as parameters
    at all.

    The FlashInfer TRTLLM pair is listed on top of verl's set after reading its
    branch in ``convert_weight_to_mxfp4_moe_kernel_format``: it swaps w1/w3,
    interleaves, and applies permutations whose indices are derived from tensor
    *shape* alone (upstream comment: "Permute indices depend only on shape"),
    returning fresh tensors without touching ``layer``. That makes it replayable.
    Its block scales come back as ``float8_e4m3fn`` over the checkpoint's ``uint8``
    bytes -- same shape, same one byte per element -- which is why
    ``_fold_into_live_param`` compares byte footprint rather than dtype.
    """
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

    return (
        Mxfp4MoeBackend.DEEPGEMM_MXFP4,
        Mxfp4MoeBackend.MARLIN,
        Mxfp4MoeBackend.BATCHED_MARLIN,
        Mxfp4MoeBackend.AITER_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
    )


def _mxfp4_checkpoint_layout(module) -> dict:
    """Layout of the expert params before kernel post-processing rewrites them.

    Mirrors ``Mxfp4MoEMethod.create_weights``. These are not guesses: vLLM
    re-asserts the same shapes at the top of ``_setup_kernel``, so a refit that
    hands back anything else fails loudly there. The third element is the
    ``quant_method`` tag ``RoutedExperts.weight_loader`` dispatches on, which
    ``replace_parameter`` does not carry over and must be reattached.
    """
    quant_method = module.quant_method
    num_experts = quant_method.num_experts
    intermediate = quant_method.intermediate_size
    hidden = quant_method.hidden_size
    return {
        'w13_weight': ((num_experts, 2 * intermediate, hidden // 2), torch.uint8, None),
        'w2_weight': ((num_experts, hidden, intermediate // 2), torch.uint8, None),
        'w13_weight_scale': ((num_experts, 2 * intermediate, hidden // _MXFP4_SF_BLOCK), torch.uint8, 'block'),
        'w2_weight_scale': ((num_experts, hidden, intermediate // _MXFP4_SF_BLOCK), torch.uint8, 'block'),
        # Listed unconditionally: models without MoE bias never registered these
        # and are skipped when the parameter turns up missing. Leaving them out
        # would instead let Marlin's bias permute fall through to the real
        # ``replace_parameter`` and quietly move the buffer.
        'w13_bias': ((num_experts, 2 * intermediate), torch.bfloat16, None),
        'w2_bias': ((num_experts, hidden), torch.bfloat16, None),
    }


def _mxfp4_staging_data(param: torch.nn.Parameter, shape, dtype: torch.dtype) -> torch.Tensor:
    """Checkpoint-layout tensor for ``load_weights`` to write into.

    This must be storage *distinct* from the live parameter. ``_process`` replays
    the backend's ``convert_weight_to_mxfp4_moe_kernel_format`` with this buffer as
    the checkpoint-layout input and folds the kernel-layout result back into the
    live param. For a permutation backend (FlashInfer TRTLLM) that conversion is not
    the identity, so if the staging buffer aliased the live storage the replay would
    permute already-permuted bytes -- corrupting the experts even when no new weights
    were loaded. A refit that skips some experts must also leave those untouched,
    which aliasing cannot guarantee.

    The buffer is initialised from the live storage (reinterpreted to the checkpoint
    dtype/shape, which is always byte-compatible here) so that an expert the reload
    stream happens to skip still replays from real data rather than zeros. The copy
    costs one extra expert-sized allocation per layer for the duration of the refit;
    that is the price of correctness for a non-identity conversion.
    """
    data = param.data
    shape = torch.Size(shape)
    if data.dtype == dtype and data.shape == shape:
        return data.clone()
    if data.is_contiguous() and data.numel() * data.element_size() == shape.numel() * dtype.itemsize:
        return data.flatten().view(dtype).reshape(shape).clone()
    # Zero rather than empty: a param the refit stream happens to skip then
    # collapses the layer's output instead of feeding the kernel whatever was left
    # in the freed memory.
    return torch.zeros(shape, dtype=dtype, device=param.device)


def _stage_mxfp4_moe_params(module) -> None:
    """Expose checkpoint-layout buffers without moving any live storage.

    Each expert param is handed a buffer in the layout its ``weight_loader``
    expects, preferring a reinterpret of the param's own storage so the reload
    lands where the kernel reads. ``_process_mxfp4_moe_params`` then replays the
    backend's conversion and folds the result back into the live storage.
    """
    live = {}
    for name, (shape, dtype, scale_kind) in _mxfp4_checkpoint_layout(module).items():
        param = getattr(module, name, None)
        if not isinstance(param, torch.nn.Parameter):
            continue
        live[name] = param

        staged = torch.nn.Parameter(_mxfp4_staging_data(param, shape, dtype), requires_grad=False)
        staged.weight_loader = module.weight_loader
        if scale_kind is not None:
            staged.quant_method = scale_kind
        setattr(module, name, staged)

    setattr(module, _MXFP4_LIVE_ATTR, live)


def _fold_into_live_param(layer, param_name: str, param: torch.nn.Parameter, new_data: torch.Tensor) -> None:
    """Copy a re-derived tensor into ``param``'s storage, byte-wise if dtypes differ.

    A backend may hand back the same bytes under a different dtype: the FlashInfer
    TRTLLM path returns block scales as ``float8_e4m3fn`` while the checkpoint -- and
    therefore the staged buffer -- holds them as ``uint8``, same shape and same one
    byte per element. Rejecting that would refuse a perfectly valid refit, so the
    comparison is on byte footprint and the copy goes through a ``uint8`` view.

    Shape is still required to match. A backend that changes it is not writing the
    same tensor back, and silently reshaping would corrupt the layer.
    """
    new_bytes = new_data.numel() * new_data.element_size()
    live_bytes = param.numel() * param.element_size()

    if new_data.shape != param.shape or new_bytes != live_bytes:
        raise RuntimeError(f'mxfp4 refit re-derived {param_name} as {tuple(new_data.shape)}/{new_data.dtype} '
                           f'({new_bytes} bytes), but the live parameter is {tuple(param.shape)}/'
                           f'{param.dtype} ({live_bytes} bytes); its storage cannot be updated in place.')

    if new_data.data_ptr() == param.data_ptr():
        # The conversion reinterpreted the staged buffer in place; nothing to copy.
        pass
    elif new_data.dtype == param.dtype:
        param.data.copy_(new_data)
    else:
        # Same bytes under a different dtype: move the raw bytes and leave the live
        # parameter's dtype alone, because that is what the kernel reads through.
        param.data.view(torch.uint8).copy_(new_data.reshape(-1).view(torch.uint8).reshape(param.shape))

    setattr(layer, param_name, param)


def _replace_parameter_in_place(layer, param_name: str, new_data, prefer_copy: bool = False) -> None:
    """Fold post-processing output into the parameter the CUDA graph captured.

    Reinstating the live parameter here rather than after
    ``process_weights_after_loading`` returns is deliberate: the tail of
    ``_setup_kernel`` rebuilds ``moe_quant_config`` and ``moe_kernel`` from
    whatever is on the layer at that moment, and those cache tensor references.
    Swapping back later would leave the kernel pointing at the temporaries.
    """
    from vllm.model_executor.utils import replace_parameter

    live = getattr(layer, _MXFP4_LIVE_ATTR, None) or {}
    param = live.pop(param_name, None)
    if param is None or new_data is None:
        return replace_parameter(layer, param_name, new_data, prefer_copy)

    if isinstance(new_data, torch.nn.Parameter):
        new_data = new_data.data

    _fold_into_live_param(layer, param_name, param, new_data)


def _process_mxfp4_moe_params(module) -> None:
    from vllm.model_executor.layers.quantization import mxfp4 as vllm_mxfp4

    with patch.object(vllm_mxfp4, 'replace_parameter', _replace_parameter_in_place):
        module.quant_method.process_weights_after_loading(module)

    # Every staged param is consumed through the patched replace_parameter, so a
    # leftover means post-processing took a path that assigns weights some other
    # way and the live storage was never refreshed.
    leftover = getattr(module, _MXFP4_LIVE_ATTR, None) or {}
    if hasattr(module, _MXFP4_LIVE_ATTR):
        delattr(module, _MXFP4_LIVE_ATTR)
    if leftover:
        raise RuntimeError(f'mxfp4 refit left {sorted(leftover)} un-reinstated; every backend listed in '
                           '_refittable_mxfp4_backends is expected to route each expert param through '
                           'replace_parameter.')


def stage_mxfp4_moe_params_for_loading(model) -> List[torch.nn.Module]:
    """Hand ``load_weights`` checkpoint-layout expert buffers.

    Returns the staged modules for ``process_mxfp4_moe_weights_after_loading``. A
    model with no mxfp4 experts yields an empty list, which makes this safe to
    call unconditionally.
    """
    staged_modules = []
    backends = set()
    supported = None
    for module in model.modules():
        if not _is_mxfp4_fused_moe_module(module):
            continue

        if supported is None:
            supported = _refittable_mxfp4_backends()

        backend = getattr(module.quant_method, 'mxfp4_backend', None)
        if backend not in supported:
            # Refusing beats skipping: an unsupported backend would quietly load
            # checkpoint-layout data into rewritten parameters, and the rollout
            # would drift with nothing pointing at the cause.
            raise NotImplementedError(f'mxfp4 MoE refit does not support the {backend} backend selected by '
                                      f'{type(module).__name__}. Supported backends: '
                                      f"{', '.join(str(b) for b in supported)}.")

        _stage_mxfp4_moe_params(module)
        staged_modules.append(module)
        backends.add(backend)

    if staged_modules:
        backend_names = ', '.join(sorted(str(b) for b in backends))
        logger.info(f'Staged {len(staged_modules)} mxfp4 MoE modules for in-place refit (backends: {backend_names}).')
    return staged_modules


def process_mxfp4_moe_weights_after_loading(modules) -> None:
    """Repack the loaded experts into the kernel layout inside the live storage."""
    for module in modules:
        _process_mxfp4_moe_params(module)
