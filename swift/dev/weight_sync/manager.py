# Copyright (c) ModelScope Contributors. All rights reserved.
"""Entry point for refitting a quantized vLLM engine during RL rollout.

Everything the weight-sync paths call lives here: startup patching
(``apply_vllm_quant_patches``), the BF16 -> FP8 quantization of the outgoing
weight stream (``quant_weights``), the load itself (``load_quanted_weights``), and
the prepare/finalize pair that brackets a bucketed transfer
(``prepare_quanted_weights_for_loading`` / ``process_quanted_weights_after_loading``).

The layout surgery each quantization scheme needs is delegated to a sibling
module: ``vllm_fp8`` for ``Fp8LinearMethod`` / ``Fp8MoEMethod``, and ``vllm_mxfp4``
for ``Mxfp4MoEMethod`` (DeepSeek-V4 routed experts). Each exposes a stage/process
pair that returns an empty result when the model has no layer it owns, so the
dispatch below stays unconditional.

refer: verl (``verl/utils/vllm/vllm_quant_utils.py``).
"""
import glob
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import json
import torch

from swift.dev.utils.logger import get_logger
from .fp8_kernel import scaled_fp8_blockwise
from .mxfp4_kernel import quantize_mxfp4_e2m1
from .vllm_fp8 import build_fp8_method_patchers, process_fp8_weights_after_loading, stage_fp8_params_for_loading
from .vllm_mxfp4 import (
    has_mxfp4_experts,
    is_deepseek_v4_model,
    process_mxfp4_moe_weights_after_loading,
    stage_mxfp4_moe_params_for_loading,
)

logger = get_logger()

_DEFAULT_WEIGHT_BLOCK_SIZE = (128, 128)


@dataclass
class _QuantState:
    # Caches which parameter names map to an FP8 weight, so the module lookup
    # behind that decision runs once per name rather than once per sync.
    seen_params: Set[str] = field(default_factory=set)
    fp8_param_names: Set[str] = field(default_factory=set)
    vllm_patches: List[Any] = field(default_factory=list)


_quant_state = _QuantState()


def is_quantized_sync_model(vllm_config) -> bool:
    """Whether weight sync into this engine has to go through quantization.

    True when vLLM built the model with a block-FP8 quant config
    """
    if vllm_config is None:
        return False

    quant_config = getattr(vllm_config, 'quant_config', None)
    if quant_config is not None:
        try:
            from vllm.model_executor.layers.quantization.fp8 import Fp8Config
            if isinstance(quant_config, Fp8Config):
                return True
        except ImportError:
            pass

    model_config = getattr(vllm_config, 'model_config', None)
    hf_config = getattr(model_config, 'hf_config', None)
    if hf_config is not None and getattr(hf_config, 'model_type', None) == 'deepseek_v4':
        return True

    return False


def _resolve_moe_classes():
    """Resolve the module classes that hold fused expert weights.

    vLLM 0.24 (MoE refactor, vllm-project/vllm#41184) turned ``FusedMoE`` from an
    ``nn.Module`` class into a factory *function* that builds a ``MoERunner``, and
    moved the fused expert tensors onto a ``RoutedExperts`` submodule. Calling
    ``isinstance(x, FusedMoE)`` when ``FusedMoE`` is a function raises TypeError,
    so the concrete classes are resolved once here and the callers duck-type.
    """
    stop_classes: List[type] = []
    weight_classes: List[type] = []

    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    except ImportError:
        FusedMoE = None

    if isinstance(FusedMoE, type):
        stop_classes.append(FusedMoE)
        weight_classes.append(FusedMoE)
    else:
        try:
            from vllm.model_executor.layers.fused_moe import RoutedExperts
            if isinstance(RoutedExperts, type):
                stop_classes.append(RoutedExperts)
                weight_classes.append(RoutedExperts)
        except ImportError:
            pass
        try:
            from vllm.model_executor.layers.fused_moe import MoERunner
            if isinstance(MoERunner, type):
                stop_classes.append(MoERunner)
        except ImportError:
            pass

    return tuple(stop_classes), tuple(weight_classes)


_MOE_STOP_CLASSES, _EXPERT_WEIGHT_CLASSES = _resolve_moe_classes()


def _is_expert_weight_module(module) -> bool:
    """Whether ``module`` directly owns fused expert weights.

    True for the pre-0.24 ``FusedMoE`` module and the post-0.24 ``RoutedExperts``
    module (both expose ``w13_weight`` / ``w2_weight``). Falls back to duck typing
    so the check keeps working even if the concrete classes cannot be imported.
    """
    if _EXPERT_WEIGHT_CLASSES and isinstance(module, _EXPERT_WEIGHT_CLASSES):
        return True
    return hasattr(module, 'w13_weight') and hasattr(module, 'w2_weight')


def _is_fused_moe_block(module) -> bool:
    """Whether traversal must stop here.

    The remaining name parts below a fused-MoE block (the expert index, the
    per-expert projection name) are not real submodules -- vLLM's fused weight
    loader consumes them -- so descending further would fail.
    """
    if _MOE_STOP_CLASSES and isinstance(module, _MOE_STOP_CLASSES):
        return True
    return hasattr(module, 'routed_experts') or _is_expert_weight_module(module)


def _resolve_expert_weight_module(module):
    """Return the submodule that actually owns the fused expert weights.

    On vLLM >= 0.24 the walked module is a ``MoERunner`` whose ``routed_experts``
    child holds the weights; on older vLLM the module already holds them.
    """
    routed_experts = getattr(module, 'routed_experts', None)
    if routed_experts is not None:
        return routed_experts
    return module


def _to_vllm_param_name(model, name: str) -> Optional[str]:
    """Translate an HF checkpoint name into the vLLM name, via the model's own mapper.

    ``load_weights`` maps every incoming name through ``hf_to_vllm_mapper`` before it
    resolves anything, so a lookup that skips the mapper is asking about a name the
    module tree does not have. For DeepSeek-V4 the two spellings differ structurally --
    ``layers.0.attn.wq_b.weight`` lives at ``model.layers.0.attn.wq_b``, ``embed.weight``
    at ``model.embed_tokens``, ``head.weight`` at ``lm_head`` -- so without this the walk
    fails on nearly every weight.

    Returns ``None`` when the mapper drops the name, which is vLLM's way of saying the
    weight is not loaded into this model at all.
    """
    mapper = getattr(model, 'hf_to_vllm_mapper', None)
    if mapper is None:
        return name
    try:
        return mapper._map_name(name)
    except Exception as e:  # noqa: BLE001 - private API; fall back to the raw name
        logger.debug(f'hf_to_vllm_mapper could not map {name!r} ({e}); using the name as given.')
        return name


def _walk_to_module(model, name: str):
    """Walk to the module owning ``name``, or None if the path does not exist."""
    path_parts = name.split('.')
    module_path = path_parts[:-1]

    packed_modules_mapping = getattr(model, 'packed_modules_mapping', None) or {}
    reversed_mapping = {
        original_name: fused_name
        for fused_name, original_names_list in packed_modules_mapping.items()
        for original_name in original_names_list
    }
    if module_path and module_path[-1] in reversed_mapping:
        module_path[-1] = reversed_mapping[module_path[-1]]

    current_module = model
    try:
        for part in module_path:
            if _is_fused_moe_block(current_module):
                return _resolve_expert_weight_module(current_module)
            elif isinstance(current_module, torch.nn.ModuleList):
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
    except (AttributeError, IndexError, ValueError):
        return None
    return _resolve_expert_weight_module(current_module)


def get_module_from_param_name(model, name: str):
    """Walk ``model`` to the module owning the parameter called ``name``.

    The incoming names are HF checkpoint names, so they are translated through the
    model's own ``hf_to_vllm_mapper`` first, and a packed module is reached through
    ``packed_modules_mapping`` in reverse (``q_proj`` lives on ``qkv_proj`` in vLLM).

    Raises when neither spelling resolves. A silent answer here is what made a
    DeepSeek-V4 refit produce NaN logprobs: 365 of 390 FP8 dense weights failed to
    resolve, ``is_fp8_weight`` therefore reported "not FP8", and the weights were
    streamed as unquantized BF16 with no scale into staged FP8 parameters.
    """
    mapped = _to_vllm_param_name(model, name)
    if mapped is None:
        # The model's own mapper drops this weight, so it is not loaded here at all.
        return None

    module = _walk_to_module(model, mapped)
    if module is None and mapped != name:
        # Some models already hand us vLLM names; try that spelling before giving up.
        module = _walk_to_module(model, name)
    if module is None:
        raise KeyError(f'Cannot resolve parameter {name!r} (mapped to {mapped!r}) to a module of '
                       f'{type(model).__name__}. Refusing to guess: treating it as un-quantized would '
                       'silently load a BF16 tensor into a quantized parameter.')
    return module


# Leaf names that can be FP8 weights: dense/per-expert weights end in "weight", while
# vLLM's packed fused-MoE expert params use projection names with no suffix. Mirrors
# verl's _FP8_CANDIDATE_LEAVES so a generic block-FP8 MoE (e.g. a future DeepSeek) whose
# experts export under the packed names is not silently skipped by the pre-screen below.
_FP8_CANDIDATE_LEAVES = frozenset({'gate_up_proj', 'down_proj'})


def is_fp8_weight(name: str, model) -> bool:
    """Whether the weight called ``name`` must be quantized before loading.

    Decided from the owning module's type and its *actual* parameter dtype rather
    than from the parameter name. A name-pattern heuristic would have to enumerate
    every DeepSeek-specific layer (MLA, the DSA indexer, linear attention) and
    silently mis-handle whatever it missed. The leaf-name pre-screen below is only a
    cheap filter before the module walk; the dtype check is what actually decides, so
    widening the pre-screen cannot mis-quantize a non-FP8 weight.
    """
    if name in _quant_state.fp8_param_names:
        return True
    if name in _quant_state.seen_params:
        return False

    _quant_state.seen_params.add(name)
    # Cheap pre-screen before the module walk: skip bias/embedding/norm leaves. Dense and
    # per-expert weights end in "weight"; packed fused-MoE experts use gate_up_proj/down_proj.
    leaf = name.rsplit('.', 1)[-1]
    if not (name.endswith('weight') or leaf in _FP8_CANDIDATE_LEAVES):
        return False

    from vllm.model_executor.layers.linear import LinearBase

    module = get_module_from_param_name(model, name)
    if module is None:
        # vLLM's own mapper drops this weight, so the engine never loads it.
        return False
    is_fp8_linear = (
        isinstance(module, LinearBase) and getattr(module, 'weight', None) is not None
        and module.weight.dtype == torch.float8_e4m3fn)
    is_fp8_moe = (
        _is_expert_weight_module(module) and module.w13_weight.dtype == torch.float8_e4m3fn
        and module.w2_weight.dtype == torch.float8_e4m3fn)

    if is_fp8_linear or is_fp8_moe:
        _quant_state.fp8_param_names.add(name)
        return True
    return False


def _get_weight_block_size(quant_config) -> Tuple[int, int]:
    block_size = getattr(quant_config, 'weight_block_size', None)
    if block_size is None and isinstance(quant_config, dict):
        block_size = quant_config.get('weight_block_size')
    if block_size is None:
        block_size = _DEFAULT_WEIGHT_BLOCK_SIZE
    return block_size


_CHECKPOINT_SCALES: Dict[str, Dict[str, torch.Tensor]] = {}


def load_checkpoint_scales(model_dir: str) -> Dict[str, torch.Tensor]:
    """The checkpoint's ``*.scale`` tensors, keyed by the ``*.weight`` they belong to.

    This is the criterion NVIDIA's Megatron-Bridge uses in
    ``requantize_hf_weight_scale_pairs``: a weight is block-quantized exactly when the
    checkpoint carries a sibling ``.scale`` for it, and the target geometry and dtype come
    from that source scale. It replaces walking the vLLM module tree, which cannot answer
    the question for DeepSeek-V4: the stream carries HF names while vLLM fuses
    ``attn.wq_a`` and ``attn.wkv`` into ``attn.fused_wqa_wkv`` and exposes no
    ``packed_modules_mapping`` for it, so 365 of 390 dense weights failed to resolve and
    were silently streamed unquantized.

    The scales are small (a few hundred KB in total), so they are read in full and cached
    per ``model_dir``.
    """
    cached = _CHECKPOINT_SCALES.get(model_dir)
    if cached is not None:
        return cached

    from safetensors import safe_open
    index_path = os.path.join(model_dir, 'model.safetensors.index.json')
    scales: Dict[str, torch.Tensor] = {}

    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)['weight_map']
        by_shard: Dict[str, List[str]] = {}
        for key, shard in weight_map.items():
            if key.endswith('.scale'):
                by_shard.setdefault(shard, []).append(key)
        shards = {os.path.join(model_dir, shard): keys for shard, keys in by_shard.items()}
    else:
        # An unsharded checkpoint carries no index, so its keys have to be read from the
        # shard itself. Treating that as "no scales" would stream every weight
        # unquantized into the engine's FP8 parameters.
        files = sorted(glob.glob(os.path.join(model_dir, '*.safetensors')))
        if not files:
            logger.warning(f'No safetensors under {model_dir!r}; cannot tell which weights are block-quantized.')
            _CHECKPOINT_SCALES[model_dir] = scales
            return scales
        shards = {}
        for path in files:
            with safe_open(path, framework='pt') as f:
                shards[path] = [key for key in f.keys() if key.endswith('.scale')]

    for path, keys in shards.items():
        if not keys:
            continue
        with safe_open(path, framework='pt') as f:
            for key in keys:
                scales[key[:-len('.scale')] + '.weight'] = f.get_tensor(key)
    logger.info(f'Loaded {len(scales)} checkpoint block scales from {model_dir!r} to drive weight-sync quantization.')
    _CHECKPOINT_SCALES[model_dir] = scales
    return scales


def _quantize_like_source_scale(weight: torch.Tensor, source_scale: torch.Tensor,
                                name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``weight`` to FP8 on the grid ``source_scale`` describes.

    Mirrors ``quantize_fp8_e4m3fn_like_scale`` / ``scale_from_amax``: the block size comes
    from the source scale's geometry and a UE8M0 source dtype forces power-of-two scales
    (``exp2(ceil(log2(s)))``), which is what reproduces an untouched DeepSeek-V4 weight
    exactly. The geometry is checked rather than assumed, so a layout this does not model
    fails loudly instead of writing a mis-shaped scale.
    """
    if weight.ndim != 2:
        raise RuntimeError(f'Block-FP8 export expects a 2D weight for {name!r}, got {weight.ndim}D')
    if source_scale.ndim != 2:
        raise RuntimeError(f'Unsupported scale rank for {name!r}: scale={tuple(source_scale.shape)}')

    rows, cols = weight.shape
    scale_rows, scale_cols = source_scale.shape
    block_m = -(-rows // scale_rows)
    block_n = -(-cols // scale_cols)
    expected = (-(-rows // block_m), -(-cols // block_n))
    if (scale_rows, scale_cols) != expected:
        raise RuntimeError(f'Unsupported block-FP8 scale geometry for {name!r}: weight={tuple(weight.shape)} '
                           f'scale={tuple(source_scale.shape)} implies blocks {(block_m, block_n)}')

    use_ue8m0 = source_scale.dtype == torch.float8_e8m0fnu
    param_lp, param_scale = scaled_fp8_blockwise(weight, weight_block_size=[block_m, block_n], use_ue8m0=use_ue8m0)
    return param_lp, param_scale


def _quant_deepseek_v4_weights(weights: Iterable[Tuple[str, torch.Tensor]],
                               model,
                               quant_config,
                               dtype: torch.dtype,
                               model_dir: Optional[str],
                               experts_staged: bool = False) -> Iterator[Tuple[str, torch.Tensor]]:
    """Prepare a DeepSeek-V4 stream for ``load_weights``.

    Three cases, because the stream's provenance decides what each tensor needs:

    * Routed experts still in their packed form (``int8`` / ``float8_e8m0fnu``) come
      straight from a checkpoint and only need to be seen as the ``uint8`` bytes
      ``Mxfp4MoEMethod`` allocated.
    * Routed experts in high precision come from a bridge that unpacked FP4 and folded
      the block scale back in (mcore_bridge does exactly this). They are always
      re-quantized to MXFP4 on the grid their checkpoint ``.scale`` describes and emitted
      as the packed weight plus a sibling ``.scale``. Re-quantizing an untouched expert is
      value-exact (verified bit-identical in an A/B run with frozen experts), so carrying
      them is safe whether or not they were trained -- which is why neither the colocate
      nor the server path needs ``tuner_type`` here. A *small* update, such as a merged
      LoRA delta of ~1e-4 relative, still falls below FP4's grid step (roughly 40% of the
      mean weight magnitude) and measures as a change 187x its own norm, which is why
      adapters on the expert linears stay refused upstream (``check_mxfp4_experts_are_syncable``).
    * Everything the checkpoint carries a sibling ``.scale`` for is quantized here, on the
      grid that scale describes -- the criterion and the geometry/dtype handling both
      mirror NVIDIA's ``requantize_hf_weight_scale_pairs``. DeepSeek-V4 declares
      ``scale_fmt: ue8m0``, so those scales are powers of two and requantizing an
      untouched weight reproduces it exactly.
    """
    # Whole-model property, so it is resolved once: only ``expert_dtype=fp4`` builds
    # MXFP4 experts. A Flash-Base engine serves block-FP8 experts, which the UE8M0
    # branch below reproduces exactly, so those requantize through the sibling-scale
    # path like any other dense weight.
    experts_are_mxfp4 = has_mxfp4_experts(model)
    checkpoint_scales = load_checkpoint_scales(model_dir) if model_dir else {}
    if not checkpoint_scales:
        logger.warning_once(
            'No checkpoint block scales are available, so no DeepSeek-V4 weight can be quantized on the way to the '
            'rollout engine. The engine would keep serving its initial weights while training moved on. Pass the '
            'model directory through to the weight sync.')
    quantized = 0
    carried_experts = 0

    for name, weight in weights:
        is_routed_expert = '.experts.' in name
        if is_routed_expert and weight.dtype in (torch.int8, torch.float8_e8m0fnu):
            if not experts_staged:
                raise RuntimeError(
                    f'{name!r} arrived as packed expert bytes, but the routed experts were not staged for this '
                    'reload. Packed bytes are in checkpoint layout, so loading them would overwrite the live '
                    'kernel-layout storage. The engine has no MXFP4 experts to stage, so this stream should not '
                    'carry packed expert bytes.')
            yield (name, weight.view(torch.uint8))
            continue
        if (is_routed_expert and experts_are_mxfp4
                and weight.dtype in (torch.bfloat16, torch.float16, torch.float32)):
            source_scale = checkpoint_scales.get(name)
            if source_scale is None:
                raise RuntimeError(
                    f'No checkpoint scale for routed expert {name!r}, so its MXFP4 block geometry is unknown. '
                    'Guessing it would silently corrupt the expert; pass the model directory through to the '
                    'weight sync so the scales can be read.')
            packed, expert_scale = quantize_mxfp4_e2m1(weight, source_scale.to(weight.device), name=name)
            carried_experts += 1
            # vLLM's DeepSeek-V4 fp4 mapper rewrites ``experts.N.wX.scale`` to
            # ``wX.weight_scale``, so the checkpoint's own name is what to emit.
            yield (name, packed)
            yield (name[:-len('.weight')] + '.scale', expert_scale)
            del weight, packed, expert_scale
            continue

        source_scale = checkpoint_scales.get(name)
        if source_scale is None:
            yield (name, weight)
            continue

        param_lp, param_scale = _quantize_like_source_scale(weight.to(dtype), source_scale.to(weight.device), name)
        quantized += 1
        yield (name, param_lp)
        yield (name + '_scale_inv', param_scale)
        del weight, param_lp, param_scale

    if quantized:
        logger.info_once(f'Quantized {quantized} DeepSeek-V4 weights to block-FP8 for the rollout engine.')

    if carried_experts:
        logger.info_once(f'Quantized {carried_experts} DeepSeek-V4 routed-expert tensors to MXFP4 for the rollout '
                         'engine.')


def quant_weights(weights: Iterable[Tuple[str, torch.Tensor]],
                  model,
                  quant_config,
                  dtype: torch.dtype = torch.bfloat16,
                  model_dir: Optional[str] = None,
                  experts_staged: bool = False) -> Iterator[Tuple[str, torch.Tensor]]:
    """Quantize an outgoing weight stream to what the engine's layers expect.

    Yields ``(name, tensor)`` lazily, each weight is dropped as soon as it has been handed
    downstream.
    """
    if is_deepseek_v4_model(model):
        yield from _quant_deepseek_v4_weights(weights, model, quant_config, dtype, model_dir,
                                              experts_staged=experts_staged)
        return

    _quant_state.seen_params.clear()
    _quant_state.fp8_param_names.clear()
    weight_block_size = _get_weight_block_size(quant_config)

    for name, weight in weights:
        if not is_fp8_weight(name, model):
            yield (name, weight)
            continue

        param_lp, param_scale = scaled_fp8_blockwise(weight.to(dtype), weight_block_size=weight_block_size)
        # No squeeze here: the kernel already returns exactly
        # (ceil(M/block), ceil(N/block)), or that with a leading expert dim. verl
        # squeezes the last dim to paper over a backend that returned a trailing
        # singleton; doing so would destroy a legitimately size-1 block dim, which
        # is what a weight with N <= block_size produces.

        yield (name, param_lp)
        yield (name + '_scale_inv', param_scale)

        del weight, param_lp, param_scale


def prepare_quanted_weights_for_loading(model) -> Dict[str, Any]:
    """Restore quantized params to the layout their ``weight_loader`` expects.

    Must run once before the first bucket, and pairs with
    ``process_quanted_weights_after_loading``, which re-applies the inference
    layout once the last bucket has landed. Each backend selects its own work by
    inspecting the model and returns an empty result when it finds nothing, so
    this is safe to call for any quantization scheme. The returned value is opaque
    reload state for the paired call.

    MXFP4 routed experts are staged whenever the engine actually serves them
    (``has_mxfp4_experts``), because the sync always re-quantizes and re-sends them.
    Carrying them is safe even when they were not trained: re-packing an untouched
    expert onto the grid its checkpoint ``.scale`` describes is value-exact -- an A/B
    run (LoRA, experts frozen) produced bit-identical ``rollout_correction/kl`` whether
    the experts were carried or dropped. This is what lets both the colocate and the
    vLLM-server paths work without knowing ``tuner_type``.

    Staging is gated on ``has_mxfp4_experts`` rather than applied unconditionally
    because it is not free: ``_mxfp4_staging_data`` clones the live parameter, which
    already holds the kernel layout, and ``process`` replays the kernel transform on it.
    When the engine has no MXFP4 experts there is nothing to stage and the list is empty.
    """
    stage_experts = has_mxfp4_experts(model)
    return {
        'fp8_layers': stage_fp8_params_for_loading(model),
        'mxfp4_moe_modules': stage_mxfp4_moe_params_for_loading(model) if stage_experts else [],
        'experts_staged': stage_experts,
    }


def process_quanted_weights_after_loading(model, reload_state: Optional[Dict[str, Any]]) -> None:
    """Re-apply the inference layout undone by ``prepare_quanted_weights_for_loading``."""
    reload_state = reload_state or {}
    process_fp8_weights_after_loading(
        reload_state.get('fp8_layers') or [], model=model, loaded_params=reload_state.get('loaded_params'))
    process_mxfp4_moe_weights_after_loading(reload_state.get('mxfp4_moe_modules') or [])


def load_quanted_weights(weights: Iterable[Tuple[str, torch.Tensor]],
                         model,
                         quant_config,
                         dtype: torch.dtype = torch.bfloat16,
                         model_dir: Optional[str] = None,
                         reload_state: Optional[Dict[str, Any]] = None):
    """Quantize ``weights`` and load them into a quantized vLLM model."""
    experts_staged = (reload_state or {}).get('experts_staged', False)
    weights_quantized = quant_weights(
        weights,
        model,
        quant_config,
        dtype=dtype,
        model_dir=model_dir,
        experts_staged=experts_staged)

    # Temporarily restore each parameter's vLLM subclass: some models dispatch on
    # the parameter type to pick the right weight loader, and the post-load hooks
    # have since rebuilt these as plain Parameters.
    restored = []
    for _, param in model.named_parameters():
        subclass_type = getattr(param, 'subclass_type', None)
        if subclass_type is not None and param.__class__ is not subclass_type:
            restored.append((param, param.__class__))
            param.__class__ = subclass_type
    try:
        loaded_params = model.load_weights(weights_quantized)
    finally:
        for param, orig_type in restored:
            param.__class__ = orig_type

    if reload_state is not None:
        # Which staged layers actually received data. A staged block-FP8 layer that got
        # nothing must not be re-derived: its staged view reinterprets the live packed
        # scale bytes as the checkpoint dtype, and re-deriving from that misread input
        # fills the scale with 0xFF, making the layer emit NaN.
        #
        # Accumulated, not assigned: the colocate paths call this once per refit, but the
        # vLLM server worker calls it once per weight bucket, so assigning would leave only
        # the last bucket's names. Every layer loaded by an earlier bucket would then look
        # untouched and be restored verbatim, silently discarding its synced weights.
        previous = reload_state.get('loaded_params') or ()
        reload_state['loaded_params'] = set(previous) | set(loaded_params or ())

    return loaded_params


def vllm_version_ge(min_version: str) -> bool:
    """Whether the installed vLLM is at least ``min_version`` (a dev build counts as latest)."""
    try:
        import vllm
    except ImportError:
        return False
    version_str = getattr(vllm, '__version__', None)
    if not version_str or 'dev' in version_str:
        return True
    from packaging import version
    return version.parse(version_str) >= version.parse(min_version)


def apply_vllm_quant_patches() -> None:
    """Install the vLLM patches the refit path depends on. Must run pre-engine.

    These have to be in place before the engine builds, because recording each
    layer's checkpoint layout happens inside the patched hook during the initial
    load. The mxfp4 path needs no startup patch -- it derives the checkpoint
    layout from the quant method and patches ``replace_parameter`` only for the
    duration of its own post-processing call.

    Runs in the vLLM worker subprocess (see
    :meth:`QuantSyncWorkerExtension.__new__`): the post-load hooks being patched
    execute there, not in the sampler/driver process that built the engine.
    """
    if _quant_state.vllm_patches:
        logger.debug('vLLM quantization patches are already applied.')
        return

    try:
        patchers = build_fp8_method_patchers(vllm_version_ge('0.20.0'))
    except ImportError as e:
        logger.warning(f'Could not build the vLLM FP8 refit patches, quantized weight sync may fail: {e}')
        return

    for patcher in patchers:
        patcher.start()
        _quant_state.vllm_patches.append(patcher)
    logger.info('Applied vLLM FP8 quantization patches for RL weight sync.')
