# Copyright (c) ModelScope Contributors. All rights reserved.
"""Refit support for vLLM's ``Fp8LinearMethod`` and ``Fp8MoEMethod``.

refer: verl (``verl/utils/vllm/vllm_fp8_utils.py``).

Two independent problems have to be solved for an RL weight refit to land
correctly on a block-FP8 layer:

* vLLM's post-load hooks rebuild parameters as plain ``torch.nn.Parameter``,
  dropping the subclass metadata (``ModelWeightParameter`` and friends) that the
  weight loaders dispatch on. The helpers here carry that metadata across.
* Kernel post-processing rewrites weights and scales into an inference layout
  that the checkpoint tensors no longer fit (DeepGEMM packs an ``(N/128, K/128)``
  fp32 block scale into an ``(N, K/512)`` int32 UE8M0 tensor). The staging
  helpers hand ``load_weights`` a checkpoint-layout buffer and then fold the
  re-derived result back into the original storage, so pointers captured by a
  CUDA graph stay valid.

``manager.py`` is the entry point that drives all of this.
"""
import inspect
import math
from typing import Dict, List, Optional
from unittest.mock import patch

import torch

from swift.dev.utils.logger import get_logger

logger = get_logger()

_FP8_PRISTINE_ATTR = '_swift_fp8_pristine'
_FP8_LIVE_ATTR = '_swift_fp8_live_params'

# ``Fp8LinearMethod`` owns the first group, ``Fp8MoEMethod`` the second. The MoE
# scale is named ``w13_weight_scale_inv`` under block quant and
# ``w13_weight_scale`` otherwise (``self.weight_scale_name``), so both spellings
# are listed; only the names a layer actually has get recorded.
_FP8_LINEAR_PARAM_NAMES = ('weight', 'weight_scale_inv', 'weight_scale')
_FP8_MOE_PARAM_NAMES = (
    'w13_weight',
    'w2_weight',
    'w13_weight_scale_inv',
    'w2_weight_scale_inv',
    'w13_weight_scale',
    'w2_weight_scale',
)
_FP8_REFIT_PARAM_NAMES = (*_FP8_LINEAR_PARAM_NAMES, *_FP8_MOE_PARAM_NAMES)


def _iter_transferable_param_attrs(src_param):
    """Yield the attributes worth moving onto a rebuilt parameter.

    Methods bound to a tensor are excluded, and that exclusion is the whole point
    of this helper. ``dir()`` reports vLLM's loader entry points
    (``load_merged_column_weight`` and friends) alongside the data attributes,
    and copying one lands a method whose ``self`` is some *other* tensor in the
    destination's instance dict, where it shadows the class. Every later load
    then writes into that stale tensor instead of the parameter it was handed --
    silently, because the shapes still agree.

    Note the test is "bound to a tensor", not "bound to the source": these
    methods are copied forward from rebuild to rebuild, so by the time one does
    damage its ``self`` is usually a temporary from several rebuilds back.
    ``weight_loader`` is bound to the layer, so it survives.
    """
    base_param_attrs = dir(torch.nn.Parameter)
    for attr in dir(src_param):
        if attr in base_param_attrs or attr.startswith('__'):
            continue
        try:
            value = getattr(src_param, attr)
        except AttributeError:
            continue
        if inspect.ismethod(value) and isinstance(value.__self__, torch.Tensor):
            continue
        yield attr, value


def copy_param_subclass_attrs(dst_param, src_param):
    """Carry vLLM's parameter metadata across a parameter rebuild.

    vLLM stores loader dispatch information (``output_dim``, ``weight_loader``,
    tp state, ...) as plain attributes on ``Parameter`` subclasses. Anything that
    re-wraps a tensor has to bring them along or the next refit will not know how
    to shard the incoming weight.
    """
    if src_param is None:
        return

    for attr, value in _iter_transferable_param_attrs(src_param):
        try:
            setattr(dst_param, attr, value)
        except AttributeError:
            pass

    subclass_type = getattr(src_param, 'subclass_type', type(src_param))
    if subclass_type is not torch.nn.Parameter:
        dst_param.subclass_type = subclass_type


def _fold_into_live_param(layer, param_name: str, param: torch.nn.Parameter, new_data) -> None:
    """Move post-processing output into the storage the CUDA graph captured."""
    if isinstance(new_data, torch.nn.Parameter):
        new_data = new_data.data

    if new_data.shape != param.shape or new_data.dtype != param.dtype:
        raise RuntimeError(f'FP8 refit re-derived {param_name} as {tuple(new_data.shape)}/{new_data.dtype}, '
                           f'but the live parameter is {tuple(param.shape)}/{param.dtype}; '
                           'its storage cannot be updated in place.')
    if new_data.data_ptr() != param.data_ptr():
        param.data.copy_(new_data)
    setattr(layer, param_name, param)


def replace_parameter_preserve_subclass(layer: torch.nn.Module,
                                        param_name: str,
                                        new_data: Optional[torch.Tensor],
                                        prefer_copy: bool = False) -> None:
    """Stand-in for vLLM's ``replace_parameter`` inside the fp8 quant module.

    During a refit of a staged layer this folds the result into the live
    parameter and reinstates it immediately, rather than after
    ``process_weights_after_loading`` returns. That timing is required for MoE:
    the tail of ``Fp8MoEMethod._setup_kernel`` rebuilds ``moe_quant_config`` and
    ``moe_kernel`` from whatever is on the layer at that moment, and those cache
    tensor references, so a later swap would leave the kernel pointing at the
    staging buffers.
    """
    live = getattr(layer, _FP8_LIVE_ATTR, None) or {}
    param = live.get(param_name)
    if param is not None and new_data is not None:
        _fold_into_live_param(layer, param_name, param, new_data)
        del live[param_name]
        return

    if new_data is None:
        setattr(layer, param_name, None)
        return

    if isinstance(new_data, torch.nn.Parameter):
        new_data = new_data.data

    old_param = getattr(layer, param_name, None)
    param = torch.nn.Parameter(new_data, requires_grad=False)
    copy_param_subclass_attrs(param, old_param)
    setattr(layer, param_name, param)


def _restore_layer_param_subclass_attrs(layer: torch.nn.Module, old_params: Dict[str, torch.nn.Parameter]) -> None:
    for name, old_param in old_params.items():
        new_param = getattr(layer, name, None)
        if isinstance(new_param, torch.nn.Parameter):
            copy_param_subclass_attrs(new_param, old_param)


def _record_pristine_fp8_layout(layer, params) -> None:
    """Remember the checkpoint layout of a layer's FP8 weights and scales.

    Kernel post-processing rewrites these into inference layouts that the
    checkpoint tensors no longer fit, so a refit feeding checkpoint-layout scales
    hits a shape assert. Keep enough state to rebuild the original buffers before
    each reload.

    Only the layout is recorded, never a tensor. This runs during model load,
    inside the memory pool vLLM tags as ``weights``, so any buffer kept here
    would be subject to that pool's discard-and-rezero across a sleep and would
    not survive to the next refit anyway.
    """
    if hasattr(layer, _FP8_PRISTINE_ATTR):
        return

    pristine = {}
    for name in _FP8_REFIT_PARAM_NAMES:
        param = params.get(name)
        if not isinstance(param, torch.nn.Parameter):
            continue
        pristine[name] = (tuple(param.shape), param.dtype)

    if pristine:
        setattr(layer, _FP8_PRISTINE_ATTR, pristine)


def _layer_needs_fp8_staging(layer, pristine) -> bool:
    for name, (shape, dtype) in pristine.items():
        param = getattr(layer, name, None)
        if isinstance(param, torch.nn.Parameter) and (tuple(param.shape) != shape or param.dtype != dtype):
            return True
    return False


def _fp8_staging_data(param: torch.nn.Parameter, shape, dtype) -> torch.Tensor:
    """Checkpoint-layout tensor that ``load_weights`` can write into.

    Reuses the live storage whenever the kernel transform only reshaped it, so
    those writes land straight in the tensor the CUDA graph points at. Otherwise
    the buffer is allocated fresh for this refit and dropped with it, which keeps
    nothing alive across the engine's sleep.

    An all-ones byte fill is NaN in every dtype staged here, so a buffer the
    incoming stream fails to write blows up on the first forward instead of being
    repacked into the live weights. Uninitialized memory would instead carry
    whatever the caching allocator last held, which for a scale is most likely a
    plausible-looking value from a previous refit.
    """
    if param.dtype == dtype and param.numel() == math.prod(shape):
        return param.data.reshape(shape)
    staging = torch.empty(shape, dtype=dtype, device=param.device)
    staging.view(torch.uint8).fill_(0xFF)
    return staging


def stage_fp8_params_for_loading(model) -> List[torch.nn.Module]:
    """Expose checkpoint-layout buffers so ``load_weights`` can write into them.

    Covers both quant methods: ``Fp8LinearMethod`` layers keyed on ``weight`` /
    ``weight_scale*`` and ``Fp8MoEMethod`` expert modules keyed on ``w13_*`` /
    ``w2_*``. The live parameters are set aside and reinstated by
    ``process_fp8_weights_after_loading``, which puts the re-derived inference
    layout back into their original storage. No storage the CUDA graph captured
    is reallocated -- only tensor contents change -- and no forward pass runs
    between the two calls.
    """
    staged_layers = []
    for layer in model.modules():
        pristine = getattr(layer, _FP8_PRISTINE_ATTR, None)
        # A layer left in checkpoint layout (its kernel did not repack, e.g. a
        # Triton or vLLM-CUTLASS MoE backend) already loads as-is.
        if not pristine or not _layer_needs_fp8_staging(layer, pristine):
            continue

        live = {}
        for name, (shape, dtype) in pristine.items():
            param = getattr(layer, name, None)
            if not isinstance(param, torch.nn.Parameter):
                continue
            live[name] = param
            staged = torch.nn.Parameter(_fp8_staging_data(param, shape, dtype), requires_grad=False)
            copy_param_subclass_attrs(staged, param)
            setattr(layer, name, staged)

        setattr(layer, _FP8_LIVE_ATTR, live)
        staged_layers.append(layer)

    # A zero here means no layer ever recorded a pristine layout, so the
    # post-processing hook never ran and the refit is unprotected.
    logger.info(f'Staged {len(staged_layers)} FP8 layers for refit.')
    return staged_layers


def process_fp8_weights_after_loading(layers, model=None, loaded_params=None) -> None:
    """Re-derive the inference layout in place and reinstate the live params.

    ``loaded_params`` names the parameters ``load_weights`` actually wrote. A staged
    layer absent from it received nothing and must be restored rather than re-derived:
    its staged view reinterprets the live packed scale bytes as the checkpoint dtype
    (DeepSeek-V4 keeps ``weight_scale_inv`` as strided int32 while the checkpoint
    layout is ``(12, 32) float8_e8m0fnu``), so re-deriving from unwritten staged data
    fills the scale with 0xFF and the layer then emits NaN. ``None`` keeps the previous
    unconditional behaviour.
    """
    layer_prefix = {}
    if loaded_params is not None and model is not None:
        layer_prefix = {id(module): name for name, module in model.named_modules()}

    for layer in layers:
        live = getattr(layer, _FP8_LIVE_ATTR, None) or {}

        if loaded_params is not None:
            prefix = layer_prefix.get(id(layer))
            touched = prefix is not None and any(f'{prefix}.{name}' in loaded_params for name in live)
            if not touched:
                # Nothing was written here, so the staged buffers hold misread bytes.
                for name, param in live.items():
                    setattr(layer, name, param)
                if hasattr(layer, _FP8_LIVE_ATTR):
                    delattr(layer, _FP8_LIVE_ATTR)
                continue

        quant_method = getattr(layer, 'quant_method', None)
        process = getattr(quant_method, 'process_weights_after_loading', None)
        if process is not None:
            process(layer)

        # Anything routed through the patched ``replace_parameter`` has already
        # been folded and dropped from ``live``; what remains was rewritten by a
        # kernel that imported ``replace_parameter`` into its own module, out of
        # reach of the patch. The block-scaled linear kernels all do that.
        for name, param in list(live.items()):
            _fold_into_live_param(layer, name, param, getattr(layer, name))

        if hasattr(layer, _FP8_LIVE_ATTR):
            delattr(layer, _FP8_LIVE_ATTR)


def _make_process_weights_after_loading_wrapper(original_fn):

    def _patched_process_weights_after_loading(self, layer) -> None:
        old_params = dict(layer.named_parameters(recurse=False))
        _record_pristine_fp8_layout(layer, old_params)
        with patch('vllm.model_executor.layers.quantization.fp8.replace_parameter',
                   replace_parameter_preserve_subclass):
            original_fn(self, layer)
        _restore_layer_param_subclass_attrs(layer, old_params)

    return _patched_process_weights_after_loading


def _process_weights_after_loading_for_vllm14(self, layer) -> None:
    """Linear post-load for vLLM 0.18-0.19.

    Same transformation as upstream, but without creating new
    ``torch.nn.Parameter`` objects, which would drop the ``weight_loader``
    attribute the refit needs.
    """
    from torch.nn import Parameter
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        maybe_post_process_fp8_weight_block,
        process_fp8_weight_block_strategy,
    )
    from vllm.model_executor.parameter import BlockQuantScaleParameter, ModelWeightParameter

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == 'dynamic'

    def _create_param_from_subclass_attributes(custom_param):
        param = Parameter(custom_param.data, requires_grad=False)
        copy_param_subclass_attrs(param, custom_param)
        return param

    weight, weight_scale_inv = process_fp8_weight_block_strategy(layer.weight, layer.weight_scale_inv)

    layer.weight = _create_param_from_subclass_attributes(
        ModelWeightParameter(
            data=weight.data,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight.weight_loader,
        ))
    layer.weight_scale_inv = _create_param_from_subclass_attributes(
        BlockQuantScaleParameter(
            data=weight_scale_inv.data,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight_scale_inv.weight_loader,
        ))

    # vLLM v0.17 removed the `else: register_parameter("input_scale", None)` from
    # create_weights() for dynamic activation, but apply() still accesses
    # layer.input_scale. Since block_quant always uses dynamic activation, ensure
    # the attribute exists.
    if not hasattr(layer, 'input_scale'):
        layer.input_scale = None

    maybe_post_process_fp8_weight_block(layer)


def build_fp8_method_patchers(vllm_ge_020: bool) -> list:
    """Patchers that make the FP8 quant methods survive a refit, not yet started.

    The caller owns starting and tracking them so all patch lifetime lives in one
    place.
    """
    linear_path = 'vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading'
    moe_path = 'vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod.process_weights_after_loading'

    # vLLM 0.20 refactored FP8 post-load handling and removed
    # maybe_post_process_fp8_weight_block. Keep its native transformation logic,
    # but preserve parameter subclass metadata for RL weight reloads.
    if vllm_ge_020:
        from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod

        wrap = _make_process_weights_after_loading_wrapper
        return [
            patch(linear_path, wrap(Fp8LinearMethod.process_weights_after_loading)),
            patch(moe_path, wrap(Fp8MoEMethod.process_weights_after_loading)),
        ]

    from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

    return [
        patch(linear_path, _process_weights_after_loading_for_vllm14),
        patch(moe_path, _make_process_weights_after_loading_wrapper(Fp8MoEMethod.process_weights_after_loading)),
    ]
