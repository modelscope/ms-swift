"""Shared load-time quantization mapping for model builders and export."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from swift.dev.config import QuantizeConfig

LOAD_TIME_QUANT_METHODS = frozenset({'bnb', 'hqq', 'eetq', 'quanto', 'fp8'})
CALIBRATION_QUANT_METHODS = frozenset({'awq', 'gptq', 'gptq_v2'})


def quantizer_kwargs(quantize_config: QuantizeConfig,
                     overrides: Optional[Dict[str, Any]] = None,
                     *,
                     modules_to_not_convert: Optional[List[str]] = None,
                     torch_dtype: Optional[str] = None) -> Dict[str, Any]:
    """Translate ``QuantizeConfig`` into kwargs understood by ``twinkle.quantizer``."""
    kwargs: Dict[str, Any] = {'quant_bits': quantize_config.quant_bits}
    method = quantize_config.quant_method
    if method == 'bnb':
        kwargs.update(
            bnb_4bit_compute_dtype=quantize_config.bnb_4bit_compute_dtype or torch_dtype,
            bnb_4bit_quant_type=quantize_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=quantize_config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_storage=quantize_config.bnb_4bit_quant_storage,
            llm_int8_skip_modules=modules_to_not_convert)
    elif method == 'hqq':
        kwargs.update(axis=quantize_config.hqq_axis, group_size=quantize_config.group_size,
                      skip_modules=modules_to_not_convert)
    elif method in {'eetq', 'quanto', 'fp8'}:
        kwargs['modules_to_not_convert'] = modules_to_not_convert
    kwargs = {name: value for name, value in kwargs.items() if value is not None}
    kwargs.update(overrides or {})
    return kwargs


def modules_to_not_convert(model_loader: Any) -> Optional[List[str]]:
    """Return architecture parts that legacy keeps in the model's compute dtype."""
    if model_loader is None:
        return None
    arch = model_loader.model_arch
    result: List[str] = []
    if getattr(model_loader.model_info, 'is_moe_model', False):
        result.extend(('mlp.gate', 'mlp.shared_expert_gate'))
    result.extend(getattr(arch, 'vision_tower', ()) or ())
    result.extend(getattr(arch, 'aligner', ()) or ())
    if getattr(arch, 'lm_head_name', None):
        result.append(arch.lm_head_name)
    return list(dict.fromkeys(result)) or None


def build_load_quantization_config(quantize_config: Optional[QuantizeConfig],
                                   *,
                                   model_loader: Any = None,
                                   torch_dtype: Optional[str] = None):
    """Build the Transformers ``quantization_config`` requested for model loading."""
    if quantize_config is None or quantize_config.quant_method is None:
        return None
    method = quantize_config.quant_method
    if method in CALIBRATION_QUANT_METHODS:
        raise ValueError(
            f'quant_method={method!r} is a calibration/export method, not a load-time quantizer. '
            'Load an already quantized checkpoint without --quant_method, or use bnb/hqq/eetq/quanto/fp8.')
    if method not in LOAD_TIME_QUANT_METHODS:
        raise ValueError(f'Unknown load-time quantization method: {method!r}.')
    from twinkle.quantizer import get_quantizer
    kwargs = quantizer_kwargs(
        quantize_config,
        modules_to_not_convert=modules_to_not_convert(model_loader),
        torch_dtype=torch_dtype)
    return get_quantizer(method, **kwargs).get_quantization_config()
