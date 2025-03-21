import math
from typing import Any, Dict

import torch


def _to_llama3_rope(inv_freq: torch.Tensor, rope_scaling: Dict[str, Any]):
    # copy from transformers
    factor = rope_scaling['factor']  # `8` in the original implementation
    low_freq_factor = rope_scaling['low_freq_factor']  # `1` in the original implementation
    high_freq_factor = rope_scaling['high_freq_factor']  # `4` in the original implementation
    old_context_len = rope_scaling['original_max_position_embeddings']  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama


def _to_linear_rope(inv_freq: torch.Tensor, rope_scaling: Dict[str, Any]):
    factor = rope_scaling['factor']
    inv_freq /= factor
    return inv_freq


ROPE_MAPPING = {'llama3': _to_llama3_rope, 'linear': _to_linear_rope}


def update_rope_inv_freq(inv_freq: torch.Tensor, rope_scaling: Dict[str, Any]) -> None:
    new_inv_freq = ROPE_MAPPING[rope_scaling['rope_type']](inv_freq, rope_scaling)
    inv_freq.data.copy_(new_inv_freq)
