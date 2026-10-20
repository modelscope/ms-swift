# Copyright (c) ModelScope Contributors. All rights reserved.
"""BF16 -> MXFP4 (E2M1) quantization for DeepSeek-V4 routed experts.

Ported from NVIDIA Megatron-Bridge's ``quantize_mxfp4_e2m1_like_scale``
(``megatron/bridge/models/conversion/quantization_utils.py``), which is the same
routine verl relies on to carry expert updates into a quantized rollout engine.

The target layout is read off the published checkpoint rather than assumed:

    layers.0.ffn.experts.0.w1.weight   int8              (2048, 2048)   packed, 2/byte
    layers.0.ffn.experts.0.w1.scale    float8_e8m0fnu    (2048, 128)    one per 32 cols

so a logical ``(rows, cols)`` weight becomes ``(rows, cols // 2)`` packed bytes plus
``(rows, cols // 32)`` power-of-two scales. Emitting the checkpoint dtypes matters: the
engine's loader is fed checkpoint names and vLLM's DeepSeek-V4 mapper rewrites
``experts.N.wX.scale`` to ``wX.weight_scale`` for the MXFP4 quant method.

Round-trip is exact for weights already on the FP4 grid: dequantizing a real expert and
re-quantizing it reproduces the values with 0.0 error and a bit-identical scale. The
packed bytes differ in ~12.6% of positions, but only ever code 8 (-0.0) against code 0
(+0.0), which decode equal -- so a frozen expert can be re-sent without drift.
"""
import torch

# E2M1: 1 sign, 2 exponent, 1 mantissa bits. Code i indexes this table, sign in bit 3.
FP4_E2M1_MAX = 6.0
MXFP4_BLOCK_SIZE = 32
_FP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Midpoints of the table above: bucketize maps |x| to the nearest representable code.
_FP4_E2M1_BOUNDARIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# Cap the working set of the vectorized path; 256 experts x 4096 cols is large enough
# to matter on a rollout engine that is already near its memory ceiling.
_MAX_CHUNK_ELEMENTS = 16_000_000


def quantize_mxfp4_e2m1(weight: torch.Tensor,
                        source_scale: torch.Tensor,
                        *,
                        name: str = '',
                        block_size: int = MXFP4_BLOCK_SIZE):
    """Quantize a 2-D BF16 weight to packed MXFP4, on the grid ``source_scale`` describes.

    ``source_scale`` supplies both the geometry and the scale family, exactly as the
    block-FP8 path takes them from the checkpoint's sibling ``.scale``. Returns
    ``(packed_int8, scale)`` with the checkpoint's own dtypes.
    """
    if weight.ndim != 2:
        label = f' for {name!r}' if name else ''
        raise RuntimeError(f'MXFP4 quantization expects a 2-D weight{label}, got {weight.ndim}D')

    rows, cols = weight.shape
    expected_scale = (rows, cols // block_size)
    if cols % 2 or cols % block_size or tuple(source_scale.shape) != expected_scale:
        label = f' for {name!r}' if name else ''
        raise RuntimeError(f'Unsupported MXFP4 geometry{label}: weight={tuple(weight.shape)} '
                           f'scale={tuple(source_scale.shape)}, expected scale={expected_scale}')

    device = weight.device
    weight_f32 = weight.to(torch.float32)
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8, device=device)
    scale_f32 = torch.empty(expected_scale, dtype=torch.float32, device=device)
    boundaries = torch.tensor(_FP4_E2M1_BOUNDARIES, dtype=torch.float32, device=device)
    scale_cols = expected_scale[1]

    rows_per_chunk = max(1, min(rows, _MAX_CHUNK_ELEMENTS // max(cols, 1)))
    for start in range(0, rows, rows_per_chunk):
        end = min(start + rows_per_chunk, rows)
        chunk = weight_f32[start:end].reshape(-1, scale_cols, block_size)
        amax = chunk.abs().amax(dim=-1)
        # Power-of-two scales: the checkpoint declares ue8m0, and only exact powers of
        # two survive the e8m0 cast without perturbing an untouched weight.
        unrounded = torch.where(amax > 0, amax / FP4_E2M1_MAX, torch.ones_like(amax))
        chunk_scale = torch.exp2(torch.ceil(torch.log2(unrounded)).clamp(min=-127, max=127))
        scale_f32[start:end] = chunk_scale

        normalized = chunk / chunk_scale[:, :, None]
        codes = torch.bucketize(normalized.abs(), boundaries).to(torch.uint8)
        codes = (codes | ((normalized < 0).to(torch.uint8) * 8)).reshape(end - start, cols)
        # Even columns occupy the low nibble.
        lo = codes[:, 0::2].to(torch.int16)
        hi = codes[:, 1::2].to(torch.int16)
        packed[start:end] = (lo | (hi << 4)).to(torch.uint8)

    return packed.contiguous().view(torch.int8), scale_f32.to(source_scale.dtype)


def dequantize_mxfp4_e2m1(packed: torch.Tensor,
                          scale: torch.Tensor,
                          *,
                          dtype: torch.dtype = torch.bfloat16,
                          block_size: int = MXFP4_BLOCK_SIZE) -> torch.Tensor:
    """Inverse of :func:`quantize_mxfp4_e2m1`, for tests and diagnostics."""
    packed_u8 = packed.view(torch.uint8)
    # A nibble is 0..15 with bit 3 as the sign, so the table has to be signed too.
    signed_values = _FP4_E2M1_VALUES + tuple(-v for v in _FP4_E2M1_VALUES)
    table = torch.tensor(signed_values, dtype=torch.float32, device=packed.device)
    lo = table[(packed_u8 & 0xF).to(torch.int64)]
    hi = table[(packed_u8 >> 4).to(torch.int64)]
    logical = torch.stack([lo, hi], dim=-1).reshape(packed_u8.shape[0], -1)

    rows, cols = logical.shape
    scale_f32 = scale.to(torch.float32).reshape(rows, cols // block_size, 1)
    return (logical.reshape(rows, cols // block_size, block_size) * scale_f32).reshape(rows, cols).to(dtype)
