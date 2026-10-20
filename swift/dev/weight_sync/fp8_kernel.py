# Copyright (c) ModelScope Contributors. All rights reserved.
"""Blockwise BF16 -> FP8 quantization for RL weight synchronization.

Ported from verl (``verl/utils/kernel/fp8_kernel.py``), which in turn credits
slime (``slime/backends/megatron_utils/kernels/fp8_kernel.py``). The arithmetic
is kept bit-identical to those implementations so a policy synced through this
path matches what a vLLM engine loaded from an FP8 checkpoint would compute.
"""
import os
from typing import List, Optional, Sequence, Tuple, Union

import torch

from swift.dev.utils.logger import get_logger

logger = get_logger()

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    logger.debug('Triton is not available; FP8 quantization falls back to the PyTorch implementation.')

# Escape hatch for debugging a suspected Triton miscompile: forces the PyTorch path.
_DISABLE_TRITON_FP8 = os.environ.get('SWIFT_DISABLE_TRITON_FP8', '0').lower() in ('1', 'true', 'yes')

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
FP8_MIN = -FP8_MAX

# Guards a fully-zero block: without it the scale would be 0 and the division below
# would produce NaN instead of the zeros the block should quantize to.
_EPS = 1e-10

BlockSize = Union[List[int], Tuple[int, int], Sequence[int]]


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def is_triton_available() -> bool:
    return _TRITON_AVAILABLE and not _DISABLE_TRITON_FP8


if _TRITON_AVAILABLE:

    @triton.jit
    def _blockwise_cast_to_fp8_kernel(
        X,
        Y,
        S,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        M,
        N,
        eps,
        fp8_min,
        fp8_max,
        BLOCK_M: tl.constexpr = 128,
        BLOCK_N: tl.constexpr = 128,
        USE_UE8M0: tl.constexpr = False,
    ):
        """Quantize one (BLOCK_M, BLOCK_N) tile per program instance.

        Computing the absmax, the scale and the clamped cast in a single pass keeps
        every intermediate in registers, so quantizing a 671B-parameter model does
        not allocate a second copy of each weight.
        """
        pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
        pid_n = tl.cast(tl.program_id(axis=1), tl.int64)

        off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = off_m < M
        mask_n = off_n < N
        mask = mask_m[:, None] & mask_n[None, :]

        x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=mask, other=0.0).to(tl.float32)

        _absmax = tl.maximum(tl.max(tl.abs(x)), eps)
        x_s = _absmax / fp8_max
        if USE_UE8M0:
            # Round up to a power of two so the scale is exactly representable as a
            # UE8M0 exponent, which is the only form DeepSeek-V4 kernels accept.
            # Rounding up (never down) keeps the block inside the FP8 range.
            x_s = tl.exp2(tl.ceil(tl.log2(x_s)))
        s_inv = 1.0 / x_s
        y_q = tl.clamp(x * s_inv, fp8_min, fp8_max).to(Y.dtype.element_ty)

        tl.store(Y + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, y_q, mask=mask)
        tl.store(S + pid_m * stride_sm + pid_n * stride_sn, x_s)

    def blockwise_cast_to_fp8_triton(x: torch.Tensor,
                                     weight_block_size: Optional[BlockSize] = None,
                                     use_ue8m0: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a 2D tensor whose dims are already multiples of the block size.

        Returns ``(fp8_tensor, scale)`` where ``scale`` has shape
        ``(ceil(M / BLOCK_M), ceil(N / BLOCK_N))`` and multiplies to dequantize.
        """
        assert x.dim() == 2, f'Expected a 2D tensor, got {x.dim()}D'

        block_m, block_n = 128, 128
        if weight_block_size is not None:
            block_m, block_n = weight_block_size[0], weight_block_size[1]

        m, n = x.shape
        y = torch.empty(m, n, device=x.device, dtype=FP8_DTYPE)
        s = torch.empty(ceil_div(m, block_m), ceil_div(n, block_n), dtype=torch.float32, device=x.device)

        def grid(meta):
            return (triton.cdiv(m, meta['BLOCK_M']), triton.cdiv(n, meta['BLOCK_N']))

        if x.is_contiguous():
            kwargs = {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'num_warps': 8, 'num_stages': 2}
        else:
            kwargs = {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'num_warps': 1, 'num_stages': 4}

        _blockwise_cast_to_fp8_kernel[grid](
            x,
            y,
            s,
            *x.stride(),
            *y.stride(),
            *s.stride(),
            m,
            n,
            _EPS,
            FP8_MIN,
            FP8_MAX,
            USE_UE8M0=use_ue8m0,
            **kwargs,
        )
        return y, s


def _pad_to_block(data_hp: torch.Tensor, block_size0: int, block_size1: int) -> Tuple[torch.Tensor, torch.Size]:
    original_shape = data_hp.shape
    pad_dim0 = (block_size0 - data_hp.shape[0] % block_size0) % block_size0
    pad_dim1 = (block_size1 - data_hp.shape[1] % block_size1) % block_size1
    if pad_dim0 > 0 or pad_dim1 > 0:
        data_hp = torch.nn.functional.pad(data_hp, (0, pad_dim1, 0, pad_dim0), mode='constant', value=0)
    return data_hp, original_shape


def _scaled_fp8_blockwise_triton(data_hp: torch.Tensor,
                                 weight_block_size: BlockSize,
                                 use_ue8m0: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    data_hp, original_shape = _pad_to_block(data_hp, weight_block_size[0], weight_block_size[1])
    fp8_data, scale = blockwise_cast_to_fp8_triton(data_hp, weight_block_size, use_ue8m0=use_ue8m0)
    if fp8_data.shape != original_shape:
        fp8_data = fp8_data[:original_shape[0], :original_shape[1]].contiguous()
    return fp8_data, scale


def _scaled_fp8_blockwise_pytorch(data_hp: torch.Tensor,
                                  weight_block_size: BlockSize,
                                  use_ue8m0: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Memory-conscious PyTorch fallback.

    Peak memory matters here: a single expert weight can be several GB, so the
    float32 working copy is freed as soon as each step is done with it rather
    than being held until the function returns.
    """
    block_size0, block_size1 = weight_block_size[0], weight_block_size[1]
    assert block_size0 == block_size1, 'The PyTorch fallback requires a square block size'

    data_hp, original_shape = _pad_to_block(data_hp, block_size0, block_size1)
    padded_shape = data_hp.shape
    blk_m, blk_n = padded_shape[0] // block_size0, padded_shape[1] // block_size1

    data_hp = data_hp.reshape(blk_m, block_size0, blk_n, block_size1)
    data_hp = data_hp.permute(0, 2, 1, 3).contiguous()
    data_hp = data_hp.to(torch.float32).flatten(start_dim=2)

    max_abs = data_hp.abs().amax(dim=-1, keepdim=True)

    scale_fp = torch.empty_like(max_abs)
    torch.div(FP8_MAX, max_abs, out=scale_fp)
    # An all-zero block gives max_abs == 0 (scale would be inf) and an overflowed
    # block gives inf; both dequantize correctly with a unit scale.
    scale_fp = torch.where(max_abs == 0, torch.ones_like(scale_fp), scale_fp)
    scale_fp = torch.where(max_abs == torch.inf, torch.ones_like(scale_fp), scale_fp)
    del max_abs

    if use_ue8m0:
        # The stored factor is the reciprocal, so that is what has to be a power of
        # two. Rounding it up (not the multiplier) keeps the block inside FP8 range.
        descale = torch.exp2(torch.ceil(torch.log2(torch.reciprocal(scale_fp))))
        scale_fp = torch.reciprocal(descale)

    # ``keepdim=True`` above left a trailing singleton; drop it so both backends
    # return the documented (ceil(M/BLOCK_M), ceil(N/BLOCK_N)) shape.
    descale_fp = torch.reciprocal(scale_fp).squeeze(-1)

    data_hp.mul_(scale_fp)
    del scale_fp
    data_hp.clamp_(min=FP8_MIN, max=FP8_MAX)

    fp8_data = data_hp.to(FP8_DTYPE)
    del data_hp

    fp8_data = fp8_data.reshape(blk_m, blk_n, block_size0, block_size1).permute(0, 2, 1, 3).reshape(padded_shape)

    if original_shape[0] != padded_shape[0] or original_shape[1] != padded_shape[1]:
        fp8_data = fp8_data[:original_shape[0], :original_shape[1]].contiguous()

    return fp8_data, descale_fp


def scaled_fp8_blockwise(data_hp: torch.Tensor,
                         weight_block_size: BlockSize,
                         use_ue8m0: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cast a high-precision tensor to blockwise FP8.

    Accepts a 2D weight ``(M, N)`` or a stacked 3D expert weight
    ``(num_experts, M, N)``. The 3D form is quantized per expert, matching how
    vLLM's fused-MoE layers store block scales -- ``(num_experts, M/block,
    N/block)`` -- so a bridge that emits pre-fused expert tensors is handled
    without the caller reshaping. A single scale spanning several experts would be
    wrong: each expert's weights have their own dynamic range.

    Args:
        data_hp: Input tensor of shape ``(M, N)`` or ``(E, M, N)`` in bf16/fp16/fp32.
        weight_block_size: ``[BLOCK_M, BLOCK_N]``, normally ``[128, 128]``.
        use_ue8m0: Constrain every block scale to a power of two. Required for a
            checkpoint that declares ``scale_fmt: ue8m0`` (DeepSeek-V4): its weights
            already sit on that grid, so a power-of-two scale reproduces them
            exactly, whereas a free fp32 scale re-quantizes onto a different grid
            and perturbs even a weight that training never touched.

    Returns:
        ``(fp8_data, descale)`` where ``descale`` holds the per-block factors to
        multiply by when dequantizing, with the same leading expert dim as the input.
    """
    if data_hp.dim() == 3:
        quantized, scales = [], []
        for expert in data_hp:
            q, s = scaled_fp8_blockwise(expert, weight_block_size, use_ue8m0=use_ue8m0)
            quantized.append(q)
            scales.append(s)
        return torch.stack(quantized, dim=0), torch.stack(scales, dim=0)

    assert data_hp.dim() == 2, f'Only 2D or 3D input tensors are supported, got {data_hp.dim()}D'

    if is_triton_available() and data_hp.is_cuda:
        return _scaled_fp8_blockwise_triton(data_hp, weight_block_size, use_ue8m0=use_ue8m0)

    return _scaled_fp8_blockwise_pytorch(data_hp, weight_block_size, use_ue8m0=use_ue8m0)
