import torch
import triton
import triton.language as tl
from contextlib import nullcontext

# signed int32 max is 2**31-1 so num_elements cannot exceed 2**31
NUM_INT32_ELEMENTS = 2**31
SAFE_INT32_BUFFER_MULTIPLIER = 4
BLOCK_SIZE = 1024
INT32_SAFETY_BUFFER = NUM_INT32_ELEMENTS - BLOCK_SIZE * SAFE_INT32_BUFFER_MULTIPLIER


def torch_gpu_device(device):
    return nullcontext()


@triton.jit
def _fg_kernel(
    e,
    g,
    h,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)  # .to(tl.float32)

    # f = e * sigmoid(e)
    f_row = e_row * tl.sigmoid(e_row)  # e_row / (1 + tl.exp(-e_row))
    f_row = f_row.to(g_row.dtype)  # Exact copy from HF
    # h = f * g
    h_row = f_row * g_row

    # Store h
    tl.store(h + offsets, h_row, mask=mask)


def _fg_grid(meta, n_elements):
    return (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )


def swiglu_fg_kernel(e, g):
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype=e.dtype, device=e.device)

    def grid(meta):
        return _fg_grid(meta, n_elements)

    with torch_gpu_device(e.device):
        _fg_kernel[grid](
            e,
            g,
            h,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
        )
    return h


@triton.jit
def _DWf_DW_dfg_kernel(
    DW,
    e,
    g,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    '''
    e = e.float()
    se = 1.0 / (1.0 + torch.exp(-e))
    f = (se * e).to(dtype)
    h = f * g
    df = DW * f
    dg = DW * g
    de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    '''
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask=mask, other=0)  # .to(tl.float32)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)  # .to(tl.float32)

    # e = e.float()
    # se = 1.0 / (1.0 + torch.exp(-e))
    se_row = tl.sigmoid(e_row)  # 1.0 / (1.0 + tl.exp(-e_row))
    # f = (se * e).to(dtype)
    f_row = se_row * e_row
    f_row = f_row.to(DW_row.dtype)
    # h = f * g
    h_row = f_row * g_row
    # df = DW * f
    df_row = DW_row * f_row
    # dg = DW * g
    dg_row = DW_row * g_row
    # de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    de_row = dg_row.to(tl.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(DW_row.dtype)

    # Store derivatives in buffers
    tl.store(DW + offsets, h_row, mask=mask)  # h  = f * g
    tl.store(e + offsets, df_row, mask=mask)  # df = DW * f
    tl.store(g + offsets, de_row, mask=mask)  # de


def _dw_grid(meta, n_elements):
    return (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )


def swiglu_DWf_DW_dfg_kernel(DW, e, g):
    batch_seq_len, hd = e.shape  # Flattened to 2D, so 1st dim is bsz * seq_len
    n_elements = e.numel()

    def grid(meta):
        return _dw_grid(meta, n_elements)

    with torch_gpu_device(e.device):
        _DWf_DW_dfg_kernel[grid](
            DW,
            e,
            g,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
        )
    return DW, e, g
