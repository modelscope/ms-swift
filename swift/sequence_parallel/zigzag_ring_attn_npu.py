# Some code borrowed from the awesome work: https://github.com/zhuzilin/ring-flash-attention
# Copyright (c) ModelScope Contributors. All rights reserved.
from functools import cache

import torch

from .utils import RingComm

_NPU_BLOCK_MASK_SIZE = 2048
_NPU_FULL_TOKENS = 2147483647


def _is_npu_tensor(tensor: torch.Tensor) -> bool:
    return tensor.device.type == 'npu'


def _cu_seqlens_to_actual_seq(cu_seqlens: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(x) for x in cu_seqlens[1:].detach().cpu().tolist())


@cache
def _get_npu_causal_mask_cpu() -> torch.Tensor:
    return torch.triu(torch.ones((_NPU_BLOCK_MASK_SIZE, _NPU_BLOCK_MASK_SIZE), dtype=torch.bool), diagonal=1)


def _get_npu_causal_mask(device: torch.device) -> torch.Tensor:
    return _get_npu_causal_mask_cpu().to(device=device)


def _normalize_window_size(window_size):
    if window_size is None:
        return -1, -1
    return window_size


def _get_npu_sparse_params(causal: bool, window_size, device: torch.device) -> dict:
    window_size = _normalize_window_size(window_size)
    if window_size != (-1, -1):
        left, right = window_size
        left = _NPU_FULL_TOKENS if left < 0 else int(left)
        right = _NPU_FULL_TOKENS if right < 0 else int(right)
        if causal:
            right = 0
        return {
            'atten_mask': _get_npu_causal_mask(device),
            'sparse_mode': 4,
            'pre_tockens': left,
            'next_tockens': right,
        }
    if causal:
        return {
            'atten_mask': _get_npu_causal_mask(device),
            'sparse_mode': 3,
            'pre_tockens': _NPU_FULL_TOKENS,
            'next_tockens': _NPU_FULL_TOKENS,
        }
    return {
        'atten_mask': None,
        'sparse_mode': 0,
        'pre_tockens': _NPU_FULL_TOKENS,
        'next_tockens': _NPU_FULL_TOKENS,
    }


def _reshape_npu_lse(lse: torch.Tensor, seqlen_q: int, num_heads: int) -> torch.Tensor:
    """Normalize Ascend softmax stats to flash-attn's [num_heads, seqlen] layout."""
    if lse.dim() == 2:
        if lse.shape == (num_heads, seqlen_q):
            return lse.contiguous()
        if lse.shape == (seqlen_q, num_heads):
            return lse.transpose(0, 1).contiguous()
    elif lse.dim() == 3:
        # Some CANN versions return an extra trailing size-8 axis with repeated
        # stats. Ring merge only needs one copy of each token/head lse.
        if lse.shape[-1] == 8:
            lse = lse[..., 0]
            if lse.shape == (seqlen_q, num_heads):
                return lse.transpose(0, 1).contiguous()
            if lse.shape == (num_heads, seqlen_q):
                return lse.contiguous()
        if lse.shape[0] == seqlen_q:
            return lse.permute(1, 2, 0).reshape(num_heads, seqlen_q).contiguous()
        if lse.shape[1] == seqlen_q:
            return lse.permute(0, 2, 1).reshape(num_heads, seqlen_q).contiguous()
    raise RuntimeError(f'Unexpected NPU lse shape {tuple(lse.shape)} for seqlen_q={seqlen_q}, num_heads={num_heads}')


def _get_npu_attention_common_kwargs(
    q: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    softmax_scale: float,
    dropout_p: float,
    causal: bool,
    window_size,
    deterministic: bool,
) -> dict:
    sparse_params = _get_npu_sparse_params(causal, window_size, q.device)
    return {
        'head_num': q.shape[1],
        'input_layout': 'TND',
        'scale_value': softmax_scale or q.shape[-1]**(-0.5),
        'keep_prob': 1. - dropout_p,
        'actual_seq_qlen': _cu_seqlens_to_actual_seq(cu_seqlens_q),
        'actual_seq_kvlen': _cu_seqlens_to_actual_seq(cu_seqlens_kv),
        'sync': bool(deterministic and dropout_p > 0),
        **sparse_params,
    }


def _call_npu_fusion_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    softmax_scale: float,
    dropout_p: float,
    causal: bool,
    window_size,
    deterministic: bool,
    return_ctx: bool = False,
):
    import torch_npu

    common_kwargs = _get_npu_attention_common_kwargs(
        q,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
    )
    params = {
        'query': q,
        'key': k,
        'value': v,
        'scale': common_kwargs['scale_value'],
        'softmax_layout': 'TND',
    }
    params.update(common_kwargs)
    params.pop('scale_value')
    outputs = torch_npu.npu_fusion_attention(**params)
    if not return_ctx:
        return outputs

    block_out, softmax_max, softmax_sum, softmax_in, seed, offset, numels = outputs
    ctx = {
        **common_kwargs,
        'softmax_max': softmax_max,
        'softmax_sum': softmax_sum,
        'softmax_in': softmax_in,
        'attention_in': block_out,
        'seed': seed,
        'offset': offset,
        'numels': numels,
        'softmax_layout': 'TND',
    }
    return outputs, ctx


def _call_npu_fusion_attention_grad(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: dict,
):
    import torch_npu

    if not hasattr(torch_npu, 'npu_fusion_attention_grad'):
        raise AttributeError('torch_npu.npu_fusion_attention_grad is not available')

    params = {
        'query': q,
        'key': k,
        'value': v,
        'dy': dout,
        'head_num': ctx['head_num'],
        'input_layout': ctx['input_layout'],
        'atten_mask': ctx['atten_mask'],
        'softmax_max': ctx['softmax_max'],
        'softmax_sum': ctx['softmax_sum'],
        'softmax_in': ctx.get('softmax_in'),
        'attention_in': (
            ctx['attention_in']
            if torch.is_tensor(ctx['attention_in']) and ctx['attention_in'].numel() > 0 else None
        ),
        'scale_value': ctx['scale_value'],
        'keep_prob': ctx['keep_prob'],
        'pre_tockens': ctx['pre_tockens'],
        'next_tockens': ctx['next_tockens'],
        'seed': ctx['seed'],
        'offset': ctx['offset'],
        'numels': ctx['numels'],
        'actual_seq_qlen': ctx['actual_seq_qlen'],
        'actual_seq_kvlen': ctx['actual_seq_kvlen'],
        'sparse_mode': ctx['sparse_mode'],
        'sync': ctx['sync'],
        'softmax_layout': ctx['softmax_layout'],
    }
    return torch_npu.npu_fusion_attention_grad(**params)


def _normalize_flash_attn_lse(softmax_lse: torch.Tensor, total_len: int) -> torch.Tensor:
    """Normalize flash-attn lse to [num_heads, total_len]."""
    lse = softmax_lse
    if lse.dim() == 3 and lse.shape[0] == 1:
        lse = lse.squeeze(0)
    if lse.dim() != 2:
        raise RuntimeError(f'Unexpected softmax_lse shape: {tuple(softmax_lse.shape)}')
    if lse.shape[1] != total_len:
        lse = lse.transpose(0, 1).contiguous()
    if lse.shape[1] != total_len:
        raise RuntimeError(f'Unexpected softmax_lse shape: {tuple(softmax_lse.shape)} for total_len={total_len}')
    return lse


def _global_lse_like_npu_softmax(
    softmax_lse_global: torch.Tensor,
    softmax_max: torch.Tensor,
    q_tokens: int,
    num_heads: int,
) -> torch.Tensor:
    """Reshape final ring lse to the local NPU softmax-stat layout."""
    lse_h_t = _normalize_flash_attn_lse(softmax_lse_global, q_tokens)
    if lse_h_t.shape[0] != num_heads:
        raise RuntimeError(
            f'Unexpected global lse shape: {tuple(softmax_lse_global.shape)} '
            f'for q_tokens={q_tokens}, num_heads={num_heads}')

    lse_t_h = lse_h_t.transpose(0, 1).contiguous()
    if softmax_max.shape == lse_t_h.shape:
        return lse_t_h.to(torch.float32)
    if softmax_max.shape == lse_h_t.shape:
        return lse_h_t.to(torch.float32)

    if softmax_max.dim() == 3:
        if softmax_max.shape[:2] == lse_t_h.shape:
            return lse_t_h.unsqueeze(-1).expand_as(softmax_max).to(torch.float32)
        if softmax_max.shape[:2] == lse_h_t.shape:
            return lse_h_t.unsqueeze(-1).expand_as(softmax_max).to(torch.float32)

    if softmax_max.numel() == lse_t_h.numel():
        return lse_t_h.reshape_as(softmax_max).to(torch.float32)

    raise RuntimeError(
        f'Cannot reshape global lse {tuple(lse_h_t.shape)} to NPU softmax stat shape {tuple(softmax_max.shape)}')


def _patch_ctx_with_global_stats(
    ctx: dict,
    global_out_slice: torch.Tensor,
    global_lse_slice: torch.Tensor,
    q_tokens: int,
    num_heads: int,
) -> dict:
    # Native NPU grad expects softmax stats from its own forward block. Ring
    # attention needs gradients for the final merged output, so patch the local
    # stats into an equivalent gauge of the global out/lse.
    local_max = ctx['softmax_max'].to(torch.float32)
    global_lse = _global_lse_like_npu_softmax(global_lse_slice, local_max, q_tokens, num_heads)

    diff = global_lse - local_max
    # softmax_max and softmax_sum are not unique: shifting max by c and scaling
    # sum by exp(-c) keeps logsumexp unchanged. Prefer the local max to avoid
    # feeding very large exp() values to the native grad kernel.
    use_local_gauge = diff < 80.0
    patched_max = torch.where(use_local_gauge, local_max, global_lse)
    patched_sum = torch.where(use_local_gauge, torch.exp(diff), torch.ones_like(diff))

    new_ctx = dict(ctx)
    new_ctx['softmax_max'] = patched_max.to(ctx['softmax_max'].dtype)
    new_ctx['softmax_sum'] = patched_sum.to(ctx['softmax_sum'].dtype)
    new_ctx['attention_in'] = global_out_slice
    return new_ctx


def _get_second_half_lse(softmax_lse: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    total_len = int(cu_seqlens[-1].item())
    lse = _normalize_flash_attn_lse(softmax_lse, total_len)

    # The step > rank branch only differentiates q[half_index1]. Slice the final
    # merged lse per sequence so the native grad ctx sees the same query span.
    second_half_lse = torch.empty((lse.shape[0], lse.shape[1] // 2), dtype=lse.dtype, device=lse.device)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        new_start, new_end = start // 2, end // 2
        start += (end - start) // 2
        second_half_lse[:, new_start:new_end] = lse[:, start:end]
    return second_half_lse


def _npu_block_backward_with_global_stats(
    block_dout,
    block_q,
    block_k,
    block_v,
    block_out_global,
    block_lse_global,
    block_causal,
    cu_seqlens_q,
    cu_seqlens_kv,
    softmax_scale,
    dropout_p,
    window_size,
    deterministic,
):
    """Run one native NPU block backward with ctx rebuilt from its forward kernel."""
    _, block_ctx = _call_npu_fusion_attention(
        block_q,
        block_k,
        block_v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=block_causal,
        window_size=window_size,
        deterministic=deterministic,
        return_ctx=True,
    )
    block_ctx = _patch_ctx_with_global_stats(
        block_ctx,
        global_out_slice=block_out_global,
        global_lse_slice=block_lse_global,
        q_tokens=block_q.shape[0],
        num_heads=block_q.shape[1],
    )
    return _call_npu_fusion_attention_grad(block_dout.to(block_q.dtype), block_q, block_k, block_v, block_ctx)[:3]


def _squeeze_batch(*tensors):
    squeezed = []
    for tensor in tensors:
        if tensor.shape[0] == 1:
            squeezed.append(tensor.squeeze(0))
        else:
            squeezed.append(tensor)
    return tuple(squeezed)


def _npu_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens,
    max_seqlen,
    half_index0,
    half_index1,
    softmax_scale,
    dropout_p=0.0,
    window_size=(-1, -1),
    deterministic=False,
):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dout, q, k, v, out, softmax_lse = _squeeze_batch(dout, q, k, v, out, softmax_lse)
    cu_seqlens = cu_seqlens // kv_comm.world_size
    del max_seqlen

    half_cu_seqlens = cu_seqlens // 2
    q1 = q[half_index1]
    dout1 = dout[half_index1]
    out1 = out[half_index1]
    softmax_lse1 = _get_second_half_lse(softmax_lse, cu_seqlens)

    dq = torch.zeros_like(q, dtype=torch.float32)
    current_step_dk = torch.empty_like(k, dtype=torch.float32)
    current_step_dv = torch.empty_like(v, dtype=torch.float32)
    next_dk = next_dv = None

    for step in range(kv_comm.world_size):
        current_step_dk.zero_()
        current_step_dv.zero_()
        if step == 0:
            bdq, bdk, bdv = _npu_block_backward_with_global_stats(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                True,
                cu_seqlens,
                cu_seqlens,
                softmax_scale,
                dropout_p,
                window_size,
                deterministic,
            )
            dq += bdq.to(torch.float32)
            current_step_dk += bdk.to(torch.float32)
            current_step_dv += bdv.to(torch.float32)
        elif step <= kv_comm.rank:
            k0 = k[half_index0]
            v0 = v[half_index0]
            bdq, bdk, bdv = _npu_block_backward_with_global_stats(
                dout,
                q,
                k0,
                v0,
                out,
                softmax_lse,
                False,
                cu_seqlens,
                half_cu_seqlens,
                softmax_scale,
                dropout_p,
                window_size,
                deterministic,
            )
            dq += bdq.to(torch.float32)
            current_step_dk[half_index0] += bdk.to(torch.float32)
            current_step_dv[half_index0] += bdv.to(torch.float32)
        else:
            bdq, bdk, bdv = _npu_block_backward_with_global_stats(
                dout1,
                q1,
                k,
                v,
                out1,
                softmax_lse1,
                False,
                half_cu_seqlens,
                cu_seqlens,
                softmax_scale,
                dropout_p,
                window_size,
                deterministic,
            )
            dq[half_index1] += bdq.to(torch.float32)
            current_step_dk += bdk.to(torch.float32)
            current_step_dv += bdv.to(torch.float32)

        # K/V gradients are owned by the rank that originally held that shard.
        # Rotate the accumulated gradients in the opposite ring direction until
        # each owner receives its final dk/dv.
        if step == 0:
            dk = current_step_dk
            dv = current_step_dv
        else:
            dk = next_dk
            dv = next_dv
            dk += current_step_dk
            dv += current_step_dv

        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)
        d_kv_comm.wait()

        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)
            kv_comm.wait()
            k, v = next_k, next_v

    return dq.to(q.dtype).unsqueeze(0), next_dk.to(q.dtype).unsqueeze(0), next_dv.to(q.dtype).unsqueeze(0)


def _npu_forward(
    q,
    k,
    v,
    causal,
    cu_seqlens_q,
    cu_seqlens_kv,
    dropout_p,
    softmax_scale,
    deterministic,
    window_size,
):
    outputs = _call_npu_fusion_attention(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
    )
    block_out, softmax_max, softmax_sum = outputs[:3]
    block_lse = softmax_max.to(torch.float32) + torch.log(softmax_sum.to(torch.float32))
    block_lse = _reshape_npu_lse(block_lse, q.shape[0], q.shape[1])
    return block_out, block_lse
