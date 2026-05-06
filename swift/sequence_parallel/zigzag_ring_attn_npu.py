# Some code borrowed from the awesome work: https://github.com/zhuzilin/ring-flash-attention
# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from functools import cache

from .utils import RingComm

_NPU_BLOCK_MASK_SIZE = 2048
_NPU_FULL_TOKENS = 2147483647
_NPU_TND_SOFTMAX_STAT_REPEAT = 8


def is_npu_tensor(tensor: torch.Tensor) -> bool:
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
    return torch_npu.npu_fusion_attention(**params)


def _call_npu_fusion_attention_grad(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attention_out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    softmax_scale: float,
    dropout_p: float,
    causal: bool,
    window_size,
    deterministic: bool,
):
    import torch_npu

    if not hasattr(torch_npu, 'npu_fusion_attention_grad'):
        raise AttributeError('torch_npu.npu_fusion_attention_grad is not available')
    # Dropout backward needs the exact seed/offset from the original forward,
    # which this ring ctx does not save. Fail instead of using a wrong mask.
    if dropout_p != 0.0:
        raise NotImplementedError('NPU ring attention native backward currently requires dropout_p=0.')

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
    softmax_max, softmax_sum = _npu_softmax_stats_from_global_lse(
        softmax_lse,
        q_tokens=q.shape[0],
        num_heads=q.shape[1],
    )

    params = {
        'query': q,
        'key': k,
        'value': v,
        'dy': dout,
        'head_num': common_kwargs['head_num'],
        'input_layout': common_kwargs['input_layout'],
        'atten_mask': common_kwargs['atten_mask'],
        'softmax_max': softmax_max,
        'softmax_sum': softmax_sum,
        'softmax_in': None,
        'attention_in': (attention_out if torch.is_tensor(attention_out) and attention_out.numel() > 0 else None),
        'scale_value': common_kwargs['scale_value'],
        'keep_prob': common_kwargs['keep_prob'],
        'pre_tockens': common_kwargs['pre_tockens'],
        'next_tockens': common_kwargs['next_tockens'],
        'seed': 0,
        'offset': 0,
        'numels': 0,
        'actual_seq_qlen': common_kwargs['actual_seq_qlen'],
        'actual_seq_kvlen': common_kwargs['actual_seq_kvlen'],
        'sparse_mode': common_kwargs['sparse_mode'],
        'sync': common_kwargs['sync'],
        'softmax_layout': 'TND',
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


def _npu_softmax_stats_from_global_lse(
    softmax_lse_global: torch.Tensor,
    q_tokens: int,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    lse_h_t = _normalize_flash_attn_lse(softmax_lse_global, q_tokens)
    if lse_h_t.shape[0] != num_heads:
        raise RuntimeError(f'Unexpected global lse shape: {tuple(softmax_lse_global.shape)} '
                           f'for q_tokens={q_tokens}, num_heads={num_heads}')

    # With softmax_layout='TND', Ascend returns softmax stats as [T, N, 8].
    # The split-attention backward only needs logsumexp; max=lse and sum=1
    # encode the same value without replaying the block forward.
    lse_t_h = lse_h_t.transpose(0, 1).contiguous().to(torch.float32)
    softmax_max = lse_t_h.unsqueeze(-1).expand(
        q_tokens,
        num_heads,
        _NPU_TND_SOFTMAX_STAT_REPEAT,
    ).contiguous()
    return softmax_max, torch.ones_like(softmax_max)


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
    """Run one native NPU block backward using the final merged ring stats."""
    return _call_npu_fusion_attention_grad(
        block_dout.to(block_q.dtype),
        block_q,
        block_k,
        block_v,
        attention_out=block_out_global,
        softmax_lse=block_lse_global,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=block_causal,
        window_size=window_size,
        deterministic=deterministic,
    )[:3]


def _squeeze_batch(*tensors):
    squeezed = []
    for tensor in tensors:
        if tensor.shape[0] == 1:
            squeezed.append(tensor.squeeze(0))
        else:
            squeezed.append(tensor)
    return tuple(squeezed)


def npu_backward(
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


def npu_forward(
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
