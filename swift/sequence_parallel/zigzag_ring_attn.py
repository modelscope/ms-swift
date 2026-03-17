# Some code borrowed from the awesome work: https://github.com/zhuzilin/ring-flash-attention
# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import torch
import torch.distributed as dist
import torch.nn.functional as F
from functools import cache

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
    if lse.dim() == 2:
        if lse.shape == (num_heads, seqlen_q):
            return lse.contiguous()
        if lse.shape == (seqlen_q, num_heads):
            return lse.transpose(0, 1).contiguous()
    elif lse.dim() == 3:
        # Ascend ring-attention related outputs commonly use an extra size-8
        # trailing axis whose values are duplicated for numerical updates.
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
        **{k: v for k, v in common_kwargs.items() if k != 'scale_value'},
    }
    used_softmax_layout = True
    try:
        outputs = torch_npu.npu_fusion_attention(**params)
    except TypeError as exc:
        if 'softmax_layout' not in str(exc):
            raise
        params.pop('softmax_layout', None)
        used_softmax_layout = False
        outputs = torch_npu.npu_fusion_attention(**params)
    if not return_ctx:
        return outputs

    block_out, softmax_max, softmax_sum, attention_in, seed, offset, numels = outputs
    ctx = {
        **common_kwargs,
        'softmax_max': softmax_max,
        'softmax_sum': softmax_sum,
        'attention_in': attention_in,
        'seed': seed,
        'offset': offset,
        'numels': numels,
        'softmax_layout': 'TND' if used_softmax_layout else '',
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
        'attention_in': ctx['attention_in'] if torch.is_tensor(ctx['attention_in']) and ctx['attention_in'].numel() > 0 else None,
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
    }
    if ctx.get('softmax_layout'):
        params['softmax_layout'] = ctx['softmax_layout']
    try:
        return torch_npu.npu_fusion_attention_grad(**params)
    except TypeError as exc:
        if 'softmax_layout' not in str(exc):
            raise
        params.pop('softmax_layout', None)
        return torch_npu.npu_fusion_attention_grad(**params)


def _get_npu_manual_attention_mask(causal: bool, window_size, q_len: int, k_len: int, device: torch.device) -> torch.Tensor | None:
    window_size = _normalize_window_size(window_size)
    offset = k_len - q_len
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    k_idx = torch.arange(k_len, device=device).unsqueeze(0)
    mask = None
    if causal:
        mask = k_idx > (q_idx + offset)
    if window_size != (-1, -1):
        left, right = window_size
        if left >= 0:
            left_mask = k_idx < (q_idx + offset - left)
            mask = left_mask if mask is None else (mask | left_mask)
        if right >= 0:
            right_mask = k_idx > (q_idx + offset + right)
            mask = right_mask if mask is None else (mask | right_mask)
    return mask


def _normalize_softmax_lse(softmax_lse: torch.Tensor, q_tokens: int, num_heads: int) -> torch.Tensor:
    if softmax_lse.dim() == 3:
        softmax_lse = softmax_lse.squeeze(0)
    if softmax_lse.dim() != 2:
        raise RuntimeError(f'Unexpected softmax_lse shape: {tuple(softmax_lse.shape)}')
    if softmax_lse.shape == (num_heads, q_tokens):
        return softmax_lse.transpose(0, 1).contiguous()
    if softmax_lse.shape == (q_tokens, num_heads):
        return softmax_lse.contiguous()
    raise RuntimeError(
        f'Unexpected softmax_lse shape: {tuple(softmax_lse.shape)} for q_tokens={q_tokens}, num_heads={num_heads}')


def _manual_varlen_attention_backward(dout,
                                      q,
                                      k,
                                      v,
                                      out,
                                      softmax_lse,
                                      cu_seqlens_q,
                                      cu_seqlens_kv,
                                      softmax_scale,
                                      causal,
                                      window_size):
    scale = softmax_scale or q.shape[-1]**(-0.5)
    num_heads_q = q.shape[1]
    num_heads_kv = k.shape[1]
    groups = num_heads_q // num_heads_kv
    assert groups * num_heads_kv == num_heads_q
    softmax_lse = _normalize_softmax_lse(softmax_lse, q.shape[0], num_heads_q)

    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    for i in range(len(cu_seqlens_q) - 1):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_start, k_end = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()
        q_seq = q[q_start:q_end].to(torch.float32)
        k_seq = k[k_start:k_end].to(torch.float32)
        v_seq = v[k_start:k_end].to(torch.float32)
        out_seq = out[q_start:q_end].to(torch.float32)
        dout_seq = dout[q_start:q_end].to(torch.float32)
        lse_seq = softmax_lse[q_start:q_end].transpose(0, 1).unsqueeze(-1).to(torch.float32)

        if groups > 1:
            k_seq_expanded = k_seq.repeat_interleave(groups, dim=1)
            v_seq_expanded = v_seq.repeat_interleave(groups, dim=1)
        else:
            k_seq_expanded = k_seq
            v_seq_expanded = v_seq

        scores = torch.einsum('qhd,khd->hqk', q_seq, k_seq_expanded) * scale
        mask = _get_npu_manual_attention_mask(causal, window_size, q_seq.shape[0], k_seq.shape[0], scores.device)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        probs = torch.exp(scores - lse_seq)
        delta = (out_seq * dout_seq).sum(dim=-1).transpose(0, 1).unsqueeze(-1)

        d_v_expanded = torch.einsum('hqk,qhd->khd', probs, dout_seq)
        d_probs = torch.einsum('qhd,khd->hqk', dout_seq, v_seq_expanded)
        d_scores = probs * (d_probs - delta)
        d_q = torch.einsum('hqk,khd->qhd', d_scores, k_seq_expanded) * scale
        d_k_expanded = torch.einsum('hqk,qhd->khd', d_scores, q_seq) * scale

        if groups > 1:
            d_k = d_k_expanded.reshape(k_seq.shape[0], num_heads_kv, groups, k_seq.shape[-1]).sum(dim=2)
            d_v = d_v_expanded.reshape(v_seq.shape[0], num_heads_kv, groups, v_seq.shape[-1]).sum(dim=2)
        else:
            d_k = d_k_expanded
            d_v = d_v_expanded

        dq[q_start:q_end] = d_q
        dk[k_start:k_end] = d_k
        dv[k_start:k_end] = d_v
    return dq, dk, dv


def _manual_varlen_lse_backward(dlse, q, k, cu_seqlens_q, cu_seqlens_kv, softmax_scale, causal, window_size):
    scale = softmax_scale or q.shape[-1]**(-0.5)
    num_heads_q = q.shape[1]
    num_heads_kv = k.shape[1]
    groups = num_heads_q // num_heads_kv
    assert groups * num_heads_kv == num_heads_q

    if dlse.dim() == 3:
        dlse = dlse.squeeze(-1)
    assert dlse.shape == (q.shape[0], num_heads_q)

    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)

    for i in range(len(cu_seqlens_q) - 1):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_start, k_end = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()
        q_seq = q[q_start:q_end].to(torch.float32)
        k_seq = k[k_start:k_end].to(torch.float32)
        dlse_seq = dlse[q_start:q_end].transpose(0, 1).to(torch.float32)

        if groups > 1:
            k_seq_expanded = k_seq.repeat_interleave(groups, dim=1)
        else:
            k_seq_expanded = k_seq

        scores = torch.einsum('qhd,khd->hqk', q_seq, k_seq_expanded) * scale
        mask = _get_npu_manual_attention_mask(causal, window_size, q_seq.shape[0], k_seq.shape[0], scores.device)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        d_scores = dlse_seq.unsqueeze(-1) * probs
        d_q = torch.einsum('hqk,khd->qhd', d_scores, k_seq_expanded) * scale
        d_k_expanded = torch.einsum('hqk,qhd->khd', d_scores, q_seq) * scale

        if groups > 1:
            d_k = d_k_expanded.reshape(k_seq.shape[0], num_heads_kv, groups, k_seq.shape[-1]).sum(dim=2)
        else:
            d_k = d_k_expanded

        dq[q_start:q_end] += d_q
        dk[k_start:k_end] += d_k
    return dq, dk


def _manual_varlen_attention_forward(q, k, v, cu_seqlens_q, cu_seqlens_kv, softmax_scale, causal, window_size):
    scale = softmax_scale or q.shape[-1]**(-0.5)
    num_heads_q = q.shape[1]
    num_heads_kv = k.shape[1]
    groups = num_heads_q // num_heads_kv
    assert groups * num_heads_kv == num_heads_q

    outputs = []
    lses = []
    for i in range(len(cu_seqlens_q) - 1):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_start, k_end = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()
        q_seq = q[q_start:q_end].to(torch.float32)
        k_seq = k[k_start:k_end].to(torch.float32)
        v_seq = v[k_start:k_end].to(torch.float32)

        if groups > 1:
            k_seq_expanded = k_seq.repeat_interleave(groups, dim=1)
            v_seq_expanded = v_seq.repeat_interleave(groups, dim=1)
        else:
            k_seq_expanded = k_seq
            v_seq_expanded = v_seq

        scores = torch.einsum('qhd,khd->hqk', q_seq, k_seq_expanded) * scale
        mask = _get_npu_manual_attention_mask(causal, window_size, q_seq.shape[0], k_seq.shape[0], scores.device)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum('hqk,khd->qhd', probs, v_seq_expanded))
        lses.append(torch.logsumexp(scores, dim=-1))

    return torch.cat(outputs, dim=0).contiguous(), torch.cat(lses, dim=1).contiguous()


def _all_gather_step_grads(step_grads: torch.Tensor, process_group) -> list[torch.Tensor]:
    gathered = [torch.empty_like(step_grads) for _ in range(dist.get_world_size(process_group))]
    dist.all_gather(gathered, step_grads.contiguous(), group=process_group)
    return gathered


def _zigzag_ring_flash_attn_varlen_backward_exact(
    process_group,
    dout,
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
    half_index0,
    half_index1,
    softmax_scale,
    window_size,
):
    kv_comm = RingComm(process_group)
    dout, q, k, v = squeeze_batch(dout, q, k, v)
    cu_seqlens = cu_seqlens // kv_comm.world_size
    max_seqlen = max_seqlen // kv_comm.world_size
    block_seq_len = q.shape[0] // 2
    half_cu_seqlens = cu_seqlens // 2

    def _get_block_cu_seqlens(seqlen_q, seqlen_kv):
        cu_seqlens_q = half_cu_seqlens if seqlen_q == block_seq_len else cu_seqlens
        cu_seqlens_kv = half_cu_seqlens if seqlen_kv == block_seq_len else cu_seqlens
        return cu_seqlens_q, cu_seqlens_kv

    with torch.enable_grad():
        q_replay = q.detach().requires_grad_(True)
        current_k = k.detach().requires_grad_(True)
        current_v = v.detach().requires_grad_(True)
        step_ks = []
        step_vs = []
        merged_out = None
        merged_lse = None

        for step in range(kv_comm.world_size):
            step_ks.append(current_k)
            step_vs.append(current_v)
            if step + 1 != kv_comm.world_size:
                next_k, next_v = kv_comm.send_recv_kv(current_k.detach(), current_v.detach())

            if step == 0:
                block_q = q_replay
                block_k = current_k
                block_v = current_v
                block_causal = True
            elif step <= kv_comm.rank:
                block_q = q_replay
                block_k = current_k[half_index0]
                block_v = current_v[half_index0]
                block_causal = False
            else:
                block_q = q_replay[half_index1]
                block_k = current_k
                block_v = current_v
                block_causal = False

            cu_seqlens_q, cu_seqlens_kv = _get_block_cu_seqlens(block_q.shape[0], block_k.shape[0])
            block_out, block_lse = _manual_varlen_attention_forward(
                block_q,
                block_k,
                block_v,
                cu_seqlens_q,
                cu_seqlens_kv,
                softmax_scale,
                block_causal,
                window_size,
            )

            if step == 0 or step <= kv_comm.rank:
                merged_out, merged_lse, _ = update_out_and_lse(merged_out, merged_lse, block_out, block_lse)
            else:
                merged_out[half_index1], merged_lse[half_index1], _ = update_out_and_lse(
                    merged_out[half_index1],
                    merged_lse[half_index1],
                    block_out,
                    block_lse,
                )

            if step + 1 != kv_comm.world_size:
                kv_comm.wait()
                current_k = next_k.detach().requires_grad_(True)
                current_v = next_v.detach().requires_grad_(True)

        grads = torch.autograd.grad(
            merged_out,
            [q_replay, *step_ks, *step_vs],
            grad_outputs=dout.to(merged_out.dtype),
        )

    dq = grads[0].to(torch.float32)
    num_steps = kv_comm.world_size
    step_dk = torch.stack([grad.to(torch.float32) for grad in grads[1:1 + num_steps]], dim=0)
    step_dv = torch.stack([grad.to(torch.float32) for grad in grads[1 + num_steps:]], dim=0)

    gathered_dk = _all_gather_step_grads(step_dk, process_group)
    gathered_dv = _all_gather_step_grads(step_dv, process_group)
    dk = sum(gathered_dk[(kv_comm.rank + step) % kv_comm.world_size][step] for step in range(num_steps))
    dv = sum(gathered_dv[(kv_comm.rank + step) % kv_comm.world_size][step] for step in range(num_steps))

    return dq.to(q.dtype).unsqueeze(0), dk.to(q.dtype).unsqueeze(0), dv.to(q.dtype).unsqueeze(0)


def _npu_forward(q,
                 k,
                 v,
                 causal,
                 cu_seqlens_q,
                 cu_seqlens_kv,
                 dropout_p,
                 softmax_scale,
                 deterministic,
                 window_size,
                 return_ctx: bool = False):
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
        return_ctx=return_ctx,
    )
    ctx = None
    if return_ctx:
        outputs, ctx = outputs
    block_out, softmax_max, softmax_sum = outputs[:3]
    block_lse = softmax_max.to(torch.float32) + torch.log(softmax_sum.to(torch.float32))
    block_lse = _reshape_npu_lse(block_lse, q.shape[0], q.shape[1])
    if return_ctx:
        return block_out, block_lse, ctx
    return block_out, block_lse


def _npu_backward(dout, q, k, v, out, softmax_lse, causal, cu_seqlens_q, cu_seqlens_kv, dq_buffer, dk_buffer, dv_buffer, dropout_p,
                  softmax_scale, deterministic, window_size, backend_ctx=None, block_dlse=None):
    dq = dk = dv = None
    ctx = backend_ctx
    use_native_grad = ctx is not None and ctx.get('input_layout') in ('BSH', 'BNSD')
    if use_native_grad:
        grad_outputs = _call_npu_fusion_attention_grad(dout.to(q.dtype), q, k, v, ctx)
        dq, dk, dv = grad_outputs[:3]
    else:
        dq, dk, dv = _manual_varlen_attention_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_kv,
            softmax_scale,
            causal,
            window_size,
        )

    dq_buffer.zero_()
    dk_buffer.zero_()
    dv_buffer.zero_()
    dq_buffer[:dq.shape[0]].copy_(dq.to(dq_buffer.dtype))
    dk_buffer[:dk.shape[0]].copy_(dk.to(dk_buffer.dtype))
    dv_buffer[:dv.shape[0]].copy_(dv.to(dv_buffer.dtype))
    if block_dlse is not None:
        dlse_dq, dlse_dk = _manual_varlen_lse_backward(
            block_dlse,
            q,
            k,
            cu_seqlens_q,
            cu_seqlens_kv,
            softmax_scale,
            causal,
            window_size,
        )
        dq_buffer[:dlse_dq.shape[0]].add_(dlse_dq.to(dq_buffer.dtype))
        dk_buffer[:dlse_dk.shape[0]].add_(dlse_dk.to(dk_buffer.dtype))


def get_half_index(cu_seqlens, *, front: bool):
    """Get half of the index

    Args:
        cu_seqlens: The cu_seqlens passed into flash_attn
        front: The head part or the tail part

    Returns:
        The slice or the tensor mask.
    """

    if len(cu_seqlens) == 2:
        if front:
            return slice(None, cu_seqlens[-1] // 2)
        else:
            return slice(cu_seqlens[-1] // 2, None)

    index = torch.zeros((cu_seqlens[-1].item(), ), dtype=torch.bool)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        if front:
            end = (start + end) // 2
        else:
            start = (start + end) // 2
        index[start:end] = True
    return index


@torch.jit.script
def get_half_lse(lse, cu_seqlens, *, front: bool):
    """Get half of the lse

    Args:
        lse: The input lse, with shape [num_heads, seqlen]
        cu_seqlens: The cu_seqlens passed into flash_attn
        front: The head part or the tail part

    Returns:
        The filtered lse with the same shape as lse
    """
    new_lse = torch.empty(
        (lse.shape[0], lse.shape[1] // 2),
        dtype=lse.dtype,
        device=lse.device,
    )
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        new_start, new_end = start // 2, end // 2
        if front:
            end -= (end - start) // 2
        else:
            start += (end - start) // 2
        new_lse[:, new_start:new_end] = lse[:, start:end]
    return new_lse


def update_out_and_lse(out, lse, block_out, block_lse):
    """Update output and lse:
    new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795

    Args:
        out: The accumulated output of shape [seqlen, num_heads, hidden_size]
        lse: The accumulated lse of shape [num_heads, seqlen]
        block_out: The current block output of shape [seqlen, num_heads, hidden_size]
        block_lse: The current block lse of shape [num_heads, seqlen]

    Returns:
        The updated output[seqlen, num_heads, hidden_size] and lse (shape: [seqlen, num_heads, 1]) and
            the intermediate value of torch.sigmoid(block_lse - lse) (shape: [seqlen, num_heads, 1])
    """
    if out is None:
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
        sig_diff = None
    else:
        block_out = block_out.to(torch.float32)
        block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

        diff = block_lse - lse
        sig_diff = torch.sigmoid(diff)

        out = out - sig_diff * (out - block_out)  # (..., D)
        lse = lse - F.logsigmoid(lse - block_lse)  # (..., 1)
    return out, lse, sig_diff


@cache
def _get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None, ) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if 'softcap' in args:
        args['softcap'] = 0.0
    return args


def get_default_args(func):
    if inspect.isfunction(func):
        return _get_default_args(func)
    else:
        # Use the origin _init_fn in CustomOpDef
        return _get_default_args(func._init_fn)


def squeeze_batch(*t):
    """Squeeze the batch dim of tensors"""
    tensors = []
    for sub in t:
        if sub.shape[0] == 1:
            tensors.append(sub.squeeze(0))
        else:
            tensors.append(sub)
    return tuple(tensors)


def padding(tensor, cu_seqlens, padding_value, front):
    """Pad the tensor according to the cu_seqlens

    Args:
        tensor: The input tensor of shape [seqlen, *]
        cu_seqlens: The cu_seqlens
        padding_value: The padding value
        front: tensor is the head or tail part
    """
    if len(cu_seqlens) == 2:
        if front:
            return torch.cat((tensor, torch.full_like(tensor, padding_value).to(tensor.dtype).to(tensor.device)), dim=0)
        else:
            return torch.cat((torch.full_like(tensor, padding_value).to(tensor.dtype).to(tensor.device), tensor), dim=0)

    output = []
    acc = 0
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        half_len = (end - start) // 2
        acc += half_len
        half_start = start // 2
        local_tensor = tensor[half_start:half_start + half_len]
        if front:
            output.append(local_tensor)
            output.append(torch.full_like(local_tensor, padding_value).to(local_tensor.dtype).to(local_tensor.device))
        else:
            output.append(torch.full_like(local_tensor, padding_value).to(local_tensor.dtype).to(local_tensor.device))
            output.append(local_tensor)
    assert acc == tensor.shape[0]
    return torch.cat(output)


def forward(q, k, v, causal, cu_seqlens, max_seqlen, block_seq_len, dropout_p, softmax_scale, alibi_slopes,
            window_size, return_ctx: bool = False):
    seqlen_q = q.shape[0]
    seqlen_kv = k.shape[0]
    half_cu_seqlens = cu_seqlens // 2
    half_max_seqlen = max_seqlen // 2
    cu_seqlens_q = half_cu_seqlens if seqlen_q == block_seq_len else cu_seqlens
    max_seqlen_q = half_max_seqlen if seqlen_q == block_seq_len else max_seqlen
    cu_seqlens_kv = half_cu_seqlens if seqlen_kv == block_seq_len else cu_seqlens
    max_seqlen_kv = half_max_seqlen if seqlen_kv == block_seq_len else max_seqlen
    assert k.shape[-0] == cu_seqlens_kv[-1]
    assert q.shape[-0] == cu_seqlens_q[-1]
    assert max_seqlen_q == (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    assert max_seqlen_kv == (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).max().item()

    if _is_npu_tensor(q):
        return _npu_forward(
            q,
            k,
            v,
            causal,
            cu_seqlens_q,
            cu_seqlens_kv,
            dropout_p,
            softmax_scale,
            deterministic=False,
            window_size=window_size,
            return_ctx=return_ctx,
        )

    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward
    params = get_default_args(_flash_attn_varlen_forward).copy()
    params.update({
        'q': q,
        'k': k,
        'v': v,
        # the first half and the second half are the same
        'cu_seqlens_q': cu_seqlens_q,
        'cu_seqlens_k': cu_seqlens_kv,
        'max_seqlen_q': max_seqlen_q,
        'max_seqlen_k': max_seqlen_kv,
        'dropout_p': dropout_p,
        'softmax_scale': softmax_scale,
        'causal': causal,
        'alibi_slopes': alibi_slopes,
        'return_softmax': True and dropout_p > 0,
    })
    if 'window_size' in params:
        params.update({'window_size': window_size})
    else:
        params.update({
            'window_size_left': window_size[0],
            'window_size_right': window_size[1],
        })
    outputs = _flash_attn_varlen_forward(**params)
    if len(outputs) == 8:
        block_out, _, _, _, _, block_lse, _, _ = outputs
    else:
        assert len(outputs) == 4
        block_out, block_lse, _, _ = outputs
    if return_ctx:
        return block_out, block_lse, None
    return block_out, block_lse


def backward(dout, q, k, v, out, softmax_lse, causal, cu_seqlens, max_seqlen, block_seq_len, dq_buffer, dk_buffer,
             dv_buffer, dropout_p, softmax_scale, alibi_slopes, deterministic, window_size, backend_ctx=None,
             block_dlse=None):
    seqlen_q = q.shape[0]
    seqlen_kv = k.shape[0]

    half_cu_seqlens = cu_seqlens // 2
    half_max_seqlen = max_seqlen // 2
    cu_seqlens_q = half_cu_seqlens if seqlen_q == block_seq_len else cu_seqlens
    max_seqlen_q = half_max_seqlen if seqlen_q == block_seq_len else max_seqlen
    cu_seqlens_kv = half_cu_seqlens if seqlen_kv == block_seq_len else cu_seqlens
    max_seqlen_kv = half_max_seqlen if seqlen_kv == block_seq_len else max_seqlen
    assert dout.shape[0] == q.shape[0]
    assert dout.shape[0] == out.shape[0]
    assert softmax_lse.shape[1] == q.shape[0]
    assert k.shape[0] == cu_seqlens_kv[-1]
    assert q.shape[0] == cu_seqlens_q[-1]
    assert max_seqlen_q == (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    assert max_seqlen_kv == (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).max().item()

    if _is_npu_tensor(q):
        _npu_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            causal,
            cu_seqlens_q,
            cu_seqlens_kv,
            dq_buffer,
            dk_buffer,
            dv_buffer,
            dropout_p,
            softmax_scale,
            deterministic,
            window_size,
            backend_ctx,
            block_dlse,
        )
        return

    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward
    params = get_default_args(_flash_attn_varlen_backward).copy()
    params.update({
        'dout': dout,
        'q': q,
        'k': k,
        'v': v,
        'out': out,
        'softmax_lse': softmax_lse,
        'dq': dq_buffer[:seqlen_q],
        'dk': dk_buffer[:seqlen_kv],
        'dv': dv_buffer[:seqlen_kv],
        # the first half and the second half are the same
        'cu_seqlens_q': cu_seqlens_q,
        'cu_seqlens_k': cu_seqlens_kv,
        'max_seqlen_q': max_seqlen_q,
        'max_seqlen_k': max_seqlen_kv,
        'dropout_p': dropout_p,
        'softmax_scale': softmax_scale,
        'causal': causal,
        'alibi_slopes': alibi_slopes,
        'deterministic': deterministic,
    })
    if 'window_size' in params:
        params.update({'window_size': window_size})
    else:
        params.update({
            'window_size_left': window_size[0],
            'window_size_right': window_size[1],
        })
    _flash_attn_varlen_backward(**params)
    if block_dlse is not None:
        dlse_dq, dlse_dk = _manual_varlen_lse_backward(
            block_dlse,
            q,
            k,
            cu_seqlens_q,
            cu_seqlens_kv,
            softmax_scale,
            causal,
            window_size,
        )
        dq_buffer[:dlse_dq.shape[0]].add_(dlse_dq.to(dq_buffer.dtype))
        dk_buffer[:dlse_dk.shape[0]].add_(dlse_dk.to(dk_buffer.dtype))


def lse_grad(out, lse, block_out, block_lse, sig, grad_out, grad_lse):
    """Calculate the grad of each block.

    Args:
        out: The accumulated output of shape [seqlen, num_heads, hidden_size]
        lse: The accumulated lse of shape [num_heads, seqlen, 1]
        block_out: The current block output of shape [seqlen, num_heads, hidden_size]
        block_lse: The current block lse of shape [num_heads, seqlen, 1]
        grad_out: The input grad of output of the current block shape [seqlen, num_heads, hidden_size]
        grad_lse: The input grad of lse of the current block shape [num_heads, seqlen, 1]

    Returns:
        The accumulated grad of out and lse, and the grad of out and lse of the current block
    """
    grad_out_input = grad_out * (1 - sig)

    grad_block_out = grad_out * sig

    d_new_out_d_lse = (out - block_out) * (sig * (1 - sig))
    grad_lse_input = (grad_out * d_new_out_d_lse).sum(dim=-1, keepdim=True)
    grad_lse_input_final = grad_lse_input + grad_lse * torch.sigmoid(lse - block_lse)

    grad_block_lse = -grad_lse_input_final + grad_lse

    return grad_out_input, grad_lse_input_final, grad_block_out, grad_block_lse


def zigzag_ring_flash_attn_varlen_forward(
        process_group,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens,
        max_seqlen,
        half_index0,
        half_index1,
        softmax_scale,
        dropout_p=0,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
):
    assert causal, 'zigzag ring is meaningless for causal=False'
    comm = RingComm(process_group)
    q, k, v = squeeze_batch(q, k, v)
    q1 = q[half_index1]
    # Input cu_seqlens is the total length, divided by world_size to fit the split ones
    cu_seqlens = cu_seqlens // comm.world_size
    # Same with above
    max_seqlen = max_seqlen // comm.world_size
    block_seq_len = q.shape[0] // 2
    out = None
    lse = None
    next_k, next_v = None, None

    for step in range(comm.world_size):
        # from step 0 to the last
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)
        """
        world_size = 4, total 8 parts
        0/7 is group0
        1/6 is group1
        2/5 is group2
        3/4 is group3
        consider 1/6，take the query as the left axis, key as the top axis:
        step 0:
           1   6
        1  ✅  ❎
        6  ✅  ✅
        all needed, causal=True
        step 1(step <= comm.rank):
           0   7
        1  ✅  ❎
        6  ✅  ❎
        the first part of kv is needed, causal=False
        step 2(step > comm.rank):
           3   4
        1  ❎  ❎
        6  ✅  ✅
        the second part of q is needed, causal=False
        """
        # Here block_lse shape: [num_heads, seqlen]
        # lse shape: [seqlen, num_heads, 1]
        if step == 0:
            block_out, block_lse = forward(q, k, v, True, cu_seqlens, max_seqlen, block_seq_len, dropout_p,
                                           softmax_scale, alibi_slopes, window_size)
            out, lse, sig_diff = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            k0 = k[half_index0]
            v0 = v[half_index0]
            block_out, block_lse = forward(q, k0, v0, False, cu_seqlens, max_seqlen, block_seq_len, dropout_p,
                                           softmax_scale, alibi_slopes, window_size)
            out, lse, sig_diff = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = forward(q1, k, v, False, cu_seqlens, max_seqlen, block_seq_len, dropout_p,
                                           softmax_scale, alibi_slopes, window_size)
            out[half_index1], lse[half_index1], sig_diff = update_out_and_lse(out[half_index1], lse[half_index1],
                                                                              block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(0, 1)  # [num_heads, seqlen]
    return out.unsqueeze(0), lse.unsqueeze(0)


def zigzag_ring_flash_attn_varlen_backward(
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
        dropout_p=0,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
):
    assert causal, 'zigzag ring is meaningless for causal=False'
    if _is_npu_tensor(q):
        return _zigzag_ring_flash_attn_varlen_backward_exact(
            process_group,
            dout,
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            half_index0,
            half_index1,
            softmax_scale,
            window_size,
        )
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer = dv_comm_buffer = None

    # squeeze the axis of batch
    dout, q, k, v, out, softmax_lse = squeeze_batch(dout, q, k, v, out, softmax_lse)
    # Input cu_seqlens is the total length, divided by world_size to fit the split ones
    cu_seqlens = cu_seqlens // kv_comm.world_size
    # Same as above
    max_seqlen = max_seqlen // kv_comm.world_size
    dout1 = dout[half_index1]
    q1 = q[half_index1]
    out1 = out[half_index1]
    softmax_lse1 = get_half_lse(softmax_lse, cu_seqlens, front=False)
    # half of the part
    block_seq_len = q.shape[0] // 2

    # repeatly allocating buffer may be slow...
    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if step == 0:
            backward(
                dout.to(dout.dtype), q, k, v, out, softmax_lse, True, cu_seqlens, max_seqlen, block_seq_len,
                dq_buffer, dk_buffer, dv_buffer, dropout_p, softmax_scale, alibi_slopes, deterministic, window_size)
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
            if dq.isnan().any() or dk.isnan().any() or dv.isnan().any():
                raise
        else:
            if step <= kv_comm.rank:
                k0 = k[half_index0]
                v0 = v[half_index0]
                backward(
                    dout.to(dout.dtype), q, k0, v0, out, softmax_lse, False, cu_seqlens, max_seqlen,
                    block_seq_len, dq_buffer, dk_buffer, dv_buffer, dropout_p, softmax_scale, alibi_slopes,
                    deterministic, window_size)
                dq += dq_buffer
            else:
                backward(dout1.to(dout.dtype), q1, k, v, out1, softmax_lse1, False, cu_seqlens, max_seqlen,
                         block_seq_len, dq_buffer, dk_buffer, dv_buffer, dropout_p, softmax_scale, alibi_slopes,
                         deterministic, window_size)
                # only need to add to the tail half, because the head half does not match the causal condition
                dq[half_index1] += dq_buffer[:block_seq_len]

            d_kv_comm.wait()
            dk_comm_buffer = torch.empty_like(dk)
            dv_comm_buffer = torch.empty_like(dv)
            dk_comm_buffer.copy_(dk)
            dv_comm_buffer.copy_(dv)
            # next_dk, next_dv comes from a previous gpu, add kv grad to them, and pass them to the next gpu
            dk, dv = next_dk, next_dv

            if step <= kv_comm.rank:
                # only need to add to the head part, because the tail part does not match the causal condition
                dk[half_index0] += dk_buffer[:block_seq_len]
                dv[half_index0] += dv_buffer[:block_seq_len]
            else:
                dk += dk_buffer
                dv += dv_buffer
            if dq.isnan().any() or dk.isnan().any() or dv.isnan().any():
                raise
        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv, dk_comm_buffer, dv_comm_buffer)

    d_kv_comm.wait()

    return dq.to(q.dtype).unsqueeze(0), next_dk.to(q.dtype).unsqueeze(0), next_dv.to(q.dtype).unsqueeze(0)


class ZigZagRingFlashAttnVarlenFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1]**(-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        rp_world_size = dist.get_world_size(group)
        half_index0 = get_half_index(cu_seqlens // rp_world_size, front=True)
        half_index1 = get_half_index(cu_seqlens // rp_world_size, front=False)
        out, softmax_lse = zigzag_ring_flash_attn_varlen_forward(
            group,
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            half_index0,
            half_index1,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        is_half_index_tensor = isinstance(half_index0, torch.Tensor)
        ctx.is_half_index_tensor = is_half_index_tensor
        if is_half_index_tensor:
            """
            Shapes:
            qkv: [1, seqlen, num_heads, hidden_size]
            out: [1, seqlen, num_heads, hidden_size]
            softmax_lse: [1, num_heads, seqlen]
            """
            ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens, half_index0, half_index1)
        else:
            ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens)
            ctx.half_index0 = half_index0
            ctx.half_index1 = half_index1
        ctx.max_seqlen = max_seqlen
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        if ctx.is_half_index_tensor:
            (q, k, v, out, softmax_lse, cu_seqlens, half_index0, half_index1) = (ctx.saved_tensors)
        else:
            q, k, v, out, softmax_lse, cu_seqlens = ctx.saved_tensors
            half_index0 = ctx.half_index0
            half_index1 = ctx.half_index1
        dq, dk, dv = zigzag_ring_flash_attn_varlen_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens,
            ctx.max_seqlen,
            half_index0,
            half_index1,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def zigzag_ring_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
