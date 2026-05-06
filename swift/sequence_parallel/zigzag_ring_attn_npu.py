# Some code borrowed from the awesome work: https://github.com/zhuzilin/ring-flash-attention
# Copyright (c) ModelScope Contributors. All rights reserved.
from functools import cache

import torch
import torch.distributed as dist
import torch.nn.functional as F

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
    try:
        return torch_npu.npu_fusion_attention(**params)
    except TypeError as exc:
        # Older torch_npu builds do not expose softmax_layout. The returned lse
        # is normalized below, so falling back preserves the same external shape.
        if 'softmax_layout' not in str(exc):
            raise
        params.pop('softmax_layout', None)
        return torch_npu.npu_fusion_attention(**params)


def _get_npu_manual_attention_mask(
    causal: bool,
    window_size,
    q_len: int,
    k_len: int,
    device: torch.device,
) -> torch.Tensor | None:
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


def _manual_varlen_attention_forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    softmax_scale,
    causal,
    window_size,
):
    """Reference TND attention used only to replay backward exactly on NPU."""
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


def _squeeze_batch(*tensors):
    squeezed = []
    for tensor in tensors:
        if tensor.shape[0] == 1:
            squeezed.append(tensor.squeeze(0))
        else:
            squeezed.append(tensor)
    return tuple(squeezed)


def _update_out_and_lse(out, lse, block_out, block_lse):
    # Match the CUDA path's online softmax merge so exact backward can
    # differentiate through the replayed ring computation.
    if out is None:
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    else:
        block_out = block_out.to(torch.float32)
        block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

        diff = block_lse - lse
        sig_diff = torch.sigmoid(diff)

        out = out - sig_diff * (out - block_out)
        lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse


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
    """Correctness-first NPU backward.

    Ascend provides a fused forward kernel for TND varlen attention. For the
    minimal ring-attention path, backward replays the same zigzag ring merge with
    differentiable PyTorch ops and gathers per-step K/V grads back to owner ranks.
    """
    del max_seqlen

    kv_comm = RingComm(process_group)
    dout, q, k, v = _squeeze_batch(dout, q, k, v)
    cu_seqlens = cu_seqlens // kv_comm.world_size
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
                merged_out, merged_lse = _update_out_and_lse(merged_out, merged_lse, block_out, block_lse)
            else:
                merged_out[half_index1], merged_lse[half_index1] = _update_out_and_lse(
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
