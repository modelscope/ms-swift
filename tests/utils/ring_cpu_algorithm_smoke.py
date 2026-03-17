import argparse
from itertools import accumulate

import torch
import torch.distributed as dist

from swift.sequence_parallel import zigzag_ring_attn as zra


def build_cu_seqlens(lengths, device):
    return torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device)


def split_packed(value, lengths, world_size, rank):
    local_values = []
    start = 0
    for length in lengths:
        sub_value = value[:, start:start + length]
        local_chunks = sub_value.chunk(2 * world_size, dim=1)
        local_values.extend([local_chunks[rank], local_chunks[2 * world_size - 1 - rank]])
        start += length
    return torch.cat(local_values, dim=1).contiguous()


def gather_packed(local_value, lengths, world_size):
    gathered = [torch.empty_like(local_value) for _ in range(world_size)]
    dist.all_gather(gathered, local_value.contiguous())
    full_value = torch.empty(
        (local_value.shape[0], sum(lengths), *local_value.shape[2:]),
        dtype=local_value.dtype,
        device=local_value.device,
    )
    accumulated_length = 0
    for length in lengths:
        local_length = length // world_size
        chunk_size = local_length // 2
        for idx_rp, rp_tensor in enumerate(gathered):
            local_tensor = rp_tensor[:, accumulated_length:accumulated_length + local_length]
            left_idx = accumulated_length * world_size + idx_rp * chunk_size
            full_value[:, left_idx:left_idx + chunk_size] = local_tensor[:, :chunk_size]
            right_idx = accumulated_length * world_size + (2 * world_size - idx_rp - 1) * chunk_size
            full_value[:, right_idx:right_idx + chunk_size] = local_tensor[:, chunk_size:]
        accumulated_length += local_length
    return full_value.contiguous()


def reference_varlen_attention(q, k, v, lengths, softmax_scale):
    groups = q.shape[2] // k.shape[2]
    output = torch.empty((q.shape[0], q.shape[1], q.shape[2], v.shape[-1]), dtype=torch.float32, device=q.device)
    start = 0
    for length in lengths:
        end = start + length
        q_seq = q[:, start:end].squeeze(0).to(torch.float32)
        k_seq = k[:, start:end].squeeze(0).to(torch.float32)
        v_seq = v[:, start:end].squeeze(0).to(torch.float32)
        if groups > 1:
            k_seq = k_seq.repeat_interleave(groups, dim=1)
            v_seq = v_seq.repeat_interleave(groups, dim=1)
        scores = torch.einsum('qhd,khd->hqk', q_seq, k_seq) * softmax_scale
        causal_mask = torch.triu(torch.ones((length, length), dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        output[:, start:end] = torch.einsum('hqk,khd->qhd', probs, v_seq).unsqueeze(0)
        start = end
    return output


def manual_block_forward(q, k, v, causal, cu_seqlens, _max_seqlen, block_seq_len, _dropout_p, softmax_scale,
                         _alibi_slopes, _window_size, return_ctx=False):
    seqlen_q = q.shape[0]
    seqlen_kv = k.shape[0]
    half_cu_seqlens = cu_seqlens // 2
    cu_seqlens_q = half_cu_seqlens if seqlen_q == block_seq_len else cu_seqlens
    cu_seqlens_kv = half_cu_seqlens if seqlen_kv == block_seq_len else cu_seqlens
    scale = softmax_scale or q.shape[-1]**(-0.5)
    groups = q.shape[1] // k.shape[1]

    outputs = []
    lses = []
    for i in range(len(cu_seqlens_q) - 1):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_start, k_end = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()
        q_seq = q[q_start:q_end].to(torch.float32)
        k_seq = k[k_start:k_end].to(torch.float32)
        v_seq = v[k_start:k_end].to(torch.float32)
        if groups > 1:
            k_seq = k_seq.repeat_interleave(groups, dim=1)
            v_seq = v_seq.repeat_interleave(groups, dim=1)
        scores = torch.einsum('qhd,khd->hqk', q_seq, k_seq) * scale
        if causal:
            offset = k_seq.shape[0] - q_seq.shape[0]
            q_idx = torch.arange(q_seq.shape[0], device=q.device).unsqueeze(1)
            k_idx = torch.arange(k_seq.shape[0], device=q.device).unsqueeze(0)
            mask = k_idx > (q_idx + offset)
            scores = scores.masked_fill(mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum('hqk,khd->qhd', probs, v_seq).to(q.dtype))
        lses.append(torch.logsumexp(scores, dim=-1))

    block_out = torch.cat(outputs, dim=0).contiguous()
    block_lse = torch.cat(lses, dim=1).contiguous()
    if return_ctx:
        return block_out, block_lse, None
    return block_out, block_lse


def manual_block_backward(dout, q, k, v, _out, _softmax_lse, causal, cu_seqlens, _max_seqlen, block_seq_len,
                          dq_buffer, dk_buffer, dv_buffer, _dropout_p, softmax_scale, _alibi_slopes, _deterministic,
                          window_size, backend_ctx=None, block_dlse=None):
    del backend_ctx
    seqlen_q = q.shape[0]
    seqlen_kv = k.shape[0]
    half_cu_seqlens = cu_seqlens // 2
    cu_seqlens_q = half_cu_seqlens if seqlen_q == block_seq_len else cu_seqlens
    cu_seqlens_kv = half_cu_seqlens if seqlen_kv == block_seq_len else cu_seqlens
    dq, dk, dv = zra._manual_varlen_attention_backward(
        dout,
        q,
        k,
        v,
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
        dlse_dq, dlse_dk = zra._manual_varlen_lse_backward(
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


def run_case(lengths, heads_q, heads_kv, dim):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device('cpu')
    total_length = sum(lengths)
    softmax_scale = dim**-0.5

    zra.forward = manual_block_forward
    zra.backward = manual_block_backward
    zra._is_npu_tensor = lambda tensor: True

    torch.manual_seed(20260317)
    ref_q = torch.randn((1, total_length, heads_q, dim), device=device, dtype=torch.float32).requires_grad_(True)
    ref_k = torch.randn((1, total_length, heads_kv, dim), device=device, dtype=torch.float32).requires_grad_(True)
    ref_v = torch.randn((1, total_length, heads_kv, dim), device=device, dtype=torch.float32).requires_grad_(True)
    ref_out = reference_varlen_attention(ref_q, ref_k, ref_v, lengths, softmax_scale)
    ref_dout = torch.randn_like(ref_out)
    ref_out.backward(ref_dout)

    dist_q = split_packed(ref_q.detach(), lengths, world_size, rank).clone().requires_grad_(True)
    dist_k = split_packed(ref_k.detach(), lengths, world_size, rank).clone().requires_grad_(True)
    dist_v = split_packed(ref_v.detach(), lengths, world_size, rank).clone().requires_grad_(True)
    dist_dout = split_packed(ref_dout.detach(), lengths, world_size, rank)
    cu_seqlens = build_cu_seqlens(lengths, device)
    max_seqlen = max(lengths)

    dist_out = zra.zigzag_ring_flash_attn_varlen_func(
        dist_q,
        dist_k,
        dist_v,
        cu_seqlens,
        max_seqlen,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=True,
        group=dist.group.WORLD,
    )
    dist_out.backward(dist_dout)

    gathered_out = gather_packed(dist_out.detach(), lengths, world_size)
    gathered_dq = gather_packed(dist_q.grad.detach(), lengths, world_size)
    gathered_dk = gather_packed(dist_k.grad.detach(), lengths, world_size)
    gathered_dv = gather_packed(dist_v.grad.detach(), lengths, world_size)

    if rank == 0:
        print(
            f'world_size={world_size} lengths={lengths} '
            f'out_diff={(gathered_out - ref_out.detach()).abs().max().item():.8f} '
            f'dq_diff={(gathered_dq - ref_q.grad.detach()).abs().max().item():.8f} '
            f'dk_diff={(gathered_dk - ref_k.grad.detach()).abs().max().item():.8f} '
            f'dv_diff={(gathered_dv - ref_v.grad.detach()).abs().max().item():.8f}'
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lengths', type=int, nargs='+', required=True)
    parser.add_argument('--heads-q', type=int, default=4)
    parser.add_argument('--heads-kv', type=int, default=2)
    parser.add_argument('--dim', type=int, default=16)
    args = parser.parse_args()

    dist.init_process_group(backend='gloo')
    try:
        run_case(args.lengths, args.heads_q, args.heads_kv, args.dim)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
