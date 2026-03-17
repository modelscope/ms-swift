import argparse
from itertools import accumulate

import torch
import torch.distributed as dist

from swift.sequence_parallel.zigzag_ring_attn import zigzag_ring_flash_attn_varlen_func


def build_cu_seqlens(lengths, device):
    return torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device)


def split_packed(value, lengths, world_size, rank):
    local_values = []
    start = 0
    for length in lengths:
        end = start + length
        sub_value = value[:, start:end]
        local_chunks = sub_value.chunk(2 * world_size, dim=1)
        local_values.extend([local_chunks[rank], local_chunks[2 * world_size - 1 - rank]])
        start = end
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
    return output.to(q.dtype)


def run_case(lengths, heads_q, heads_kv, dim, dtype, atol, rtol):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f'npu:{int(torch.npu.current_device())}')
    total_length = sum(lengths)
    softmax_scale = dim**-0.5

    torch.manual_seed(20260317)
    torch.npu.manual_seed_all(20260317)

    ref_q = torch.randn((1, total_length, heads_q, dim), device=device, dtype=dtype).requires_grad_(True)
    ref_k = torch.randn((1, total_length, heads_kv, dim), device=device, dtype=dtype).requires_grad_(True)
    ref_v = torch.randn((1, total_length, heads_kv, dim), device=device, dtype=dtype).requires_grad_(True)
    ref_out = reference_varlen_attention(ref_q, ref_k, ref_v, lengths, softmax_scale)
    ref_dout = torch.randn_like(ref_out)
    ref_out.backward(ref_dout)

    dist_q = split_packed(ref_q.detach(), lengths, world_size, rank).clone().requires_grad_(True)
    dist_k = split_packed(ref_k.detach(), lengths, world_size, rank).clone().requires_grad_(True)
    dist_v = split_packed(ref_v.detach(), lengths, world_size, rank).clone().requires_grad_(True)
    dist_dout = split_packed(ref_dout.detach(), lengths, world_size, rank)
    cu_seqlens = build_cu_seqlens(lengths, device)
    max_seqlen = max(lengths)

    dist_out = zigzag_ring_flash_attn_varlen_func(
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

    out_diff = (gathered_out - ref_out.detach()).abs().max().item()
    dq_diff = (gathered_dq - ref_q.grad.detach()).abs().max().item()
    dk_diff = (gathered_dk - ref_k.grad.detach()).abs().max().item()
    dv_diff = (gathered_dv - ref_v.grad.detach()).abs().max().item()

    local_ok = all([
        torch.allclose(gathered_out, ref_out.detach(), atol=atol, rtol=rtol),
        torch.allclose(gathered_dq, ref_q.grad.detach(), atol=atol, rtol=rtol),
        torch.allclose(gathered_dk, ref_k.grad.detach(), atol=atol, rtol=rtol),
        torch.allclose(gathered_dv, ref_v.grad.detach(), atol=atol, rtol=rtol),
    ])
    ok_tensor = torch.tensor([int(local_ok)], device=device, dtype=torch.int32)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)

    if rank == 0:
        print(
            f'world_size={world_size} lengths={lengths} '
            f'out_diff={out_diff:.8f} dq_diff={dq_diff:.8f} '
            f'dk_diff={dk_diff:.8f} dv_diff={dv_diff:.8f}'
        )
    if ok_tensor.item() != 1:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lengths', type=int, nargs='+', required=True)
    parser.add_argument('--heads-q', type=int, default=4)
    parser.add_argument('--heads-kv', type=int, default=2)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--dtype', choices=['fp16', 'bf16'], default='fp16')
    parser.add_argument('--atol', type=float, default=5e-3)
    parser.add_argument('--rtol', type=float, default=5e-3)
    args = parser.parse_args()

    dist.init_process_group(backend='hccl')
    local_rank = int(torch.distributed.get_rank() % torch.npu.device_count())
    torch.npu.set_device(local_rank)
    dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16

    try:
        run_case(args.lengths, args.heads_q, args.heads_kv, args.dim, dtype, args.atol, args.rtol)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
