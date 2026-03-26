# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from megatron.core import mpu
from typing import Optional


def reduce_max_stat_across_model_parallel_group(stat: float) -> float:
    """
    Ranks without an optimizer will have no grad_norm or num_zeros_in_grad stats.
    We need to ensure the logging and writer rank has those values.
    This function reduces a stat tensor across the model parallel group.

    We use an all_reduce max since the values have already been summed across optimizer ranks where possible
    """
    stat = torch.tensor([stat], dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(stat, op=torch.distributed.ReduceOp.MAX, group=mpu.get_model_parallel_group())
    return stat.item()


def logical_and_across_model_parallel_group(input: bool) -> bool:
    """
    This function gathers a bool value across the model parallel group
    """
    input = int(bool(input))
    input = torch.tensor([input], dtype=torch.int, device=torch.cuda.current_device())
    torch.distributed.all_reduce(input, op=torch.distributed.ReduceOp.MIN, group=mpu.get_model_parallel_group())
    return bool(input.item())

def _gather_cp_outputs_impl(inputs: torch.Tensor, cu_seqlens: Optional[torch.Tensor], dim: int):
    """Reverse of split_cp_inputs: all_gather across CP ranks and reconstruct the full sequence.

    Args:
        inputs: The local tensor on this CP rank (output of split_cp_inputs).
        cu_seqlens: Full-sequence cumulative lengths for packed mode, or None.
        dim: The dimension along which the split was performed.

    Returns:
        The reconstructed full tensor with the original sequence order.
    """
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size <= 1:
        return inputs

    if dim < 0:
        dim = (dim + inputs.ndim) % inputs.ndim

    cp_rank = mpu.get_context_parallel_rank()
    cp_group = mpu.get_context_parallel_group()

    # All-gather across CP ranks
    gathered = [torch.empty_like(inputs) for _ in range(cp_size)]
    torch.distributed.all_gather(gathered, inputs.contiguous(), group=cp_group)
    gathered[cp_rank] = inputs

    num_seqs = 1 if cu_seqlens is None else (cu_seqlens.shape[0] - 1)

    if cu_seqlens is None:
        local_seq_len = inputs.shape[dim]
        full_seq_len = local_seq_len * cp_size
        chunk_len = local_seq_len // 2  # = full_seq_len // (2 * cp_size)

        out_shape = list(inputs.shape)
        out_shape[dim] = full_seq_len
        output = inputs.new_zeros(out_shape)

        for j in range(cp_size):
            o = gathered[j]
            slices_first = [slice(None)] * inputs.ndim
            slices_first[dim] = slice(0, chunk_len)
            slices_second = [slice(None)] * inputs.ndim
            slices_second[dim] = slice(chunk_len, 2 * chunk_len)

            o0 = o[tuple(slices_first)]   # chunk index j
            o1 = o[tuple(slices_second)]  # chunk index (2*cp_size - j - 1)

            dst_first = [slice(None)] * inputs.ndim
            dst_first[dim] = slice(j * chunk_len, (j + 1) * chunk_len)
            output[tuple(dst_first)] = o0

            reverse_idx = 2 * cp_size - j - 1
            dst_second = [slice(None)] * inputs.ndim
            dst_second[dim] = slice(reverse_idx * chunk_len, (reverse_idx + 1) * chunk_len)
            output[tuple(dst_second)] = o1

        return output
    else:
        # Packed (padding-free) mode: reconstruct each sequence individually.
        # IMPORTANT:
        # `split_cp_inputs` uses `chunk_len = seq_len // (2 * cp_size)` (floor) per-sequence
        # and then keeps exactly 2 chunks per CP rank, so each local sequence length is
        # `2 * chunk_len`. Using a ratio-based scaling of `cu_seqlens` can introduce
        # rounding/aliasing and make local boundaries inconsistent with split's truncation,
        # which may produce empty slices during gather.
        total_full_len = cu_seqlens[num_seqs].item()
        local_total_len = inputs.shape[dim]

        # Reconstruct local cu_seqlens using the same per-sequence truncation rule as split.
        cu_seqlens_local = cu_seqlens.new_zeros(num_seqs + 1)
        for i in range(num_seqs):
            seq_len = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
            chunk_len = seq_len // (2 * cp_size)
            cu_seqlens_local[i + 1] = cu_seqlens_local[i] + 2 * chunk_len

        # Sanity-check: local_total_len should match what split would have produced.
        if cu_seqlens_local[num_seqs].item() != local_total_len:
            raise RuntimeError(
                f'CP packed gather boundary mismatch: '
                f'inputs.shape[dim]={local_total_len}, '
                f'computed cu_seqlens_local_total={cu_seqlens_local[num_seqs].item()}, '
                f'total_full_len={total_full_len}. '
                f'Check that packed_seq_params.cu_seqlens_q matches the tensor domain used by '
                f'split_cp_inputs/gather_cp_outputs.'
            )

        out_shape = list(inputs.shape)
        out_shape[dim] = total_full_len
        output = inputs.new_zeros(out_shape)

        for i in range(num_seqs):
            start_full = cu_seqlens[i].item()
            end_full = cu_seqlens[i + 1].item()
            seq_len = end_full - start_full
            chunk_len = seq_len // (2 * cp_size)

            start_cp = cu_seqlens_local[i].item()
            local_len = cu_seqlens_local[i + 1].item() - start_cp  # = 2 * chunk_len

            for j in range(cp_size):
                o = gathered[j]
                slices_first = [slice(None)] * inputs.ndim
                slices_first[dim] = slice(start_cp, start_cp + chunk_len)
                slices_second = [slice(None)] * inputs.ndim
                slices_second[dim] = slice(start_cp + chunk_len, start_cp + local_len)

                o0 = o[tuple(slices_first)]
                o1 = o[tuple(slices_second)]

                dst_first = [slice(None)] * inputs.ndim
                dst_first[dim] = slice(start_full + j * chunk_len, start_full + (j + 1) * chunk_len)
                output[tuple(dst_first)] = o0

                reverse_idx = 2 * cp_size - j - 1
                dst_second = [slice(None)] * inputs.ndim
                dst_second[dim] = slice(start_full + reverse_idx * chunk_len,
                                        start_full + (reverse_idx + 1) * chunk_len)
                output[tuple(dst_second)] = o1

        return output


def _split_cp_inputs_impl(inputs: torch.Tensor, cu_seqlens: Optional[torch.Tensor], dim: int):
    """Pure split logic without autograd (used inside autograd Functions)."""
    if dim < 0:
        dim = (dim + inputs.ndim) % inputs.ndim
    new_inputs = []
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    for i in range(1 if cu_seqlens is None else (cu_seqlens.shape[0] - 1)):
        if cu_seqlens is None:
            val = inputs
        else:
            slices = [slice(None)] * inputs.ndim
            slices[dim] = slice(cu_seqlens[i], cu_seqlens[i + 1])
            val = inputs[tuple(slices)]
        view_shape = (*inputs.shape[:dim], 2 * cp_size, val.shape[dim] // (2 * cp_size), *inputs.shape[dim + 1:])
        val = val.view(view_shape)
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device='cpu',
                             pin_memory=True).cuda(non_blocking=True)
        val = val.index_select(dim, index)
        view_shape = (*inputs.shape[:dim], -1, *inputs.shape[dim + 1:])
        new_inputs.append(val.view(view_shape))
    return torch.cat(new_inputs, dim=dim)


class GatherSequenceAcrossCPRegion(torch.autograd.Function):
    """Autograd Function for gather_cp_outputs.

    Forward:  all_gather across CP ranks and reconstruct full sequence.
    Backward: split — each rank keeps only its own slice of the gradient,
              then multiplies by cp_size to cancel the /cp_size in SplitSequenceAcrossCPRegion.backward.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, cu_seqlens: Optional[torch.Tensor], dim: int) -> torch.Tensor:
        ctx.cu_seqlens = cu_seqlens
        ctx.dim = dim
        return _gather_cp_outputs_impl(inputs, cu_seqlens, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cp_size = mpu.get_context_parallel_world_size()
        grad_input = _split_cp_inputs_impl(grad_output, ctx.cu_seqlens, ctx.dim)
        # Multiply by cp_size to cancel the /cp_size in SplitSequenceAcrossCPRegion.backward,
        # ensuring in_proj.weight.grad is not under-scaled.
        return grad_input * cp_size, None, None


class SplitSequenceAcrossCPRegion(torch.autograd.Function):
    """Autograd Function for split_cp_inputs.

    Forward:  split the full sequence and keep only this rank's chunks.
    Backward: all_gather gradients across CP ranks to reconstruct the full gradient,
              then divide by cp_size so that finish_grad_sync (which sums across CP ranks)
              does not double-count GDN internal parameter gradients (conv1d, A_log, dt_bias, etc.).
              The /cp_size here is cancelled by the *cp_size in GatherSequenceAcrossCPRegion.backward,
              so in_proj.weight.grad is unaffected.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, cu_seqlens: Optional[torch.Tensor], dim: int) -> torch.Tensor:
        ctx.cu_seqlens = cu_seqlens
        ctx.dim = dim
        return _split_cp_inputs_impl(inputs, cu_seqlens, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cp_size = mpu.get_context_parallel_world_size()
        # all_gather reconstructs the full gradient for recurrent ops (chunk_gated_delta_rule).
        # Divide by cp_size: finish_grad_sync sums across CP ranks, so without this division
        # GDN internal parameters (conv1d, A_log, dt_bias) would have gradients cp_size× too large.
        grad_input = _gather_cp_outputs_impl(grad_output, ctx.cu_seqlens, ctx.dim)
        return grad_input / cp_size, None, None


def gather_cp_outputs(inputs: torch.Tensor, cu_seqlens: Optional[torch.Tensor], dim: int):
    """all_gather across CP ranks and reconstruct the full sequence.

    Forward: all_gather + reorder to full sequence.
    Backward: split — each rank keeps its own gradient slice.
    """
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size <= 1:
        return inputs
    if dim < 0:
        dim = (dim + inputs.ndim) % inputs.ndim
    return GatherSequenceAcrossCPRegion.apply(inputs, cu_seqlens, dim)


def split_cp_inputs(inputs: torch.Tensor, cu_seqlens: Optional[torch.Tensor], dim: int):
    """Split full sequence to this CP rank's portion.

    Forward: keep only this rank's 2 chunks out of 2*cp_size.
    Backward: all_gather gradients to reconstruct the full gradient tensor,
              so that upstream recurrent ops receive complete gradients.
    """
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size <= 1:
        return inputs
    if dim < 0:
        dim = (dim + inputs.ndim) % inputs.ndim
    return SplitSequenceAcrossCPRegion.apply(inputs, cu_seqlens, dim)
