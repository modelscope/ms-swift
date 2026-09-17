# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real-NCCL check that cached host boundaries reconstruct exactly like the CUDA path.

This exercises ``swift.megatron.trainers.utils.reconstruct_tensor_cp``, the
reconstruction used by ``grpo_trainer.py`` and ``rlhf_mixin.py``. Nothing is
patched: the batch is sharded with the production ``mcore_bridge.split_cp_inputs``
entry point and gathered over a real context-parallel group.

The companion unit tests in ``tests/megatron/test_cp_packed_metadata.py`` cover the
same differential with a simulated collective.

Run from the repository root with at least two GPUs::

    PYTHONPATH=. torchrun --standalone --nproc_per_node=2 -m pytest \
        tests/megatron/test_cp_packed_metadata_cuda_e2e.py -q
    PYTHONPATH=. torchrun --standalone --nproc_per_node=4 -m pytest \
        tests/megatron/test_cp_packed_metadata_cuda_e2e.py -q

Without torchrun the cases skip, so a plain ``pytest`` run stays green.

Only ``zigzag`` is exercised: it is the supported production partition mode.
"""
import os
import pytest
import torch

# The Megatron stack is imported lazily below. A runner without it would otherwise
# fail to import this module, which `unittest` discovery reports as an error rather
# than a skip.


@pytest.fixture(scope='module')
def cp_group():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if not torch.cuda.is_available() or 'LOCAL_RANK' not in os.environ or world_size < 2:
        pytest.skip('requires torchrun with at least two GPUs')
    if world_size > torch.cuda.device_count():
        pytest.skip(f'requires {world_size} visible GPUs, found {torch.cuda.device_count()}')
    pytest.importorskip('mcore_bridge', reason='Megatron stack is required')
    pytest.importorskip('megatron.core', reason='Megatron stack is required')
    from swift.utils import init_process_group

    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    init_process_group(backend='nccl', timeout=120)
    import torch.distributed as dist
    from megatron.core import mpu

    mpu.initialize_model_parallel(context_parallel_size=world_size)
    try:
        yield mpu.get_context_parallel_world_size()
    finally:
        dist.barrier()
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def _packed_batch(cp_size, device, dtype):
    """Shard a packed batch exactly as training does, returning local and reference tensors."""
    from mcore_bridge import split_cp_inputs

    # Every sample must be divisible by 2 * cp_size for the zigzag layout.
    seq_lens = [4 * cp_size, 12 * cp_size]
    boundaries = [0]
    for seq_len in seq_lens:
        boundaries.append(boundaries[-1] + seq_len)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=device)

    if dtype == torch.long:
        # Token ids double as position markers, so an ordering bug is visible directly.
        reference = torch.arange(boundaries[-1], device=device, dtype=dtype).view(1, -1)
    else:
        # Seed identically on every rank: an unseeded generator would diverge and the
        # all-gather would then compare mismatched data.
        torch.manual_seed(1234)
        reference = torch.randn(1, boundaries[-1], device=device, dtype=dtype)

    local = split_cp_inputs(reference, cu_seqlens, -1)
    assert local.shape[-1] == boundaries[-1] // cp_size
    return reference, boundaries, cu_seqlens, local


def _build_params(cu_seqlens, boundaries, with_host_boundaries):
    from megatron.core.packed_seq_params import PackedSeqParams

    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max(boundaries),
        max_seqlen_kv=max(boundaries),
        qkv_format='thd',
    )
    if with_host_boundaries:
        packed_seq_params.swift_cu_seqlens = tuple(boundaries)
    return packed_seq_params


@pytest.mark.parametrize('dtype', (torch.long, torch.bfloat16))
def test_cuda_host_boundaries_match_device_path_e2e(cp_group, dtype):
    """The cached-boundary path must reconstruct bit-identically to the CUDA path."""
    from swift.megatron.trainers.utils import reconstruct_tensor_cp

    cp_size = cp_group
    device = torch.cuda.current_device()
    reference, boundaries, cu_seqlens, local = _packed_batch(cp_size, device, dtype)
    num_samples = len(boundaries) - 1

    host = reconstruct_tensor_cp(cp_size, local, _build_params(cu_seqlens, boundaries, True), num_samples, 'zigzag')
    device_path = reconstruct_tensor_cp(cp_size, local, _build_params(cu_seqlens, boundaries, False), num_samples,
                                        'zigzag')

    assert torch.equal(host, reference), 'cached-boundary reconstruction diverged from the original order'
    assert torch.equal(host, device_path), 'cached-boundary and CUDA-boundary paths disagree'


def test_cuda_host_boundary_path_preserves_gradient_e2e(cp_group):
    """Reconstruction keeps the local autograd graph, so the fix cannot silently detach."""
    from swift.megatron.trainers.utils import reconstruct_tensor_cp

    cp_size = cp_group
    device = torch.cuda.current_device()
    reference, boundaries, cu_seqlens, local = _packed_batch(cp_size, device, torch.bfloat16)
    local = local.clone().requires_grad_(True)

    packed_seq_params = _build_params(cu_seqlens, boundaries, True)
    reconstructed = reconstruct_tensor_cp(cp_size, local, packed_seq_params, len(boundaries) - 1, 'zigzag')
    assert torch.equal(reconstructed.detach(), reference)

    reconstructed.float().pow(2).sum().backward()
    assert local.grad is not None
    assert torch.isfinite(local.grad.float()).all()
