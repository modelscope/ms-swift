import os
import torch
import torch.distributed as dist

from swift.sequence_parallel.zigzag_ring_attn import zigzag_ring_flash_attn_varlen_func
from swift.utils import init_process_group


def _run_e2e():
    import torch_npu  # noqa: F401

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local_rank)
    init_process_group(backend='hccl', timeout=120)
    try:
        world_size = dist.get_world_size()
        if world_size != 2:
            raise AssertionError(f'This smoke test expects 2 ranks, got {world_size}')

        # Each global sequence is divisible by 2 * world_size, which is
        # required by the existing zigzag layout. More than four packed
        # sequences are used so this E2E exercises the vectorized path.
        lengths = [4, 8, 12, 16, 8, 12, 4, 8]
        cu_seqlens = torch.tensor([0, 4, 12, 24, 40, 48, 60, 64, 72], dtype=torch.int32, device='npu')
        local_tokens = sum(lengths) // world_size
        num_heads = 2
        head_dim = 8

        torch.manual_seed(1234 + local_rank)
        q = torch.randn((local_tokens, num_heads, head_dim), dtype=torch.bfloat16, device='npu', requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)

        output = zigzag_ring_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            max(lengths),
            causal=True,
            group=dist.group.WORLD,
        )
        if output.shape != (1, local_tokens, num_heads, head_dim):
            raise AssertionError(f'Unexpected output shape: {tuple(output.shape)}')
        if not torch.isfinite(output).all().item():
            raise AssertionError('Ring attention output contains non-finite values')

        loss = output.float().square().mean()
        loss.backward()
        for name, gradient in (('dq', q.grad), ('dk', k.grad), ('dv', v.grad)):
            if gradient is None:
                raise AssertionError(f'{name} was not produced')
            if gradient.shape != q.shape:
                raise AssertionError(f'Unexpected {name} shape: {tuple(gradient.shape)}')
            if not torch.isfinite(gradient).all().item():
                raise AssertionError(f'{name} contains non-finite values')

        print(f'rank={local_rank} output={tuple(output.shape)} loss={loss.item():.6f}')
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_npu_ring_attention_e2e():
    import pytest

    if os.environ.get('SWIFT_RUN_NPU_E2E') != '1':
        pytest.skip('Set SWIFT_RUN_NPU_E2E=1 to run the distributed NPU E2E test')
    if 'LOCAL_RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        pytest.skip('Run this test under torchrun with at least 2 ranks')
    try:
        import torch_npu  # noqa: F401
    except Exception:
        pytest.skip('torch_npu is not available')
    if not hasattr(torch, 'npu') or not torch.npu.is_available():
        pytest.skip('Ascend NPU is not available')
    _run_e2e()


if __name__ == '__main__':
    if os.environ.get('SWIFT_RUN_NPU_E2E') != '1':
        raise SystemExit('Set SWIFT_RUN_NPU_E2E=1 to run the distributed NPU E2E test')
    _run_e2e()
