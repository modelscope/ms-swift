# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Test split_cp_inputs and gather_cp_outputs roundtrip correctness.

Usage:
    torchrun --nproc_per_node=2 tests/megatron/test_cp_split_gather.py
    torchrun --nproc_per_node=4 tests/megatron/test_cp_split_gather.py
"""
import sys
import types
import importlib.util
import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)

# ── mock megatron.core.mpu ──
_cp_group = dist.new_group(list(range(world_size)))

megatron_mod = types.ModuleType('megatron')
megatron_mod.__spec__ = importlib.util.spec_from_loader('megatron', loader=None)
megatron_core_mod = types.ModuleType('megatron.core')
megatron_core_mod.__spec__ = importlib.util.spec_from_loader('megatron.core', loader=None)


class _FakeMPU:
    @staticmethod
    def get_context_parallel_world_size():
        return world_size

    @staticmethod
    def get_context_parallel_rank():
        return rank

    @staticmethod
    def get_context_parallel_group():
        return _cp_group


megatron_core_mod.mpu = _FakeMPU()
megatron_mod.core = megatron_core_mod
sys.modules['megatron'] = megatron_mod
sys.modules['megatron.core'] = megatron_core_mod

_spec = importlib.util.spec_from_file_location(
    'parallel_utils',
    'swift/megatron/utils/parallel_utils.py')
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
split_cp_inputs = _mod.split_cp_inputs
gather_cp_outputs = _mod.gather_cp_outputs


def _log(msg):
    if rank == 0:
        print(msg, flush=True)


def _broadcast(tensor):
    dist.broadcast(tensor, src=0)
    return tensor


def test_content_correctness():
    """Verify split chunk assignment and gather reconstruction with a known tensor."""
    if world_size != 2:
        _log('  SKIP  content_correctness (requires cp_size=2)')
        return True

    full = _broadcast(torch.arange(8, dtype=torch.float32, device='cuda'))
    local = split_cp_inputs(full, cu_seqlens=None, dim=0)

    # rank0 should get chunks 0,3 -> [0,1,6,7]; rank1 gets chunks 1,2 -> [2,3,4,5]
    expected = {0: [0., 1., 6., 7.], 1: [2., 3., 4., 5.]}
    ok = torch.allclose(local, torch.tensor(expected[rank], device='cuda'))
    _log(f'  {"PASS" if ok else "FAIL"}  split content rank={rank}: {local.tolist()}')

    reconstructed = gather_cp_outputs(local, cu_seqlens=None, dim=0)
    match = torch.allclose(reconstructed, full)
    _log(f'  {"PASS" if match else "FAIL"}  gather content: {reconstructed.tolist()}')
    return ok and match


def test_roundtrip_no_cu_seqlens():
    """gather(split(x)) == x, without cu_seqlens."""
    passed = True
    for dim in [0, 1, 2, -1]:
        seq_len = 2 * world_size * 8
        shapes = {0: (seq_len, 3), 1: (2, seq_len, 4),
                  2: (2, 3, seq_len), -1: (2, 3, seq_len)}
        shape = shapes[dim]
        full = _broadcast(torch.randn(shape, device='cuda'))

        local = split_cp_inputs(full, cu_seqlens=None, dim=dim)
        recon = gather_cp_outputs(local, cu_seqlens=None, dim=dim)

        if torch.allclose(recon, full):
            _log(f'  PASS  dim={dim} shape={list(shape)}')
        else:
            diff = (recon - full).abs().max().item()
            _log(f'  FAIL  dim={dim} shape={list(shape)} max_diff={diff}')
            passed = False
    return passed


def test_roundtrip_with_cu_seqlens():
    """gather(split(x)) == x, with cu_seqlens (packed mode)."""
    passed = True
    for dim in [0, 1]:
        seq_lens = [2 * world_size * 4, 2 * world_size * 6]
        total = sum(seq_lens)
        cu = torch.tensor([0] + [sum(seq_lens[:i + 1]) for i in range(len(seq_lens))],
                          dtype=torch.int32, device='cuda')
        shape = (total,) if dim == 0 else (1, total)
        full = _broadcast(torch.randn(shape, device='cuda'))

        local = split_cp_inputs(full, cu_seqlens=cu, dim=dim)
        recon = gather_cp_outputs(local, cu_seqlens=cu, dim=dim)

        if torch.allclose(recon, full):
            _log(f'  PASS  dim={dim} seqlens={seq_lens}')
        else:
            diff = (recon - full).abs().max().item()
            _log(f'  FAIL  dim={dim} seqlens={seq_lens} max_diff={diff}')
            passed = False
    return passed


def main():
    _log(f'cp_size={world_size}')
    ok = True

    _log('--- content correctness ---')
    ok &= test_content_correctness()

    _log('--- roundtrip without cu_seqlens ---')
    ok &= test_roundtrip_no_cu_seqlens()

    _log('--- roundtrip with cu_seqlens (packed) ---')
    ok &= test_roundtrip_with_cu_seqlens()

    _log('')
    _log('ALL PASSED' if ok else 'SOME TESTS FAILED')
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
