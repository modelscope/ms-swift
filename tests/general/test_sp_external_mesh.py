# Copyright (c) ModelScope Contributors. All rights reserved.
"""M2 acceptance test: an external (duck-typed) mesh object must derive the same
SP mesh topology as the legacy self-computed mesh, rank by rank, group by group.

The external object only needs three attributes (no DeviceMesh class required):
``ulysses_size``, ``data_world_size`` and ``cp_world_size`` — here provided by a
plain ``SimpleNamespace``, the same way a twinkle ``DeviceMesh`` would duck-type in.

Run (from the repository root, 4 ranks):

    torchrun --nproc_per_node=4 tests/general/test_sp_external_mesh.py

Covers two configurations:
  case A (pure ulysses): world=4, num_heads=8, sp_size=2  -> legacy (dp=2, sp=2)
      external: SimpleNamespace(ulysses_size=2, data_world_size=2, cp_world_size=None)
  case B (ulysses + ring): world=4, num_heads=2, sp_size=4 -> legacy (dp=1, rp=2, sp=2)
      external: SimpleNamespace(ulysses_size=2, data_world_size=1, cp_world_size=2)
"""
import os
import torch
import torch.distributed as dist
from types import SimpleNamespace

from swift.sequence_parallel.sequence_parallel import SequenceParallel


def _build_mesh(sp_size, num_heads, device_mesh=None):
    sp = SequenceParallel()
    sp.num_heads = num_heads
    sp.world_size = sp_size
    sp._init_device_mesh(device_mesh)
    return sp


def _group_ranks(torch_mesh, dim_name):
    if dim_name not in (torch_mesh.mesh_dim_names or ()):
        return None
    return sorted(dist.get_process_group_ranks(torch_mesh[dim_name].get_group()))


def _check_case(tag, sp_size, num_heads, external_mesh):
    rank = dist.get_rank()
    old = _build_mesh(sp_size, num_heads)
    new = _build_mesh(None, num_heads, external_mesh)

    for attr in ('dp_world_size', 'sp_world_size', 'rp_world_size', 'world_size'):
        o, n = getattr(old, attr), getattr(new, attr)
        assert o == n, f'[{tag}][r{rank}] {attr}: legacy={o} external={n}'

    for dim in ('data', 'ring', 'sequence'):
        o, n = _group_ranks(old.device_mesh, dim), _group_ranks(new.device_mesh, dim)
        assert o == n, f'[{tag}][r{rank}] group {dim}: legacy={o} external={n}'
    dist.barrier()
    if rank == 0:
        print(f'[{tag}] topology identical: dp={new.dp_world_size} rp={new.rp_world_size} sp={new.sp_world_size}')


def main():
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    assert world == 4, 'this test expects exactly 4 ranks'
    backend = 'hccl' if torch.npu.is_available() else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    if backend == 'hccl':
        torch.npu.set_device(rank)

    _check_case(
        'A: pure ulysses',
        sp_size=2,
        num_heads=8,
        external_mesh=SimpleNamespace(ulysses_size=2, data_world_size=2, cp_world_size=None))
    _check_case(
        'B: ulysses + ring',
        sp_size=4,
        num_heads=2,
        external_mesh=SimpleNamespace(ulysses_size=2, data_world_size=1, cp_world_size=2))

    dist.barrier()
    if rank == 0:
        print('ALL_OK: external mesh object derives the same (data, ring, sequence) topology')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
