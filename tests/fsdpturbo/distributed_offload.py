# Copyright (c) ModelScope Contributors. All rights reserved.
"""Run with torchrun --nproc_per_node=8 to check offload collective routing."""
import sys

from swift.cli._fsdpturbo.sft import ensure_npu_model_patch_disabled


def main():
    ensure_npu_model_patch_disabled(sys.argv)
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from swift.fsdpturbo.arguments import FSDPTurboSftArguments

    args = object.__new__(FSDPTurboSftArguments)
    args.offload_params = True
    args.ddp_backend = None
    args.ddp_timeout = 120
    args._init_device()
    try:
        if dist.get_world_size() != 8:
            raise ValueError('This integration probe requires eight accelerator workers.')
        device = torch.accelerator.current_accelerator()
        mesh = init_device_mesh(device.type, (2, 2, 2), mesh_dim_names=('dp', 'fsdp', 'tp'))
        rank = dist.get_rank()
        for group in [None] + [mesh.get_group(dim) for dim in range(3)]:
            ranks = range(8) if group is None else dist.get_process_group_ranks(group)
            for tensor_device in ('cpu', device):
                value = torch.tensor(float(rank), device=tensor_device)
                dist.all_reduce(value, group=group)
                torch.testing.assert_close(value.cpu(), torch.tensor(float(sum(ranks))))
        print(f'Offload collective routing passed: rank={rank}, backend={dist.get_backend_config()}', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
