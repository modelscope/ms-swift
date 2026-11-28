# Copyright (c) ModelScope Contributors. All rights reserved.
"""Four actual Gloo ranks: branch-local FP32 gradients and globally unused replicas."""
import torch
import torch.distributed as dist

from swift.fsdpturbo.trainer import FSDPTurboTrainer


def main():
    dist.init_process_group('gloo')
    try:
        assert dist.get_world_size() == 4
        rank = dist.get_rank()
        active = torch.nn.Parameter(torch.ones(2))
        unused = torch.nn.Parameter(torch.ones(2))
        frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        if rank % 2 == 0:
            active.grad = torch.full_like(active, rank + 1.)
        trainer = object.__new__(FSDPTurboTrainer)
        trainer._replicated_parameters = [active, unused, frozen]
        trainer._sync_replicated_gradients()
        # Contributions are [1,0,3,0] / 4, including the two locally unused branches.
        torch.testing.assert_close(active.grad, torch.ones_like(active), rtol=0, atol=0)
        assert unused.grad is None and frozen.grad is None
        optimizer = torch.optim.AdamW([active, unused], lr=0.1, weight_decay=0.2)
        optimizer.step()
        torch.testing.assert_close(unused, torch.ones_like(unused), rtol=0, atol=0)
        torch.testing.assert_close(active, torch.full_like(active, 0.88), rtol=1e-6, atol=1e-7)
        assert unused not in optimizer.state
        dist.barrier()
        if rank == 0:
            print('FP32_REPLICA_GLOO4_BRANCH_AND_UNUSED_GRADIENTS_PASS', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
