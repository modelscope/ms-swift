# Copyright (c) ModelScope Contributors. All rights reserved.
"""Two-rank rendezvous smoke for S0-09: the c10d port-0 contract, proved by a real torchrun run.

Not a test -- addressed by path from test_s0_infra.py, never imported. Both ranks join one gloo
group through ``--rdzv-backend=c10d --rdzv-endpoint=localhost:0`` and all-reduce their rank+1; each
rank must see 3, which only happens when BOTH ranks made it through rendezvous. With the broken
``--master_port=0`` form rank 1 dials port 0 and the run hangs until timeout instead.
"""
import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(backend='gloo')
    try:
        rank = dist.get_rank()
        value = torch.tensor([rank + 1], dtype=torch.float64)
        dist.all_reduce(value)
        assert int(value.item()) == 3, f'rank {rank} saw {value.item()}, expected the 1+2 all-reduce'
        print(f'rank{rank} rendezvous-ok', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
