"""CPU/Gloo regression for Swift's inherited loss reducer; no model or accelerator is used."""
import torch
import torch.distributed as dist
from fsdp_turbo.distributed.parallel_state import ParallelState, Singleton
from torch.distributed.device_mesh import init_device_mesh
from types import SimpleNamespace

from swift.fsdpturbo.trainer import FSDPTurboTrainer


def main():
    dist.init_process_group(backend='gloo')
    try:
        if dist.get_world_size() != 4:
            raise ValueError('Run this regression with exactly four CPU processes.')
        for shape in ((2, 1, 2), (1, 2, 2), (2, 2, 1)):
            mesh = init_device_mesh('cpu', shape, mesh_dim_names=('data', 'cp', 'tp'))
            # Exercise the production ParallelState group accessors with real CPU
            # meshes; its accelerator-only bootstrap is outside this metric test.
            state = object.__new__(ParallelState)
            state.device_mesh_map = {'data': mesh['data'], 'cp': mesh['cp']}
            Singleton._instances[ParallelState] = state
            data_rank, cp_rank, tp_rank = mesh.get_coordinate()
            losses = {
                'loss': [
                    torch.tensor(4096, dtype=torch.bfloat16),
                    torch.tensor(10 * data_rank + cp_rank + 100 * tp_rank, dtype=torch.bfloat16),
                    torch.tensor(1, dtype=torch.bfloat16),
                ],
                'aux_loss': [torch.tensor(2 * data_rank + 3 * cp_rank + 100 * tp_rank, dtype=torch.bfloat16)],
                'empty': [],
            }
            trainer = object.__new__(FSDPTurboTrainer)
            trainer.model = SimpleNamespace(parallel_state=state)
            trainer.config = SimpleNamespace(run=SimpleNamespace(gradient_accumulation_steps=3))
            trainer._accumulated_tokens = torch.tensor(3)
            result = trainer.sync_loss(losses)
            expected = 4097 + 5 * (shape[0] - 1) + (shape[1] - 1) / 2 + 100 * tp_rank
            torch.testing.assert_close(result['loss'], torch.tensor(expected), rtol=0, atol=0)
            assert result['loss'].dtype == torch.float32
            expected_aux = (shape[0] - 1) + 1.5 * (shape[1] - 1) + 100 * tp_rank
            torch.testing.assert_close(result['aux_loss'], torch.tensor(expected_aux), rtol=0, atol=0)
            assert result['empty'] is None
            dist.barrier()
            Singleton._instances.pop(ParallelState, None)
        if dist.get_rank() == 0:
            print('PASS: FP32 loss sums, data/CP means, TP exclusion and empty metrics on CPU/Gloo', flush=True)
    finally:
        Singleton._instances.pop(ParallelState, None)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
