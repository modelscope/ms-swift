"""Four CPU/Gloo ranks compare token-weighted updates with one logical batch."""
import torch
import torch.distributed as dist
from fsdp_turbo.distributed.parallel_state import ParallelState, Singleton
from test_token_normalization import TinyCausalLM, accumulation_trainer, token_batch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from types import SimpleNamespace


def check_host_gradients(state, data_rank, cp_rank, cp_size, ep_shape):
    from fsdp_turbo.distributed.context_parallel.loss import _shift_and_shard_labels, cp_all_reduce_forward_backward
    from fsdp_turbo.models.deepseek.deepseek_v41.engram.engram_optimizer import register_engram_optimizers
    from fsdp_turbo.models.deepseek.deepseek_v41.engram.host_offload_engram import TorchHostOffloadEngram
    from fsdp_turbo.training.factories import OptimizerFactory

    mesh = init_device_mesh('cpu', ep_shape, mesh_dim_names=('replica', 'ep'))
    cp_group = state.get_group('cp') if cp_size > 1 else None

    class HostModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Parameter(torch.tensor(1.))
            self.table = TorchHostOffloadEngram(8, 4, layer_id=0, num_max_tokens_per_rank=32, pin_memory=False)
            self.table.shard_(mesh['ep'])
            self.table.wire_replica_mesh(mesh['replica'])
            torch.nn.init.ones_(self.table.weight)

        def forward(self, input_ids, labels, **kwargs):
            targets = _shift_and_shard_labels(labels, cp_group, -100)
            value = self.table(input_ids).float().sum(-1) + self.dense
            total = (value * targets.ne(-100)).sum()
            if cp_group is not None:
                total = cp_all_reduce_forward_backward(total, cp_group)
            return SimpleNamespace(loss=total / labels[:, 1:].ne(-100).sum().clamp_min(1))

    model = HostModel()
    trainer = accumulation_trainer(model, 2, state=state)
    trainer._replicated_parameters = [model.dense]
    register_engram_optimizers(OptimizerFactory)
    trainer.optimizer = OptimizerFactory.create(
        'EngramOptimizer', model, [model.table], dense_optimizer_type='SGD', lr=0.03)
    trainer.lr_scheduler = None
    counts = ((1, 3), (2, 0), (4, 1), (0, 2))
    losses = []
    for micro, count in enumerate(counts[data_rank]):
        batch = token_batch((count, ), micro + data_rank)
        batch['input_ids'] = batch['input_ids'].chunk(cp_size, dim=1)[cp_rank]
        losses.append(trainer.train_step(batch)[0].detach())
    trainer.sync_loss({'loss': losses})
    oracle = torch.zeros(model.table.num_embeddings, 4)
    total_tokens = 0
    for data in range(state.get_group_size('data')):
        for micro, count in enumerate(counts[data]):
            ids = token_batch((count, ), micro + data)['input_ids'][0, :count]
            oracle.index_add_(0, ids, torch.ones(count, 4))
            total_tokens += count
    oracle /= total_tokens
    expected = oracle[model.table.row_start:model.table.row_start + model.table.local_rows]

    def checked_clip():
        torch.testing.assert_close(model.dense.grad, torch.tensor(1.))
        model.table.reduce_sparse_gradient_across_replicas()
        pending = model.table.pending_sparse_grad()
        actual = torch.zeros_like(expected) if pending is None else pending.to_dense()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        return 0.

    trainer._clip_grad_norm = checked_clip
    trainer._on_optimizer_step()


def main():
    dist.init_process_group('gloo')
    try:
        assert dist.get_world_size() == 4
        for shape in ((4, 1, 1), (2, 2, 1), (1, 2, 2)):
            mesh = init_device_mesh('cpu', shape, mesh_dim_names=('data', 'cp', 'tp'))
            state = object.__new__(ParallelState)
            state.device_mesh_map = {'data': mesh['data'], 'cp': mesh['cp'], 'tp': mesh['tp']}
            Singleton._instances[ParallelState] = state
            data_rank, cp_rank, _ = mesh.get_coordinate()
            torch.manual_seed(42)
            model = TinyCausalLM(cp_group=mesh['cp'].get_group() if shape[1] > 1 else None, fp32_loss=True)
            reference = TinyCausalLM(fp32_loss=True)
            reference.load_state_dict(model.state_dict())
            trainer = accumulation_trainer(model, 2, state=state, optimizer='AdamW')
            trainer._replicated_parameters = list(model.parameters())
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.03)
            reference_scheduler = torch.optim.lr_scheduler.StepLR(reference_optimizer, step_size=3, gamma=0.9)
            counts = ((1, 3), (2, 0), (4, 1), (0, 2))
            for step in range(20):
                losses = []
                for micro, count in enumerate(counts[data_rank]):
                    batch = token_batch((count, ), step + micro + data_rank)
                    batch['input_ids'] = batch['input_ids'].chunk(shape[1], dim=1)[cp_rank]
                    losses.append(trainer.train_step(batch)[0].detach())
                result = trainer.sync_loss({'loss': losses})
                all_batches = [
                    token_batch((count, ), step + micro + data) for data in range(shape[0])
                    for micro, count in enumerate(counts[data])
                ]
                combined = {name: torch.cat([batch[name] for batch in all_batches]) for name in ('input_ids', 'labels')}
                expected_loss = reference(**combined).loss
                expected_loss.backward()
                torch.testing.assert_close(result['loss'], expected_loss, rtol=2e-6, atol=2e-7)

                def checked_clip():
                    for actual, expected in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-6, atol=2e-7)
                    return torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

                trainer._clip_grad_norm = checked_clip
                trainer._on_optimizer_step()
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
                reference_optimizer.step()
                reference_scheduler.step()
                reference_optimizer.zero_grad()
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
            if shape[2] == 1:
                for ep_shape in ((1, 4), (2, 2)):
                    check_host_gradients(state, data_rank, cp_rank, shape[1], ep_shape)
            dist.barrier()
            Singleton._instances.pop(ParallelState, None)

        mesh = init_device_mesh('cpu', (4, ), mesh_dim_names=('fsdp', ))
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(DTensor.from_local(torch.ones(1, dtype=torch.float64), mesh, [Shard(0)]))
        model.weight.grad = DTensor.from_local(
            torch.full((1, ), dist.get_rank() + 1., dtype=torch.float64), mesh, [Shard(0)])
        trainer = accumulation_trainer(model, 2)
        trainer.lr_scheduler = None
        trainer._gradient_scale = 0.25

        def check_shard():
            torch.testing.assert_close(model.weight.grad.to_local(),
                                       torch.tensor([(dist.get_rank() + 1) * 0.25], dtype=torch.float64))
            return 0.

        trainer._clip_grad_norm = check_shard
        trainer._on_optimizer_step()
        expected = 1 - 0.03 * (dist.get_rank() + 1) * 0.25
        torch.testing.assert_close(model.weight.to_local(), torch.tensor([expected], dtype=torch.float64))
        if dist.get_rank() == 0:
            print('PASS: DP/CP/TP token normalization,20 AdamW updates,clipping and DTensor shards', flush=True)
    finally:
        Singleton._instances.pop(ParallelState, None)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
