# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from torch.distributed import init_device_mesh
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func

from swift.sequence_parallel import sequence_parallel


class RouterOutputModel(torch.nn.Module):

    def forward(self, router_logits, attention_mask=None):
        return MoeModelOutputWithPast(router_logits=router_logits)


def _check_moe_aux_loss(rank, rendezvous, ring_size, sequence_size):
    world_size = ring_size * sequence_size
    dist.init_process_group(
        'gloo', init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=60))
    try:
        sp = sequence_parallel
        sp.world_size = world_size
        sp.rp_world_size = ring_size
        sp.sp_world_size = sequence_size
        sp.device_mesh = init_device_mesh(
            'cpu', (1, ring_size, sequence_size), mesh_dim_names=('data', 'ring', 'sequence'))
        model = RouterOutputModel()
        sp._prepare_moe_aux_loss(model)
        # Internal padding, tail-only padding, aligned sequences, and a one-token sample.
        for lengths in ([3, 5], [8, 3], [8, 8], [1, 3, 5], [5]):
            positions = torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0)
            sp.extra_kwargs['text_position_ids'] = positions
            padded_positions = sp.pad(positions, padding_value=-1, position_ids=positions)
            reference = torch.randn(2, sum(lengths), 4, generator=torch.Generator().manual_seed(42))
            reference.requires_grad_()
            padded = sp.pad(reference.detach(), padding_value=0, position_ids=positions)
            local = sp.split(padded, dim=1, position_ids=padded_positions).detach().requires_grad_()
            output = model(tuple(local.unbind(0)), attention_mask=None)
            actual = torch.stack(output.router_logits)
            torch.testing.assert_close(actual, reference)
            actual_loss = load_balancing_loss_func(output.router_logits, num_experts=4, top_k=2)
            expected_loss = load_balancing_loss_func(tuple(reference.unbind(0)), num_experts=4, top_k=2)
            torch.testing.assert_close(actual_loss, expected_loss)
            actual_loss.backward()
            expected_loss.backward()
            padded_grad = sp.pad(reference.grad, padding_value=0, position_ids=positions)
            expected_grad = sp.split(padded_grad, dim=1, position_ids=padded_positions) * world_size
            # GatherLoss compensates for the subsequent distributed gradient average.
            torch.testing.assert_close(local.grad, expected_grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='Gloo is not available')
@pytest.mark.parametrize(('ring_size', 'sequence_size'), [(2, 1), (2, 2), (1, 2), (1, 1)])
def test_moe_aux_loss_excludes_sequence_parallel_padding(tmp_path, ring_size, sequence_size):
    rendezvous = (tmp_path / 'rendezvous').as_uri()
    mp.spawn(_check_moe_aux_loss, args=(rendezvous, ring_size, sequence_size), nprocs=ring_size * sequence_size)
