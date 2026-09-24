# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from torch.distributed import init_device_mesh
from types import SimpleNamespace

from swift.metrics import compute_acc
from swift.sequence_parallel import sequence_parallel
from swift.trainers.mixin import SwiftMixin


class MetricValues(list):

    def update(self, values):
        self.extend(values)


def _check_accuracy(rank, rendezvous, ring_size, sequence_size):
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
        for lengths in ([3, 5], [4, 4], [5], [1, 3, 5]):
            positions = torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0)
            cu_seqlens = SwiftMixin.get_cu_seqlens(None, positions, None)
            labels = torch.full_like(positions, 5)
            labels[positions < 2] = -100
            sp.extra_kwargs['text_position_ids'] = positions
            padded_positions = sp.pad(positions, padding_value=-1, position_ids=positions)
            for wrong_token in [None, *torch.where(labels[0] != -100)[0].tolist()]:
                preds = labels.roll(-1, dims=1).clamp_min(0)
                if wrong_token is not None:
                    preds[0, wrong_token - 1] = 6
                padded_preds = sp.pad(preds, padding_value=0, position_ids=positions)
                shifted_labels = sp.pad(labels, padding_value=-100, position_ids=positions).roll(-1, dims=1)
                local_preds = sp.split(padded_preds, dim=1, position_ids=padded_positions)
                local_labels = sp.split(shifted_labels, dim=1, position_ids=padded_positions)
                logits = torch.nn.functional.one_hot(local_preds, num_classes=8).float()
                for strategy in ('seq', 'token'):
                    expected = compute_acc(preds, labels, acc_strategy=strategy, cu_seqlens=cu_seqlens)
                    metric = MetricValues()
                    trainer = SimpleNamespace(
                        args=SimpleNamespace(acc_strategy=strategy),
                        task_type='causal_lm',
                        problem_type=None,
                        template=SimpleNamespace(sequence_parallel_size=world_size, is_encoder_decoder=False),
                        model=SimpleNamespace(training=True),
                        custom_metrics={'train': {
                            f'{strategy}_acc': metric
                        }})
                    if world_size == 1:
                        logits = torch.nn.functional.one_hot(preds, num_classes=8).float()
                        local_labels = labels
                    metric_boundaries = cu_seqlens if strategy == 'seq' else None
                    SwiftMixin._compute_acc(trainer, SimpleNamespace(logits=logits), local_labels, metric_boundaries)
                    assert metric == expected[f'{strategy}_acc'], (lengths, wrong_token, strategy, metric, expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='Gloo is not available')
@pytest.mark.parametrize(('ring_size', 'sequence_size'), [(2, 1), (2, 2), (1, 2), (1, 1)])
def test_sequence_parallel_accuracy_matches_unsharded(tmp_path, ring_size, sequence_size):
    mp.spawn(
        _check_accuracy,
        args=((tmp_path / 'rendezvous').as_uri(), ring_size, sequence_size),
        nprocs=ring_size * sequence_size)
