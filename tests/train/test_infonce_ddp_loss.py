# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regression test: InfoNCE under torch DDP premultiplies gradients by the world size.

torch DDP averages each rank's local gradient by the data-parallel world size, but the
gathered InfoNCE loss only carries a local gradient. The loss compensates by premultiplying
the local gradient with the world size so that DDP's 1/world_size averaging recovers the
summed (correct) gradient. This test checks that compensation without a real DDP wrapper by
comparing the local autograd gradient against the full-batch contrastive objective.

Run from the repository root with at least two GPUs::

    PYTHONPATH=. torchrun --standalone --nproc_per_node=4 -m pytest tests/train/test_infonce_ddp_loss.py -q
"""
import os
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from swift.loss.embedding import InfonceLoss


@pytest.fixture(scope='module')
def distributed():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if not torch.cuda.is_available() or world_size < 2:
        pytest.skip('requires torchrun with at least two GPUs')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    yield world_size
    dist.destroy_process_group()


@pytest.mark.parametrize('uneven_negatives', [False, True])
def test_infonce_torch_ddp_gradients(distributed, monkeypatch, uneven_negatives):
    world_size = distributed
    rank = int(os.environ['RANK'])
    for name, value in {
            'INFONCE_TEMPERATURE': '0.5',
            'INFONCE_USE_BATCH': 'True',
            'INFONCE_MASK_FAKE_NEGATIVE': 'False',
            'INFONCE_INCLUDE_QQ': 'False',
            'INFONCE_INCLUDE_DD': 'False',
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv('INFONCE_HARD_NEGATIVES', raising=False)

    groups = []
    for dp_rank in range(world_size):
        negatives = dp_rank + 1 if uneven_negatives else 1
        generator = torch.Generator().manual_seed(42 + dp_rank)
        groups.append(F.normalize(torch.randn(negatives + 2, 5, generator=generator), dim=-1).cuda())
    local = groups[rank].detach().clone().requires_grad_()
    labels = torch.tensor([1] + [0] * (len(local) - 2), device=local.device)
    # trainer=None keeps is_megatron False, exercising the torch DDP (gather_object) path.
    loss_func = InfonceLoss(None, None)
    actual = loss_func({'last_hidden_state': local}, labels)

    # All queries classify their positive among every rank's documents.
    reference = [group.detach().clone().requires_grad_() for group in groups]
    queries = torch.stack([group[0] for group in reference])
    documents = torch.cat([group[1:] for group in reference])
    targets = torch.tensor([sum(len(group) - 1 for group in reference[:i]) for i in range(world_size)],
                           device=local.device)
    expected = F.cross_entropy(queries @ documents.T / 0.5, targets)

    torch.testing.assert_close(actual, expected)
    actual_grad, = torch.autograd.grad(actual, local)
    expected_grad, = torch.autograd.grad(expected, reference[rank])
    # torch DDP averages local grads by world_size; the loss premultiplies to recover the sum.
    torch.testing.assert_close(actual_grad, world_size * expected_grad)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
