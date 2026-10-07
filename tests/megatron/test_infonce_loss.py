# Copyright (c) ModelScope Contributors. All rights reserved.
"""Compare distributed InfoNCE with the full-batch contrastive objective.

Run from the repository root with at least two GPUs::

    PYTHONPATH=. torchrun --standalone --nproc_per_node=4 -m pytest tests/megatron/test_infonce_loss.py -q
"""
import os
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from types import SimpleNamespace

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


@pytest.fixture(scope='module', params=[1, 2])
def parallel_groups(request, distributed):
    world_size = distributed
    tp_size = request.param
    if not torch.cuda.is_available() or world_size < 2 * tp_size or world_size % tp_size:
        pytest.skip('requires torchrun with at least two data-parallel ranks')
    from megatron.core import mpu

    mpu.initialize_model_parallel(tensor_model_parallel_size=tp_size)
    try:
        yield mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
    finally:
        dist.barrier()
        mpu.destroy_model_parallel()


@pytest.mark.parametrize('calculate_per_token_loss', [False, True])
@pytest.mark.parametrize('uneven_negatives', [False, True])
def test_infonce_data_parallel_loss_and_gradients(parallel_groups, monkeypatch, uneven_negatives,
                                                  calculate_per_token_loss):
    rank, world_size = parallel_groups
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
    loss_func = InfonceLoss(None, None)
    loss_func.is_megatron = True
    loss_func.args = SimpleNamespace(calculate_per_token_loss=calculate_per_token_loss)
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
    # calculate_per_token_loss=False: Megatron averages local grads by the DP world size, so the
    # loss premultiplies by it to recover the summed gradient. calculate_per_token_loss=True sums
    # grads without the 1/dp scaling, so no compensation is applied.
    grad_scale = 1 if calculate_per_token_loss else world_size
    torch.testing.assert_close(actual_grad, grad_scale * expected_grad)
