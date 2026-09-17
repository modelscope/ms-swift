# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end check for InfoNCE under a real Megatron DistributedDataParallel wrapper.

Wraps a tiny linear model in Megatron's DDP, runs InfoNCE forward/backward and the DDP
gradient reduction, then compares the reduced ``main_grad`` against a single-process
full-batch backward on identical weights. This exercises the full loop:

    calculate_per_token_loss=False: loss premultiplies by dp_size, DDP divides by dp_size
    calculate_per_token_loss=True:  loss keeps scale 1,        DDP sums (no 1/dp)

Both regimes must recover the summed (correct) gradient.

Run from the repository root with at least two GPUs::

    PYTHONPATH=. torchrun --standalone --nproc_per_node=4 -m pytest tests/megatron/test_infonce_ddp_e2e.py -q
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


@pytest.mark.parametrize('calculate_per_token_loss', [False, True])
def test_infonce_megatron_ddp_matches_single_process(distributed, monkeypatch, calculate_per_token_loss):
    from megatron.core import mpu
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.transformer import TransformerConfig

    world_size = distributed
    for name, value in {
            'INFONCE_TEMPERATURE': '0.5',
            'INFONCE_USE_BATCH': 'True',
            'INFONCE_MASK_FAKE_NEGATIVE': 'False',
            'INFONCE_INCLUDE_QQ': 'False',
            'INFONCE_INCLUDE_DD': 'False',
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv('INFONCE_HARD_NEGATIVES', raising=False)

    mpu.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        rank = mpu.get_data_parallel_rank()
        dim = 5
        # Deterministic per-rank inputs, identical view on every rank (1 positive + 1 negative).
        groups = []
        for dp_rank in range(world_size):
            generator = torch.Generator().manual_seed(1000 + dp_rank)
            groups.append(torch.randn(3, dim, generator=generator).cuda())

        # Identical model weights across ranks (fixed seed, no random init broadcast needed).
        torch.manual_seed(0)
        model = torch.nn.Linear(dim, dim, bias=False).cuda()

        config = TransformerConfig(
            num_attention_heads=1, num_layers=1, hidden_size=dim, calculate_per_token_loss=calculate_per_token_loss)
        ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=False, use_distributed_optimizer=False)
        ddp_model = DistributedDataParallel(config, ddp_config=ddp_config, module=model)
        ddp_model.zero_grad_buffer()

        local_x = groups[rank]
        emb = ddp_model(local_x)
        labels = torch.tensor([1] + [0] * (len(local_x) - 2), device=emb.device)
        loss_func = InfonceLoss(None, None)
        loss_func.is_megatron = True
        loss_func.args = SimpleNamespace(calculate_per_token_loss=calculate_per_token_loss)
        loss = loss_func({'last_hidden_state': emb}, labels)
        loss.backward()
        ddp_model.finish_grad_sync()
        reduced_grad = model.weight.main_grad.clone()

        # Single-process reference: full-batch InfoNCE over every rank's embeddings, same weights.
        w_ref = model.weight.detach().clone().requires_grad_()
        embs = [g @ w_ref.T for g in groups]
        queries = torch.stack([e[0] for e in embs])
        documents = torch.cat([e[1:] for e in embs])
        targets = torch.tensor([sum(len(e) - 1 for e in embs[:i]) for i in range(world_size)], device=w_ref.device)
        ref_loss = F.cross_entropy(queries @ documents.T / 0.5, targets)
        ref_grad, = torch.autograd.grad(ref_loss, w_ref)

        torch.testing.assert_close(reduced_grad, ref_grad, rtol=1e-4, atol=1e-4)
    finally:
        dist.barrier()
        mpu.destroy_model_parallel()


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
