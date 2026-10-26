# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exercise the production TP loss primitives with two CPU/Gloo vocabulary shards."""
import importlib.util
import pytest
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
from test_abkd_loss import reference_abkd
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from swift.rlhf_trainers.gkd_loss import TeacherOutput, gkd_loss


def load_tp_primitives(rank):
    """Supply the test process group without loading Megatron's model/runtime setup."""
    core = ModuleType('megatron.core')
    core.mpu = SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: 2,
        get_tensor_model_parallel_rank=lambda: rank,
        get_tensor_model_parallel_group=lambda: dist.group.WORLD)
    core.tensor_parallel = None
    bridge = ModuleType('mcore_bridge')
    bridge.split_cp_inputs = None
    root = Path(__file__).resolve().parents[2] / 'swift' / 'megatron' / 'trainers'
    modules = []
    with patch.dict(sys.modules, {'megatron': ModuleType('megatron'), 'megatron.core': core, 'mcore_bridge': bridge}):
        for name in ('vocab_parallel_utils', 'gkd_utils'):
            spec = importlib.util.spec_from_file_location(name, root / f'{name}.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
    return modules


def check_sharded_abkd(rank, rendezvous, alpha, beta, topk):
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        vocab, gather = load_tp_primitives(rank)
        generator = torch.Generator().manual_seed(57)
        full_student = torch.randn(2, 6, 32, generator=generator)
        full_teacher = torch.randn(2, 6, 32, generator=generator)
        labels = torch.tensor([[-100, -100, 1, 2, 3, -100], [-100, 1, 2, 3, 4, -100]])
        lo, hi = rank * 16, (rank + 1) * 16
        student = full_student[..., lo:hi].clone().requires_grad_()
        teacher = TeacherOutput(full_logits=full_teacher[..., lo:hi])
        if topk is not None:
            teacher = teacher.to_topk(topk, gather.vocab_parallel_topk)
        total, count = gkd_loss(
            student,
            teacher,
            labels,
            0.5,
            0.8,
            gather_fn=gather.tp_gather_topk,
            log_softmax_fn=vocab.vocab_parallel_log_softmax,
            sum_fn=vocab.vocab_parallel_sum,
            chunk_size=3,
            loss_type='abkd',
            abkd_alpha=alpha,
            abkd_beta=beta)
        total.backward()

        reference_student = full_student.double().requires_grad_()
        reference_teacher = full_teacher[labels != -100]
        selected_student = reference_student[labels != -100]
        if topk is not None:
            reference_teacher, indices = reference_teacher.topk(topk, dim=-1)
            selected_student = selected_student.gather(-1, indices)
        expected = reference_abkd(selected_student, reference_teacher, alpha, beta, 0.8)
        expected.backward()
        assert count.item() == 7
        torch.testing.assert_close(total.double(), expected, rtol=3e-5, atol=2e-6)
        torch.testing.assert_close(student.grad.double(), reference_student.grad[..., lo:hi], rtol=1e-4, atol=3e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('alpha,beta', [(0.2, 0.7), (1.2, -0.2)])
@pytest.mark.parametrize('topk', [None, 5])
def test_two_vocab_shards_match_full_reference(tmp_path, alpha, beta, topk):
    mp.spawn(check_sharded_abkd, args=(str(tmp_path / 'gloo'), alpha, beta, topk), nprocs=2, join=True)
