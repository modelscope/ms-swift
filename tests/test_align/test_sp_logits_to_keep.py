import torch
import torch.nn.functional as F
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from swift.sequence_parallel import GatherLoss, sequence_parallel
from swift.trainers.mixin import SwiftMixin
from swift.trainers.utils import per_token_loss_func_sp


@contextmanager
def _sequence_parallel_state(*, world_size, sp_world_size, rp_world_size, extra_kwargs=None):
    old_state = (sequence_parallel.world_size, sequence_parallel.sp_world_size, sequence_parallel.rp_world_size,
                 sequence_parallel.extra_kwargs)
    sequence_parallel.world_size = world_size
    sequence_parallel.sp_world_size = sp_world_size
    sequence_parallel.rp_world_size = rp_world_size
    sequence_parallel.extra_kwargs = {} if extra_kwargs is None else extra_kwargs
    try:
        yield
    finally:
        (sequence_parallel.world_size, sequence_parallel.sp_world_size, sequence_parallel.rp_world_size,
         sequence_parallel.extra_kwargs) = old_state


def _trainer(sequence_parallel_size=2):
    trainer = object.__new__(SwiftMixin)
    trainer.template = SimpleNamespace(sequence_parallel_size=sequence_parallel_size, is_encoder_decoder=False)
    return trainer


def test_prepare_logits_to_keep_preserves_sp_label_frame():
    labels = torch.tensor([[-100, 4, -100, 7, 8]])
    loss_scale = torch.arange(labels.numel(), dtype=torch.float).reshape_as(labels)
    inputs = {'labels': labels.clone(), 'loss_scale': loss_scale.clone()}

    with patch('swift.trainers.mixin.is_mp', return_value=False):
        _trainer().prepare_logits_to_keep(inputs)

    torch.testing.assert_close(inputs['labels'], labels)
    torch.testing.assert_close(inputs['loss_scale'], loss_scale)
    assert inputs['logits_to_keep'].dtype == torch.bool
    assert inputs['logits_to_keep'].tolist() == [False, True, False, True, True]


def test_prepare_logits_to_keep_uses_shared_suffix_for_batched_sp_inputs():
    labels = torch.tensor([[-100, -100, 4, 7], [-100, 2, -100, 8]])
    inputs = {'labels': labels.clone()}

    with patch('swift.trainers.mixin.is_mp', return_value=True):
        _trainer().prepare_logits_to_keep(inputs)

    # The earliest target is at index one, so every row can use the same
    # trailing three hidden states while labels retain the local frame.
    assert inputs['logits_to_keep'].tolist() == [False, True, True, True]
    torch.testing.assert_close(inputs['labels'], labels)


def test_non_sft_sequence_parallel_trainer_keeps_full_logit_path():
    trainer = _trainer()
    trainer.args = SimpleNamespace(use_logits_to_keep=True)
    assert trainer.get_use_logits_to_keep() is False


def test_sp_selective_loss_scatter_matches_full_frame():
    labels = torch.tensor([[-100, 2, -100, 4, 5]])
    keep = labels[0] != -100
    torch.manual_seed(17)
    selected_logits = torch.randn(1, int(keep.sum()), 9, requires_grad=True)
    reference_logits = selected_logits.detach().clone().requires_grad_(True)

    # GatherLoss is a distributed autograd function; replacing it with an
    # identity is sufficient to exercise the local compact-to-full mapping.
    with _sequence_parallel_state(world_size=1, sp_world_size=1, rp_world_size=1), \
            patch.object(GatherLoss, 'apply', side_effect=lambda loss, gathered_labels, *_:
                         (loss, gathered_labels)):
        actual = per_token_loss_func_sp(SimpleNamespace(logits=selected_logits), labels, logits_to_keep=keep)

    expected = torch.zeros_like(labels, dtype=selected_logits.dtype)
    expected[:, keep] = F.cross_entropy(
        reference_logits.reshape(-1, reference_logits.shape[-1]), labels[:, keep].reshape(-1),
        reduction='none').reshape(1, -1)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(selected_logits.grad, reference_logits.grad)


def test_sp_selective_loss_handles_an_all_ignored_shard():
    labels = torch.full((1, 4), -100)
    keep = torch.tensor([False, False, False, True])
    logits = torch.randn(1, 1, 9, requires_grad=True)
    with _sequence_parallel_state(world_size=1, sp_world_size=1, rp_world_size=1), \
            patch.object(GatherLoss, 'apply', side_effect=lambda loss, gathered_labels, *_:
                         (loss, gathered_labels)):
        loss = per_token_loss_func_sp(SimpleNamespace(logits=logits), labels, logits_to_keep=keep)
    assert loss.shape == labels.shape
    assert loss.numel() == labels.numel()
    assert loss.sum().item() == 0


def test_sp_packed_cu_seqlens_gathers_local_mask():
    trainer = _trainer()
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]])
    local_mask = torch.tensor([True, False, True])
    gathered_mask = torch.tensor([[True, False, True, False, True, True]])
    with _sequence_parallel_state(world_size=2, sp_world_size=2, rp_world_size=1), \
            patch.object(sequence_parallel, 'gather', return_value=gathered_mask):
        cu_seqlens = trainer.get_cu_seqlens(position_ids, local_mask)
    torch.testing.assert_close(cu_seqlens, torch.tensor([0, 2, 4], dtype=cu_seqlens.dtype))


def test_sp_accuracy_reconstructs_selected_predictions():
    trainer = _trainer()
    trainer.args = SimpleNamespace(acc_strategy='token')
    trainer.task_type = 'causal_lm'
    trainer.problem_type = None
    trainer.model = SimpleNamespace(training=False)
    metric_updates = []
    trainer.custom_metrics = {'train': {}, 'eval': {'token_acc': SimpleNamespace(update=metric_updates.append)}}
    labels = torch.tensor([[-100, 2, -100, 4]])
    keep = labels[0] != -100
    # Argmax values at the two selected positions are 2 and 4.
    logits = torch.zeros(1, int(keep.sum()), 6)
    logits[0, 0, 2] = 5
    logits[0, 1, 4] = 5
    with _sequence_parallel_state(world_size=1, sp_world_size=1, rp_world_size=1), \
            patch.object(sequence_parallel, 'gather', side_effect=lambda value, **_: value):
        with patch('swift.trainers.mixin.compute_acc', return_value={'token_acc': [True]}) as compute_acc:
            trainer._compute_acc(SimpleNamespace(logits=logits), labels, logits_to_keep=keep)
    preds, aligned_labels = compute_acc.call_args.args[:2]
    assert preds.shape == labels.shape
    assert aligned_labels.shape == labels.shape
    assert preds.tolist() == [[0, 2, 0, 4]]
    assert aligned_labels.tolist() == [[4, -100, 2, -100]]
