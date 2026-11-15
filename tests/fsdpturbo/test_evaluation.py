"""Validation weights tokens correctly and excludes duplicated sampler padding."""
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import patch

from swift.fsdpturbo.trainer import FSDPTurboTrainer


def make_trainer(*, rank=0, size=1, batch_size=2):

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.parallel_state = SimpleNamespace(
                get_data_group_size=lambda: size, get_data_rank=lambda: rank, get_data_group=lambda: None)

        def forward(self, input_ids, labels, **_kwargs):
            count = labels[:, 1:].ne(-100).sum(dim=1)
            loss = (input_ids[:, 0] * count * self.weight).sum() / count.sum()
            return SimpleNamespace(loss=loss)

    trainer = object.__new__(FSDPTurboTrainer)
    trainer.eval_dataset = []
    for value, count in [(1, 2), (2, 4), (4, 1)]:
        labels = torch.full((5, ), -100)
        labels[:count + 1] = 1
        trainer.eval_dataset.append({'input_ids': torch.full((5, ), value), 'labels': labels})
    trainer.model = Model()
    trainer.train_args = SimpleNamespace(device='cpu', per_device_eval_batch_size=batch_size)
    trainer.args = SimpleNamespace(router_aux_loss_coef=0)
    trainer.dataloader = SimpleNamespace(collate_fn=torch.utils.data.default_collate)
    trainer.prepare_batch = lambda batch: batch
    trainer._global_step = 2
    trainer._eval_history = []
    return trainer


def test_evaluation_is_token_weighted_and_restores_training_mode():
    trainer = make_trainer()
    with patch('torch.distributed.is_initialized', return_value=False):
        metrics = trainer.evaluate()
    assert metrics == {'step': 2, 'eval_loss': 2.0, 'eval_tokens': 7}
    assert trainer.model.training
    assert trainer.model.weight.grad is None


@pytest.mark.parametrize('rank,local_loss,local_tokens', [(0, 6, 3), (1, 8, 4)])
def test_sampler_duplicates_contribute_no_validation_targets(rank, local_loss, local_tokens):
    trainer = make_trainer(rank=rank, size=2, batch_size=1)
    trainer.model.eval()

    def reduce(value, **_kwargs):
        if value.is_floating_point():
            assert value.item() == local_loss
            value.fill_(14)
        else:
            assert value.item() == local_tokens
            value.fill_(7)

    with patch('torch.distributed.is_initialized', return_value=False), \
            patch('torch.distributed.all_reduce', side_effect=reduce):
        metrics = trainer.evaluate()
    assert metrics == {'step': 2, 'eval_loss': 2.0, 'eval_tokens': 7}
    assert not trainer.model.training
