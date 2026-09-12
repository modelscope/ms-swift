import pytest
import torch
from types import SimpleNamespace

from swift.rlhf_trainers.grpo_trainer import GRPOTrainer

WINDOW_NORMALIZED_LOSS_TYPES = ['dapo', 'cispo', 'fipo']
MICRO_BATCH_LOSS_TYPES = ['grpo', 'sapo', 'bnpo', 'dr_grpo']


def _make_trainer(loss_type,
                  gradient_accumulation_steps,
                  training=True,
                  model_accepts_loss_kwargs=False,
                  compute_loss_func=None,
                  has_current_gradient_accumulation_steps=True):
    trainer = object.__new__(GRPOTrainer)
    trainer.loss_type = loss_type
    trainer.model = SimpleNamespace(training=training)
    trainer.model_accepts_loss_kwargs = model_accepts_loss_kwargs
    trainer.compute_loss_func = compute_loss_func
    trainer.args = SimpleNamespace(gradient_accumulation_steps=gradient_accumulation_steps)
    if has_current_gradient_accumulation_steps:
        trainer.current_gradient_accumulation_steps = gradient_accumulation_steps
    return trainer


@pytest.mark.parametrize('loss_type', WINDOW_NORMALIZED_LOSS_TYPES)
@pytest.mark.parametrize('gradient_accumulation_steps', [1, 2, 4, 8])
def test_window_normalized_loss_keeps_global_token_mean_gradient(loss_type, gradient_accumulation_steps):
    # The loss is already normalized by the completion tokens of the whole accumulation window, so after the
    # division performed by `Trainer.training_step` its gradient must stay equal to the gradient of the
    # uncompensated loss, i.e. to the global token mean.
    trainer = _make_trainer(loss_type, gradient_accumulation_steps)
    parameter = torch.tensor(2.0, requires_grad=True)
    loss = GRPOTrainer._undo_gradient_accumulation_scaling(trainer, parameter.square())
    (loss / gradient_accumulation_steps).backward()

    torch.testing.assert_close(loss, parameter.square() * gradient_accumulation_steps)
    torch.testing.assert_close(parameter.grad, torch.tensor(4.0))


@pytest.mark.parametrize('has_current_gradient_accumulation_steps', [True, False])
def test_window_normalized_loss_on_older_transformers(has_current_gradient_accumulation_steps):
    # `current_gradient_accumulation_steps` only exists in recent transformers versions.
    trainer = _make_trainer('dapo', 4, has_current_gradient_accumulation_steps=has_current_gradient_accumulation_steps)
    loss = torch.tensor(3.0)

    actual = GRPOTrainer._undo_gradient_accumulation_scaling(trainer, loss)

    torch.testing.assert_close(actual, loss * 4)


@pytest.mark.parametrize('loss_type', MICRO_BATCH_LOSS_TYPES)
def test_micro_batch_normalized_loss_is_not_scaled(loss_type):
    # These losses are averaged over the current micro-batch, so scaling them is the correct behaviour.
    trainer = _make_trainer(loss_type, 4)
    loss = torch.tensor(3.0)

    actual = GRPOTrainer._undo_gradient_accumulation_scaling(trainer, loss)

    torch.testing.assert_close(actual, loss)


@pytest.mark.parametrize(
    'training, model_accepts_loss_kwargs, compute_loss_func',
    [
        (False, False, None),  # evaluation
        (True, True, None),  # the model handles the gradient-accumulation scaling itself
        (True, False, object()),  # `compute_loss_func` handles the gradient-accumulation scaling itself
    ],
)
def test_no_scaling_without_trainer_scaling(training, model_accepts_loss_kwargs, compute_loss_func):
    trainer = _make_trainer(
        'dapo',
        4,
        training=training,
        model_accepts_loss_kwargs=model_accepts_loss_kwargs,
        compute_loss_func=compute_loss_func)
    loss = torch.tensor(3.0)

    actual = GRPOTrainer._undo_gradient_accumulation_scaling(trainer, loss)

    torch.testing.assert_close(actual, loss)
