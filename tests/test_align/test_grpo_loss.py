import inspect
import pytest
import torch
from types import SimpleNamespace

from swift.rlhf_trainers.grpo_trainer import WINDOW_NORMALIZED_LOSS_TYPES, GRPOTrainer

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


def _window_normalized_loss(per_token_loss, completion_mask, num_items_in_batch, num_processes=1):
    # Mirrors GRPOTrainer._compute_loss_and_metrics for cispo/dapo/fipo.
    normalizer = num_items_in_batch / num_processes
    return (per_token_loss * completion_mask).sum() / normalizer


@pytest.mark.parametrize('loss_type', list(WINDOW_NORMALIZED_LOSS_TYPES))
@pytest.mark.parametrize('gradient_accumulation_steps', [1, 2, 4, 8])
def test_window_normalized_loss_keeps_global_token_mean_gradient(loss_type, gradient_accumulation_steps):
    # The policy loss is already a slice of the global token mean, so after the
    # division performed by Trainer.training_step its gradient must stay equal to
    # the uncompensated loss, i.e. independent of gradient_accumulation_steps.
    trainer = _make_trainer(loss_type, gradient_accumulation_steps)
    parameter = torch.tensor(2.0, requires_grad=True)
    loss = GRPOTrainer._undo_gradient_accumulation_scaling(trainer, parameter.square())
    (loss / gradient_accumulation_steps).backward()

    torch.testing.assert_close(loss, parameter.square() * gradient_accumulation_steps)
    torch.testing.assert_close(parameter.grad, torch.tensor(4.0))


@pytest.mark.parametrize('loss_type', list(WINDOW_NORMALIZED_LOSS_TYPES))
@pytest.mark.parametrize('gradient_accumulation_steps', [1, 2, 4, 8])
def test_dapo_formula_gas1_matches_gas_n(loss_type, gradient_accumulation_steps):
    # Split a window of 8 tokens across GAS micro-batches. Each micro-batch uses the
    # DAPO/CISPO/FIPO normalizer (total tokens in the window). After undoing Trainer's
    # extra /GAS, the accumulated gradient must match a single GAS=1 window.
    torch.manual_seed(0)
    total_tokens = 8
    assert total_tokens % gradient_accumulation_steps == 0
    per_mb = total_tokens // gradient_accumulation_steps
    token_scale = torch.arange(1, total_tokens + 1, dtype=torch.float32)

    weight = torch.tensor(1.5, requires_grad=True)
    trainer = _make_trainer(loss_type, gradient_accumulation_steps)
    for i in range(gradient_accumulation_steps):
        sl = slice(i * per_mb, (i + 1) * per_mb)
        per_token_loss = weight * token_scale[sl]
        completion_mask = torch.ones_like(per_token_loss)
        policy_loss = _window_normalized_loss(per_token_loss, completion_mask, total_tokens)
        scaled = GRPOTrainer._undo_gradient_accumulation_scaling(trainer, policy_loss)
        (scaled / gradient_accumulation_steps).backward(retain_graph=True)

    weight_ref = torch.tensor(1.5, requires_grad=True)
    per_token_loss_ref = weight_ref * token_scale
    ref_loss = _window_normalized_loss(per_token_loss_ref, torch.ones_like(per_token_loss_ref), total_tokens)
    ref_loss.backward()

    torch.testing.assert_close(weight.grad, weight_ref.grad)


@pytest.mark.parametrize('has_current_gradient_accumulation_steps', [True, False])
def test_window_normalized_loss_on_older_transformers(has_current_gradient_accumulation_steps):
    # current_gradient_accumulation_steps only exists in recent transformers versions.
    trainer = _make_trainer('dapo', 4, has_current_gradient_accumulation_steps=has_current_gradient_accumulation_steps)
    loss = torch.tensor(3.0)

    actual = GRPOTrainer._undo_gradient_accumulation_scaling(trainer, loss)

    torch.testing.assert_close(actual, loss * 4)


@pytest.mark.parametrize('loss_type', MICRO_BATCH_LOSS_TYPES)
def test_micro_batch_normalized_loss_is_not_scaled(loss_type):
    # These losses are averaged over the current micro-batch, so Trainer /GAS is correct.
    trainer = _make_trainer(loss_type, 4)
    loss = torch.tensor(3.0)

    actual = GRPOTrainer._undo_gradient_accumulation_scaling(trainer, loss)

    torch.testing.assert_close(actual, loss)


@pytest.mark.parametrize(
    'training, model_accepts_loss_kwargs, compute_loss_func',
    [
        (False, False, None),  # evaluation
        (True, True, None),  # the model handles the gradient-accumulation scaling itself
        (True, False, object()),  # compute_loss_func handles the gradient-accumulation scaling itself
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


def test_compute_loss_and_metrics_undoes_gas_for_window_normalized_types():
    source = inspect.getsource(GRPOTrainer._compute_loss_and_metrics)
    assert 'self._undo_gradient_accumulation_scaling(loss)' in source
    assert 'WINDOW_NORMALIZED_LOSS_TYPES' in source
