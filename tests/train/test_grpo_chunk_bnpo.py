# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
from collections import defaultdict
from types import SimpleNamespace

from swift.rl_core.data import GRPOBatch
from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
from swift.trainers import per_token_loss_func


class TinyModel(torch.nn.Module):

    def __init__(self, dtype):
        super().__init__()
        self.head = torch.nn.Embedding(16, 5, dtype=dtype)
        with torch.no_grad():
            self.head.weight.copy_(torch.arange(80, dtype=dtype).view(16, 5).sin() / 3)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.head(input_ids))


def _case(chunk_size=2, training=True, batch_size=4, dtype=torch.float64, mask_mode='unequal', loss_type='bnpo'):
    trainer = object.__new__(GRPOTrainer)
    trainer.model = TinyModel(dtype).train(training)
    trainer.args = SimpleNamespace(
        per_device_train_batch_size=chunk_size,
        per_device_eval_batch_size=chunk_size,
        log_entropy=False,
        top_entropy_quantile=1.,
        delta=None,
        gradient_accumulation_steps=1,
        chord_mu_warmup_steps=0,
        chord_mu_decay_steps=10,
        chord_mu_peak=0.25,
        chord_mu_valley=0.25,
        chord_enable_phi_function=False,
    )
    trainer.template = SimpleNamespace(padding_free=False, data_collator=lambda values: dict(values))
    trainer.accelerator = SimpleNamespace(num_processes=1, gather_for_metrics=lambda value: value, device='cpu')
    trainer.dynamic_num_samples = True
    trainer.is_multimodal = False
    trainer.compute_entropy = False
    trainer.overlong_filter = mask_mode == 'overlong'
    trainer.beta = 0.
    trainer.kl_in_reward = False
    trainer.rollout_importance_sampling_mode = None
    trainer.log_rollout_offpolicy_metrics = False
    trainer.disable_rollout_importance_sampling = False
    trainer.importance_sampling_level = 'token'
    trainer._has_teacher = False
    trainer.loss_type = loss_type
    trainer.max_completion_length = 3
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer.off_policy_sequence_mask_delta = 0.5 if mask_mode == 'offpolicy' else None
    trainer.sdar_loss_coef = 0.
    trainer.chord_sft_iterator = None
    trainer.model_accepts_loss_kwargs = False
    trainer.compute_loss_func = None
    trainer._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
    trainer._update_metrics = lambda metrics: None
    trainer._aggregate_and_update_metrics = lambda metrics, mode: None

    def logps(model, inputs, batch, compute_entropy=False):
        return model(**inputs).logits.log_softmax(-1)[..., 0], None

    trainer._get_per_token_logps_and_entropies = logps
    inputs = {'input_ids': torch.arange(batch_size * 3).view(batch_size, 3)}
    lengths = torch.tensor([1, 1] + [3] * (batch_size - 2))
    if mask_mode == 'equal':
        lengths.fill_(3)
    if mask_mode in ['empty', 'first-empty']:
        lengths[:batch_size if mask_mode == 'empty' else 2] = 0
    mask = torch.arange(3).unsqueeze(0) < lengths.unsqueeze(1)
    truncated = torch.zeros(batch_size, dtype=torch.bool)
    if mask_mode == 'overlong':
        truncated[0] = True
    advantages = torch.arange(1, batch_size + 1, dtype=dtype).unsqueeze(1).expand(-1, 3)
    old_logps = trainer.model(**inputs).logits.log_softmax(-1)[..., 0].detach()
    if mask_mode == 'offpolicy':
        old_logps[:2] += 2
        advantages = advantages.clone()
        advantages[:2] *= -1
    batch = GRPOBatch(
        completion_mask=mask,
        truncated_mask=truncated,
        seq_lengths=torch.full((batch_size, ), 3),
        old_per_token_logps=old_logps,
        advantages=advantages,
        num_items_in_batch=mask.sum(),
        logits_to_keep=3,
    )
    return trainer, inputs, batch


def _compare_full_batch(trainer, inputs, batch):
    expected, _ = trainer._compute_loss_and_metrics(trainer.model, inputs, batch)
    expected_gradient, = torch.autograd.grad(expected, trainer.model.head.weight)
    actual = trainer._compute_loss(trainer.model, inputs, batch)
    actual_gradient, = torch.autograd.grad(actual, trainer.model.head.weight)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('batch_size', [4, 5])
@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('chunk_size', [1, 2, 3])
def test_bnpo_chunks_match_whole_batch_token_mean(chunk_size, training, batch_size, dtype):
    _compare_full_batch(*_case(chunk_size, training, batch_size, dtype))


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('mask_mode', ['overlong', 'offpolicy', 'first-empty'])
def test_bnpo_chunks_use_effective_mask_token_count(mask_mode, training):
    _compare_full_batch(*_case(training=training, mask_mode=mask_mode))


def test_bnpo_dummy_chunks_keep_full_batch_gradient(monkeypatch):
    trainer, inputs, batch = _case()
    # Only the collective return is simulated; dispatch, slicing, losses and autograd are real.
    monkeypatch.setitem(GRPOTrainer._compute_loss_chunked.__globals__, 'gather_object', lambda value: value + [16])
    _compare_full_batch(trainer, inputs, batch)


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('mask_mode', ['empty', 'equal'])
def test_bnpo_zero_tokens_and_equal_length_controls(mask_mode, training):
    _compare_full_batch(*_case(training=training, mask_mode=mask_mode))


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('loss_type', ['grpo', 'dr_grpo', 'dapo', 'cispo'])
def test_other_chunk_loss_normalizers_are_unchanged(loss_type, training):
    _compare_full_batch(*_case(training=training, loss_type=loss_type))


def test_bnpo_keeps_chord_auxiliary_sample_weighting():
    trainer, inputs, batch = _case()
    policy_loss, _ = trainer._compute_loss_and_metrics(trainer.model, inputs, batch)
    sft_batches = [
        {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'labels': torch.tensor([[-100, 2, 3]])
        },
        {
            'input_ids': torch.tensor([[6, 7, 8]]),
            'labels': torch.tensor([[-100, 0, 1]])
        },
    ]
    sft_losses = [
        per_token_loss_func(trainer.model(input_ids=values['input_ids']), values['labels']).sum() / 2
        for values in sft_batches
    ]
    expected = 0.75 * policy_loss + 0.25 * torch.stack(sft_losses).mean()
    expected_gradient, = torch.autograd.grad(expected, trainer.model.head.weight)
    trainer.state = SimpleNamespace(global_step=0)
    trainer.chord_sft_iterator = iter(sft_batches)
    actual = trainer._compute_loss(trainer.model, inputs, batch)
    actual_gradient, = torch.autograd.grad(actual, trainer.model.head.weight)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)
