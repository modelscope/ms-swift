# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
from collections import defaultdict
from types import SimpleNamespace

from swift.rl_core.advantage import compute_sdar_loss
from swift.rl_core.data import GRPOBatch
from swift.rlhf_trainers.grpo_trainer import GRPOTrainer


def _batch(teacher=True):
    return GRPOBatch(
        completion_mask=torch.ones(4, 3, dtype=torch.bool),
        truncated_mask=torch.zeros(4, dtype=torch.bool),
        seq_lengths=torch.full((4, ), 3),
        old_per_token_logps=torch.full((4, 3), -2.0, dtype=torch.float64),
        teacher_per_token_logps=(torch.arange(12, dtype=torch.float64).view(4, 3) / 20 - 1.0 if teacher else None),
        advantages=torch.zeros(4, 3, dtype=torch.float64),
        num_items_in_batch=torch.tensor(12),
        logits_to_keep=3,
    )


def _trainer(chunk_size=2, training=True):
    trainer = object.__new__(GRPOTrainer)
    trainer.model = torch.nn.Embedding(12, 8, dtype=torch.float64).train(training)
    with torch.no_grad():
        trainer.model.weight.copy_(torch.arange(96, dtype=torch.float64).view(12, 8) / 100)
    trainer.args = SimpleNamespace(
        per_device_train_batch_size=chunk_size,
        per_device_eval_batch_size=chunk_size,
        log_entropy=False,
        top_entropy_quantile=1.0,
        delta=None,
    )
    trainer.template = SimpleNamespace(padding_free=False)
    trainer.accelerator = SimpleNamespace(num_processes=1, gather_for_metrics=lambda value: value)
    trainer.dynamic_num_samples = True
    trainer.is_multimodal = False
    trainer.compute_entropy = False
    trainer.overlong_filter = False
    trainer.beta = 0.0
    trainer.kl_in_reward = False
    trainer.rollout_importance_sampling_mode = None
    trainer.log_rollout_offpolicy_metrics = False
    trainer.disable_rollout_importance_sampling = False
    trainer.importance_sampling_level = 'token'
    trainer._has_teacher = True
    trainer.loss_type = 'grpo'
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer.off_policy_sequence_mask_delta = None
    trainer.sdar_loss_coef = 0.5
    trainer.sdar_gate_beta = 5.0
    trainer.chord_sft_iterator = None
    trainer._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
    trainer._update_metrics = lambda metrics: None
    trainer._aggregate_and_update_metrics = lambda metrics, mode: None

    def logps(model, inputs, batch, compute_entropy=False):
        logits = model(inputs['input_ids'])
        return logits.log_softmax(-1)[..., 0], None

    trainer._get_per_token_logps_and_entropies = logps
    return trainer


@pytest.mark.parametrize('start,end', [(0, 1), (1, 3), (3, 4)])
@pytest.mark.parametrize('has_teacher', [True, False])
def test_chunk_preserves_teacher_logps(start, end, has_teacher):
    trainer = _trainer()
    batch = _batch(has_teacher)
    input_ids = torch.arange(12).view(4, 3)

    inputs, chunk = trainer.get_chunked_inputs({'input_ids': input_ids}, batch, start, end)

    torch.testing.assert_close(inputs['input_ids'], input_ids[start:end])
    if has_teacher:
        assert chunk.teacher_per_token_logps is not None
        torch.testing.assert_close(chunk.teacher_per_token_logps, batch.teacher_per_token_logps[start:end])
        assert chunk.teacher_per_token_logps.dtype == batch.teacher_per_token_logps.dtype
        assert chunk.teacher_per_token_logps.device == batch.teacher_per_token_logps.device
    else:
        assert chunk.teacher_per_token_logps is None


@pytest.mark.parametrize('remote_batch_size', [None, 16])
@pytest.mark.parametrize('chunk_size', [1, 2, 3])
@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('has_teacher', [True, False])
def test_dynamic_chunks_keep_sdar_loss_and_gradient(chunk_size, training, has_teacher, remote_batch_size, monkeypatch):
    trainer = _trainer(chunk_size, training)
    batch = _batch(has_teacher)
    if remote_batch_size is not None:
        # Simulate only the collective return, including ranks requiring extra dummy chunks.
        def gather_object(value):
            return value + [True if isinstance(value[0], bool) else remote_batch_size]

        monkeypatch.setitem(GRPOTrainer._compute_loss_chunked.__globals__, 'gather_object', gather_object)
    inputs = {'input_ids': torch.arange(12).view(4, 3)}
    full_logps, _ = trainer._get_per_token_logps_and_entropies(trainer.model, inputs, batch)
    # All completions have equal lengths, so the weighted chunk mean must equal the unchunked SDAR loss.
    expected = full_logps.sum() * 0.0
    if has_teacher:
        sdar_loss, _ = compute_sdar_loss(full_logps, batch.teacher_per_token_logps, batch.completion_mask,
                                         trainer.sdar_gate_beta)
        expected = trainer.sdar_loss_coef * sdar_loss
    expected_gradient, = torch.autograd.grad(expected, trainer.model.weight)

    actual = trainer._compute_loss(trainer.model, inputs, batch)
    actual_gradient, = torch.autograd.grad(actual, trainer.model.weight)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    if has_teacher:
        assert actual_gradient.abs().sum() > 0
        mode = 'train' if training else 'eval'
        assert len(trainer._metrics[mode]['sdar/loss']) > 0
