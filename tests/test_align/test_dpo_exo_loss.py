# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import GPT2Config, GPT2LMHeadModel
from types import SimpleNamespace

from swift.rlhf_trainers.dpo_trainer import DPOTrainer


def _trainer(loss_types, loss_weights, label_smoothing=0.0):
    # Exercise the real forward and mixed-loss paths without downloading a model.
    trainer = object.__new__(DPOTrainer)
    trainer.accelerator = Accelerator(cpu=True)
    trainer.args = SimpleNamespace(use_logits_to_keep=False, ld_alpha=None, rpo_alpha=None)
    trainer.template = SimpleNamespace(sequence_parallel_size=1, padding_free=False)
    trainer.loss_type = loss_types
    trainer.loss_weights = loss_weights
    trainer.label_smoothing = label_smoothing
    trainer.beta = 0.3
    trainer.reference_free = False
    trainer.f_divergence_type = 'reverse_kl'
    trainer.aux_loss_enabled = False
    trainer.is_encoder_decoder = False
    trainer.label_pad_token_id = -100
    trainer.use_weighting = False
    trainer._peft_has_been_casted_to_bf16 = False
    config = GPT2Config(
        vocab_size=16,
        bos_token_id=1,
        eos_token_id=2,
        n_positions=16,
        n_embd=8,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        trainer.model = GPT2LMHeadModel(config).double()
        torch.manual_seed(23)
        trainer.ref_model = GPT2LMHeadModel(config).double().eval()
    return trainer


def _batch():
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 8], [1, 2, 8, 9, 10], [1, 2, 11, 12, 13]])
    labels = input_ids.clone()
    labels[:, :2] = -100
    return {'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids), 'labels': labels}


def _loss_and_grad(trainer):
    trainer.model.zero_grad(set_to_none=True)
    loss, _ = trainer.get_batch_loss_metrics(trainer.model, _batch())
    loss.backward()
    gradients = torch.cat([parameter.grad.flatten() for parameter in trainer.model.parameters()])
    return loss.detach(), gradients


@pytest.mark.parametrize('label_smoothing', [0.0, 0.2])
def test_exo_mixed_loss_order_and_repeated_calls(label_smoothing):
    first = _trainer(['sigmoid', 'exo_pair'], [0.7, 0.3], label_smoothing)
    reordered = _trainer(['exo_pair', 'sigmoid'], [0.3, 0.7], label_smoothing)
    expected_loss, expected_grad = _loss_and_grad(first)
    for trainer in (reordered, first):
        loss, grad = _loss_and_grad(trainer)
        torch.testing.assert_close(loss, expected_loss, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(grad, expected_grad, rtol=1e-12, atol=1e-12)
        assert trainer.label_smoothing == label_smoothing


def test_exo_evaluation_does_not_change_next_training_loss():
    control = _trainer(['sigmoid', 'exo_pair'], [0.7, 0.3])
    evaluated = _trainer(['sigmoid', 'exo_pair'], [0.7, 0.3])
    with torch.no_grad():
        evaluated.get_batch_loss_metrics(evaluated.model, _batch(), train_eval='eval')
    expected_loss, expected_grad = _loss_and_grad(control)
    loss, grad = _loss_and_grad(evaluated)
    torch.testing.assert_close(loss, expected_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad, expected_grad, rtol=1e-12, atol=1e-12)
    assert evaluated.label_smoothing == 0.0


@pytest.mark.parametrize('label_smoothing', [0.0, 0.2])
def test_exo_matches_binary_reverse_kl(label_smoothing):
    trainer = _trainer(['exo_pair'], [1.0], label_smoothing)
    chosen = torch.tensor([-1.0, -2.0], dtype=torch.float64, requires_grad=True)
    rejected = torch.tensor([-3.0, -1.0], dtype=torch.float64, requires_grad=True)
    ref_chosen = torch.tensor([-2.0, -1.5], dtype=torch.float64)
    ref_rejected = torch.tensor([-1.0, -2.5], dtype=torch.float64)
    logits = trainer.beta * (chosen - rejected - ref_chosen + ref_rejected)
    log_probs = torch.stack((F.logsigmoid(logits), F.logsigmoid(-logits)), dim=-1)
    smoothing = label_smoothing or 1e-3
    target_log_probs = torch.tensor([1 - smoothing, smoothing], dtype=torch.float64).log()
    expected = F.kl_div(target_log_probs.expand_as(log_probs), log_probs, reduction='none', log_target=True).sum(-1)
    actual, _, _ = trainer.dpo_loss(chosen, rejected, ref_chosen, ref_rejected, 'exo_pair')
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), (chosen, rejected), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (chosen, rejected))
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value)
    assert trainer.label_smoothing == label_smoothing
