# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
from accelerate import Accelerator
from collections import defaultdict
from transformers import Qwen3Config, Qwen3ForCausalLM
from types import SimpleNamespace

from swift.rl_core.data import GRPOBatch
from swift.rlhf_trainers.grpo_trainer import GRPOTrainer


def _case(loss_type, off_policy_mask, training, chunk_size):
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4)).train(training)
    trainer = object.__new__(GRPOTrainer)
    trainer.model = model
    trainer.args = SimpleNamespace(
        per_device_train_batch_size=chunk_size,
        per_device_eval_batch_size=chunk_size,
        delta=None,
        report_to=[],
        gradient_accumulation_steps=1)
    trainer.template = SimpleNamespace(padding_free=False, sequence_parallel_size=1)
    trainer.accelerator = Accelerator(cpu=True)
    trainer.compute_entropy = False
    trainer.overlong_filter = False
    trainer.beta = 0.0
    trainer.kl_in_reward = False
    trainer.rollout_importance_sampling_mode = None
    trainer.log_rollout_offpolicy_metrics = False
    trainer.disable_rollout_importance_sampling = False
    trainer.importance_sampling_level = 'token'
    trainer.loss_type = loss_type
    trainer.off_policy_sequence_mask_delta = 0.5 if off_policy_mask else None
    trainer.num_generations = 2
    trainer.real_tau = 1.0
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer.fipo_gamma = 0.9
    trainer.fipo_clip_range = None
    trainer.fipo_safety_threshold = None
    trainer.sdar_loss_coef = 0
    trainer.chord_sft_iterator = None
    trainer.dynamic_num_samples = True
    trainer.is_multimodal = False
    trainer.use_liger_loss = False
    trainer.model_accepts_loss_kwargs = False
    trainer.compute_loss_func = None
    trainer.model_kwarg_keys = ['use_cache', 'logits_to_keep']
    trainer.temperature = 1.0
    trainer._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
    # The normal setup enables dynamic teacher capability without a teacher model.
    trainer._setup_teacher()
    assert trainer._is_dynamic_self_distillation and trainer._has_teacher
    assert not trainer._has_teacher_explicit()
    inputs = {
        'input_ids': torch.tensor([[1, 2, 3, 4], [1, 2, 5, 0], [1, 6, 7, 8], [1, 6, 9, 0]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0]])
    }
    batch = GRPOBatch(
        completion_mask=torch.tensor([[True, True], [True, False], [True, True], [True, False]]),
        truncated_mask=torch.zeros(4, dtype=torch.bool),
        seq_lengths=torch.tensor([2, 1, 2, 1]),
        advantages=torch.tensor([[1., 1.], [-1., -1.], [1., 1.], [-1., -1.]]),
        num_items_in_batch=torch.tensor(6),
        logits_to_keep=2)
    with torch.no_grad():
        logps, _ = trainer._get_per_token_logps_and_entropies(model, inputs.copy(), batch)
    batch.old_per_token_logps = logps - torch.tensor([[0.1], [-0.1], [0.05], [-0.05]])
    return trainer, inputs, batch


@pytest.mark.parametrize('loss_type,off_policy_mask', [('real', False), ('fipo', False), ('grpo', True),
                                                       ('grpo', False)])
@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('chunk_size', [2, 4])
def test_reward_only_batches_allow_dynamic_teacher_capability(loss_type, off_policy_mask, training, chunk_size):
    trainer, inputs, batch = _case(loss_type, off_policy_mask, training, chunk_size)
    assert batch.teacher_per_token_logps is None
    loss = trainer.compute_loss(trainer.model, {'model_inputs': inputs, 'grpo_batch': batch})
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainer.model.parameters())
    assert any(torch.count_nonzero(parameter.grad) > 0 for parameter in trainer.model.parameters())


@pytest.mark.parametrize('loss_type,off_policy_mask', [('real', False), ('fipo', False), ('grpo', True)])
@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('chunk_size', [2, 4])
def test_active_teacher_keeps_unsupported_combinations_rejected(loss_type, off_policy_mask, training, chunk_size):
    trainer, inputs, batch = _case(loss_type, off_policy_mask, training, chunk_size)
    batch.teacher_per_token_logps = batch.old_per_token_logps.clone()
    # Check the original batch before dynamic chunks can discard their teacher metadata.
    with pytest.raises(ValueError, match='OPD-RL'):
        trainer.compute_loss(trainer.model, {'model_inputs': inputs, 'grpo_batch': batch})
