# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import pytest
import torch
from collections import defaultdict
from types import SimpleNamespace

from swift.rlhf_trainers.grpo_trainer import GRPOTrainer


@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('training', [False, True])
def test_missing_task_does_not_poison_logging_window(dynamic, training):
    trainer = object.__new__(GRPOTrainer)
    trainer.model = torch.nn.Linear(1, 1).train(training)
    trainer.accelerator = SimpleNamespace(device=torch.device('cpu'))
    trainer.num_generations = trainer.num_generations_eval = 2
    trainer.dynamic_num_samples = dynamic
    trainer.kl_in_reward = False
    trainer.beta = 0.0
    trainer.reward_weights = torch.ones(2)
    trainer.reward_func_names = ['task_a', 'task_b']
    trainer.advantage_estimator = 'grpo'
    trainer.scale_rewards = 'group'
    trainer._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
    trainer._logs = {'rewards': defaultdict(list)}
    samples = [SimpleNamespace(prompt_id='p', request_id='a'), SimpleNamespace(prompt_id='p', request_id='b')]
    batches = [[[1., float('nan')], [3., float('nan')]], [[float('nan'), 2.], [float('nan'), 4.]],
               [[5., float('nan')], [7., float('nan')]]]
    for batch in batches:
        rewards = torch.tensor(batch)
        current_samples = samples
        if dynamic:
            rewards = rewards.repeat_interleave(2, dim=0)
            current_samples = [sample for sample in samples for _ in range(2)]
        advantages = trainer._compute_advantages(current_samples, rewards, [])
        assert torch.isfinite(advantages).all()
    metrics = trainer._metrics['train' if training else 'eval']
    assert metrics['rewards/task_a/mean'] == [2.0, 6.0], metrics
    assert metrics['rewards/task_b/mean'] == [3.0], metrics
    assert metrics['rewards/task_a/std'] == pytest.approx([math.sqrt(2), math.sqrt(2)])
    assert metrics['rewards/task_b/std'] == pytest.approx([math.sqrt(2)])
    assert all(math.isfinite(value) for values in metrics.values() for value in values), metrics
    assert math.isclose(sum(metrics['rewards/task_a/mean']) / 2, 4.0)
