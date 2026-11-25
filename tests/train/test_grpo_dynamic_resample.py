import pytest
import torch
from accelerate import Accelerator
from types import SimpleNamespace

import swift.rlhf_trainers.grpo_trainer as grpo_trainer_mod
from swift.rlhf_trainers.grpo_trainer import GRPOTrainer


def _make_trainer(
    *,
    num_items=64,
    per_device_train_batch_size=4,
    steps_per_generation=4,
    num_generations=4,
    num_iterations=1,
    dynamic_sample=True,
    truncation_strategy='delete',
    sequence_parallel_size=1,
    shuffle_dataset=True,
    seed=42,
    max_resample_times=3,
    good_after=0,
):
    trainer = object.__new__(GRPOTrainer)
    trainer.accelerator = Accelerator(cpu=True)
    trainer.args = SimpleNamespace(
        seed=seed,
        dynamic_sample=dynamic_sample,
        per_device_train_batch_size=per_device_train_batch_size,
        steps_per_generation=steps_per_generation,
        generation_batch_size=per_device_train_batch_size * steps_per_generation,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_drop_last=False,
        dataloader_prefetch_factor=None,
        process_index=0,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer._train_batch_size = per_device_train_batch_size
    trainer.num_generations = num_generations
    trainer.num_generations_eval = num_generations
    trainer.num_iterations = num_iterations
    trainer.shuffle_dataset = shuffle_dataset
    trainer.dynamic_num_samples = False
    trainer.max_resample_times = max_resample_times
    trainer.reward_weights = torch.tensor([1.0], dtype=torch.float32)
    trainer.model = SimpleNamespace(training=True)
    trainer.data_collator = lambda batch: batch
    trainer.template = SimpleNamespace(
        sequence_parallel_size=sequence_parallel_size,
        truncation_strategy=truncation_strategy,
    )
    trainer.train_dataset = [{
        'messages': [{
            'role': 'user',
            'content': f'prompt-{idx}',
        }],
        'prompt_idx': idx,
        'reward_mode': 'bad' if idx < good_after else 'good',
    } for idx in range(num_items)]
    return trainer


@pytest.mark.parametrize(
    'steps_per_generation,num_iterations,sp_size',
    [
        (4, 1, 1),
        (2, 2, 1),
        (1, 3, 1),
        (2, 2, 2),
    ],
)
def test_dynamic_resample_iterator_does_not_repeat_batches(
    monkeypatch,
    steps_per_generation,
    num_iterations,
    sp_size,
):
    monkeypatch.setattr(grpo_trainer_mod, 'sequence_parallel', SimpleNamespace(world_size=sp_size))
    trainer = _make_trainer(
        num_items=64,
        per_device_train_batch_size=4,
        steps_per_generation=steps_per_generation,
        num_generations=4,
        num_iterations=num_iterations,
        sequence_parallel_size=sp_size,
        shuffle_dataset=True,
        seed=42,
    )

    # Normal training dataloader must keep repeating each generated batch
    # num_iterations * steps_per_generation * sp_size times.
    expected_repeat = num_iterations * steps_per_generation * sp_size
    train_iter = iter(trainer.get_train_dataloader())
    train_batches = [tuple(row['prompt_idx'] for row in next(train_iter)) for _ in range(expected_repeat + 1)]
    assert len(set(train_batches[:expected_repeat])) == 1
    assert train_batches[expected_repeat] != train_batches[0]

    trainer._prepare_resample_data_iterator()
    assert trainer.args.seed == 42
    assert not getattr(trainer, '_resample_mode', False)

    resample_batches = [
        tuple(sorted({row['prompt_idx']
                      for row in next(trainer.dynamic_resample_iterator)})) for _ in range(4)
    ]
    assert len(set(resample_batches)) == 4


def test_dynamic_sampling_replaces_std_zero_groups_with_fresh_prompts():
    trainer = _make_trainer(
        num_items=24,
        per_device_train_batch_size=4,
        steps_per_generation=4,
        num_generations=4,
        num_iterations=1,
        shuffle_dataset=False,
        seed=42,
        max_resample_times=3,
        good_after=4,
    )
    trainer._prepare_resample_data_iterator()

    seen_resample_batches = []

    def fake_generate_completions(samples):
        seen_resample_batches.append(tuple(s.extra['prompt_idx'] for s in samples))
        return samples

    def fake_compute_rewards_per_func(samples):
        rewards = []
        for idx, sample in enumerate(samples):
            if sample.extra.get('reward_mode') == 'good':
                rewards.append([float((idx % 2) == 0)])
            else:
                rewards.append([1.0])
        return torch.tensor(rewards, dtype=torch.float32)

    trainer._generate_completions = fake_generate_completions
    trainer._compute_rewards_per_func = fake_compute_rewards_per_func

    origin_rows = []
    for pid, mode in [(100, 'good'), (101, 'bad'), (102, 'bad'), (103, 'bad')]:
        for _ in range(4):
            origin_rows.append({
                'messages': [{
                    'role': 'user',
                    'content': f'prompt-{pid}',
                }],
                'prompt_idx': pid,
                'reward_mode': mode,
            })
    origin_samples = trainer.to_samples(origin_rows)
    origin_rewards = fake_compute_rewards_per_func(origin_samples)

    out_samples, out_rewards = trainer._dynamic_sampling(origin_samples, origin_rewards)

    assert len(seen_resample_batches) >= 2
    assert seen_resample_batches[0] != seen_resample_batches[1]
    assert len(out_samples) == 16
    assert torch.all(trainer.compute_std(out_samples, out_rewards) > 0)


def test_truncated_resample_iterator_does_not_repeat_with_num_iterations():
    trainer = _make_trainer(
        num_items=16,
        per_device_train_batch_size=2,
        steps_per_generation=2,
        num_generations=2,
        num_iterations=3,
        dynamic_sample=False,
        truncation_strategy='raise',
        shuffle_dataset=True,
        seed=7,
    )

    trainer._prepare_resample_data_iterator()
    assert trainer.num_generations == 2
    assert trainer.args.generation_batch_size == 4
    assert trainer.args.steps_per_generation == 2
    assert trainer.args.seed == 7
    assert not getattr(trainer, '_resample_mode', False)

    drawn_batches = [next(trainer.truncated_resample_iterator) for _ in range(4)]
    prompt_ids = [row['prompt_idx'] for batch in drawn_batches for row in batch]
    assert len(prompt_ids) == 8
    assert len(set(prompt_ids)) == 8


def test_prepare_resample_data_iterator_restores_state_on_error(monkeypatch):
    trainer = _make_trainer(dynamic_sample=True, truncation_strategy='delete', seed=19)

    def broken_dataloader():
        raise RuntimeError('boom')

    monkeypatch.setattr(trainer, 'get_train_dataloader', broken_dataloader)
    with pytest.raises(RuntimeError, match='boom'):
        trainer._prepare_resample_data_iterator()

    assert trainer.args.seed == 19
    assert not getattr(trainer, '_resample_mode', False)
