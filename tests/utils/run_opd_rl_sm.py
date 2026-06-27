#!/usr/bin/env python3
"""Smoke tests for OPD-RL PR. Run: CUDA_VISIBLE_DEVICES=0,1 python tests/utils/run_opd_rl_sm.py
"""
import os
import sys
import traceback

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
PY = sys.executable

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append((name, ok, detail))
    mark = 'PASS' if ok else 'FAIL'
    print(f'[{mark}] {name} {detail}', flush=True)


def smoke_plain_grpo_liger():
    from swift import RLHFArguments, rlhf_main
    rlhf_main(
        RLHFArguments(
            rlhf_type='grpo',
            model='Qwen/Qwen2.5-0.5B-Instruct',
            tuner_type='full',
            dataset=['modelscope/gsm8k#20'],
            reward_funcs=['format'],
            num_generations=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=2,
            save_steps=100,
            logging_steps=1,
            max_completion_length=128,
            max_length=512,
            use_liger_kernel=True,
            report_to=[],
        ))


def smoke_plain_grpo():
    from swift import RLHFArguments, rlhf_main
    rlhf_main(
        RLHFArguments(
            rlhf_type='grpo',
            model='Qwen/Qwen2.5-0.5B-Instruct',
            tuner_type='full',
            dataset=['modelscope/gsm8k#20'],
            reward_funcs=['format'],
            num_generations=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=2,
            save_steps=100,
            logging_steps=1,
            max_completion_length=256,
            max_length=512,
            report_to=[],
        ))


def smoke_opd_rl():
    from swift import RLHFArguments, rlhf_main
    rlhf_main(
        RLHFArguments(
            rlhf_type='grpo',
            model='Qwen/Qwen2.5-0.5B-Instruct',
            teacher_model='Qwen/Qwen2.5-7B-Instruct',
            tuner_type='full',
            dataset=['modelscope/gsm8k#20'],
            reward_funcs=[],
            num_generations=1,
            teacher_kl_coef=1.0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=2,
            save_steps=100,
            logging_steps=1,
            max_completion_length=128,
            max_length=512,
            offload_teacher_model=True,
            report_to=[],
        ))


def smoke_ray_teacher_server_rejected():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from swift.ray.megatron.grpo_trainer import GRPOTrainer

    args = SimpleNamespace(
        teacher_model=None,
        teacher_model_server='http://localhost:8000',
        teacher_model_dir=None,
        _teacher_use_disable_adapter=False,
        teacher_kl_coef=1.0,
        num_generations=8,
        advantage_estimator='grpo',
        scale_rewards='group',
        kl_in_reward=False,
        reward_funcs=[],
        use_gym_env=False,
        multi_turn_scheduler=None,
        max_turns=None,
        gym_env=None,
        router_replay_mode='disabled',
        dynamic_sample=False,
        max_resample_times=3,
        truncation_strategy='left',
        reward_weights=None,
        global_batch_size=4,
        temperature=0.9,
        beta=0.04,
        generation_batch_size=None,
        steps_per_generation=1,
    )
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer._data_info = {'_driver_args': args, 'template': MagicMock()}
    try:
        GRPOTrainer._prepare_state(trainer)
        record('ray_teacher_server', False, 'expected NotImplementedError')
    except NotImplementedError as e:
        record('ray_teacher_server', True, str(e)[:80])


def main():
    tests = [
        ('ray_teacher_server', smoke_ray_teacher_server_rejected),
        ('plain_grpo_hf', smoke_plain_grpo),
        ('plain_grpo_liger_hf', smoke_plain_grpo_liger),
        ('opd_rl_hf', smoke_opd_rl),
    ]
    for name, fn in tests:
        try:
            fn()
            if name != 'ray_teacher_server':
                record(name, True)
        except Exception as e:
            record(name, False, f'{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}')

    print('\n=== Summary ===')
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f'{passed}/{len(RESULTS)} passed')
    for name, ok, detail in RESULTS:
        print(f'  {"OK" if ok else "XX"} {name}: {detail[:120]}')
    return 0 if passed == len(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
