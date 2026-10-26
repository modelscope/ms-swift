"""Standalone entry that runs ONE offline-preference RLHF side and writes its per-step loss trajectory.

Launched by ``test_parity_legacy.py`` as TWO separate plain-python subprocesses -- one per side (legacy
``swift.rlhf_main`` -> HF/TRL ``<TYPE>Trainer``; dev ``run_rl`` -> Ray ``run_dpo``) -- so their
per-step losses can be compared. The two pipelines cannot share one interpreter: legacy builds a TRL
trainer on the HF Trainer/accelerate stack while dev initializes a Ray session plus the twinkle runtime
(worker actors, loss registry), and running both in one process risks cross-contaminating those globals.
Isolating each side is the sound way to compare them -- the same rationale as ``megatron_sft.py`` (which
isolates for Megatron/TransformerEngine global state). Unlike ``megatron_sft`` this needs NO torchrun: the
offline preference family (dpo/kto/cpo/orpo/simpo/rm) trains on a single card with no rollout, so each side
is one plain process.

The dev side reuses the SAME config builder + driver every other RL e2e test uses
(``conftest.rl_configs``/``run_rl``), so parity is measured on the production path, not a bespoke one. Both
sides are fed matched deterministic hyperparameters (constant LR / no warmup / fixed seed / shuffle off /
``logging_steps=1``) so the trajectories are directly comparable.

Usage:
    python rl_parity.py --backend {legacy|dev} --rl_type {cpo|orpo|...} --data DATA.jsonl \
        --out RESULT.json --out_dir DIR [--steps 4] [--lr 1e-4] [--tuner lora]

Writes {"backend", "rl_type", "tuner", "losses": [per-step train loss ...]} to --out.
"""
import argparse
import json
import os

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
BETA = 0.1
# ``rpo_alpha`` is the NLL-on-chosen weight (legacy TRL's rpo/cpo_alpha; dev maps it to twinkle's
# ``sft_weight`` via loss/configure.py, where a falsy 0.0 falls back to the DPOLoss default -- identical to
# passing nothing). It is set explicitly and EQUAL on both sides, which is the parity contract: same
# hyperparameters in, same loss out. beta is the preference-contrast temperature, likewise matched.
RPO_ALPHA = 0.0


def _model_path():
    from modelscope import snapshot_download
    return snapshot_download(MODEL)


#: The content-keyed ORM shared by BOTH sides of the grpo reward-path parity (see test_parity_legacy).
#: Keying on completion content gives most groups a non-zero std (GRPO's advantage needs one) and
#: leaves a reachable 1.0 optimum, so a learning policy's mean reward climbs over steps. The built-in
#: 'format'/'accuracy' are NOT shared on purpose: legacy's Format anchors the whole string while
#: twinkle's FormatReward does not (a real semantic fork), and accuracy needs solvable math.
#:
#: The two pipelines hand the scorer different row shapes: legacy calls it with a list of completion
#: STRINGS, dev with trajectory rows (``messages`` + extras). One adaptive function keeps the FORMULA
#: single-sourced -- the compared quantity is the same code on both sides, which is the point.
def grpo_content_reward(completions, **kwargs):
    if isinstance(completions[0], str):
        texts = list(completions)
    else:
        from twinkle.data_format import last_assistant_text
        texts = [last_assistant_text(row) for row in completions]
    return [float(sum(ord(ch) for ch in text) % 13) / 12.0 for text in texts]


def _write_grpo_data(path, n: int = 32) -> str:
    """Prompt-only rows for the GRPO parity run; 32 rows so successive rollout windows draw fresh
    prompts (a 4-row set replays the same prompts every window and the reward path flatlines for a
    data reason, not a training one)."""
    questions = ('1 + 1 = ?', 'What is twice three?', 'How much is 10 - 4?', 'Compute 2 x 5.',
                 'What is half of twelve?', 'Add 7 and 6.', 'What is 3 squared?', 'Subtract 9 from 20.',
                 'Multiply 4 by 4.', 'What is 15 divided by 5?', 'Sum 8 and 8.', 'What is 5 plus 9?',
                 'Count 6 times 3.', 'What is 21 minus 11?', 'Double the number seven.', 'What follows 12 + 12?')
    rows = [{'messages': [{'role': 'user', 'content': f'Solve: {questions[i % len(questions)]} Show your reasoning.'}]}
            for i in range(n)]
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + chr(10))
    return str(path)


#: Legacy reward_func name after _prepare_rewards: a plain function is registered by __name__, so the
#: log_history key is deterministic on both sides.
REWARD_FUNC_NAME = grpo_content_reward.__name__


def _run_legacy(data, out_dir, tuner, rl_type, steps, lr):
    """legacy ``swift.rlhf_main`` (HF/RLHFArguments -> TRL trainer); per-step loss from ``log_history``."""
    from swift import rlhf_main
    argv = [
        '--rlhf_type', rl_type,
        '--model', _model_path(),
        '--template', 'qwen2_5',
        '--dataset', data,
        '--torch_dtype', 'bfloat16',
        '--max_length', '1024',
        '--learning_rate', str(lr),
        '--lr_scheduler_type', 'constant',
        '--warmup_ratio', '0.0',
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '1',
        '--max_steps', str(steps),
        '--seed', '42',
        '--data_seed', '42',
        '--dataset_shuffle', 'false',
        '--train_dataloader_shuffle', 'false',
        '--logging_steps', '1',
        '--split_dataset_ratio', '0.0',
        '--beta', str(BETA),
        '--rpo_alpha', str(RPO_ALPHA),
        '--output_dir', out_dir,
        '--report_to', 'none',
        '--add_version', 'false',
        '--save_steps', '100000',
    ]
    if tuner == 'lora':
        argv += ['--tuner_type', 'lora', '--lora_rank', '8', '--lora_alpha', '32', '--target_modules', 'all-linear']
    else:
        argv += ['--tuner_type', tuner]
    rewards = None
    if rl_type == 'grpo':
        # The reward-path parity contract (below) needs the rollout ON with a colocated vLLM and the
        # SHARED content-keyed ORM. num_generations>1 + temperature>0 so a group's completions differ
        # (a greedy or single-sample group has std=0 -> advantage 0 -> nothing to compare climbing).
        argv += [
            '--use_vllm', 'true',
            '--vllm_mode', 'colocate',
            '--vllm_gpu_memory_utilization', '0.4',
            '--max_completion_length', '64',
            '--num_generations', '4',
            '--temperature', '1.0',
            '--per_device_train_batch_size', '4',
            '--generation_batch_size', '16',
            '--external_plugins', _reward_plugin_path(),
            '--reward_funcs', REWARD_FUNC_NAME,
        ]
    msg = rlhf_main(argv)
    history = msg['log_history']
    losses = [rec['loss'] for rec in history if 'loss' in rec]
    if rl_type == 'grpo':
        key = f'rewards/{REWARD_FUNC_NAME}/mean'
        rewards = [rec[key] for rec in history if key in rec]
    return losses, rewards


def _reward_plugin_path():
    """Write the shared ORM out as a one-file external plugin (legacy registers it into ``orms``)."""
    import textwrap
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_rl_parity_reward_plugin.py')
    with open(path, 'w') as f:
        f.write(textwrap.dedent(f'''
            from swift.rewards.orm import orms


            def {REWARD_FUNC_NAME}(completions, **kwargs):
                return [float(sum(ord(ch) for ch in text) % 13) / 12.0 for text in completions]


            orms['{REWARD_FUNC_NAME}'] = {REWARD_FUNC_NAME}
        ''').lstrip())
    return path


def _run_dev(data, out_dir, tuner, rl_type, steps, lr):
    """dev ``run_rl`` (Ray run_dpo); per-step loss from the returned history.

    Hyperparameters mirror ``_run_legacy`` exactly (that is the point). ``tuner='full'`` is NOT supported by
    dev (it raises ``NotImplementedError``: only lora/adalora/trainable_tokens), so a full-param parity cell
    has no dev side -- ``test_parity_legacy`` records that as an honest exclusion rather than run it here.
    """
    from swift.dev.tests.feature.rl.conftest import rl_configs, run_rl
    tuner_over = {'lora_rank': 8, 'lora_alpha': 32, 'target_modules': ['all-linear']} if tuner == 'lora' else {}
    rlhf_over = {'beta': BETA, 'rpo_alpha': RPO_ALPHA}
    reward_over, generation_over = None, None
    train_over = {'learning_rate': lr, 'lr_scheduler': 'constant', 'warmup_ratio': 0.0}
    rollout_over = {'vllm_gpu_memory_utilization': 0.4, 'vllm_enforce_eager': True, 'vllm_max_model_len': 512}
    if rl_type == 'grpo':
        # Mirror _run_legacy's grpo contract: colocated vLLM, 4-way groups at temperature 1, the SAME
        # content-keyed ORM. NOTE the same-name-different-semantics below -- the two CLIs disagree on
        # what generation_batch_size counts, so the two values differ on purpose to express the SAME
        # window (4 prompts x 4 generations = 16 samples = 4 optimizer steps at batch 4):
        #   legacy: total SAMPLES per vLLM call           -> 16
        #   dev:    PROMPT ROWS per rollout window        -> 4
        # Every 4 steps a FRESH window scores new completions, so the reward trajectory has one
        # distinct point per window. (Registered as a cross-repo naming divergence, not smoothed over.)
        rlhf_over = {'beta': 0.0, 'num_generations': 4, 'max_completion_length': 64}
        reward_over = {'orm': [grpo_content_reward]}
        generation_over = {'temperature': 1.0, 'max_new_tokens': 64}
        train_over.update(per_device_train_batch_size=4)
        rollout_over['generation_batch_size'] = 4
    configs = rl_configs(
        rl_type=rl_type,
        model=_model_path(),
        model_type='qwen2',
        template='qwen2_5',
        dataset=data,
        out_dir=out_dir,
        nproc=1,
        tuner=tuner,
        tuner_over=tuner_over,
        max_steps=steps,
        train_over=train_over,
        template_over={'max_length': 1024},
        rlhf_over=rlhf_over,
        reward_over=reward_over,
        generation_over=generation_over,
        rollout_over=rollout_over,
    )
    history = run_rl(configs)
    losses = [h['loss'] for h in history]
    rewards = None
    if rl_type == 'grpo':
        # The driver-side CompletionRewardMetric emits train/<name>_reward per rollout window; the
        # model-side GRPOMetric emits train/approx_kl etc. The reward PATH is the former.
        key = f'train/{REWARD_FUNC_NAME}_reward'
        rewards = [h[key] for h in history if key in h]
    return losses, rewards


def _cleanup_output_dir(out_dir):
    """Remove a finished run's output tree, best-effort (runs in pairs; never costs a measurement)."""
    import shutil
    if not out_dir or not os.path.isdir(out_dir):
        return
    try:
        shutil.rmtree(out_dir)
    except OSError as e:  # pragma: no cover - cleanup is best-effort
        print(f'  WARNING: could not clean up {out_dir}: {e}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['legacy', 'dev'], required=True)
    parser.add_argument('--rl_type', default='cpo')
    parser.add_argument('--tuner', default='lora')
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    if args.backend == 'legacy':
        losses, rewards = _run_legacy(args.data, args.out_dir, args.tuner, args.rl_type, args.steps, args.lr)
    else:
        losses, rewards = _run_dev(args.data, args.out_dir, args.tuner, args.rl_type, args.steps, args.lr)

    with open(args.out, 'w') as f:
        json.dump(
            {
                'backend': args.backend,
                'rl_type': args.rl_type,
                'tuner': args.tuner,
                'losses': losses,
                'rewards': rewards
            }, f)
    print(f'RUNNER_DONE backend={args.backend} rl_type={args.rl_type} losses={losses} rewards={rewards}',
          flush=True)
    _cleanup_output_dir(args.out_dir)


if __name__ == '__main__':
    main()
