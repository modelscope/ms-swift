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

For ``--rl_type grpo`` the runner additionally drives the full online rollout (colocated vLLM,
4-way groups at temperature 1.0) against the shared ``grpo_content_reward`` ORM on both sides and
emits the per-window reward trajectory: legacy via ``rewards/<ClassName>/mean`` from ``log_history``
(one point per generation cycle), dev via the driver-side ``CompletionRewardMetric`` (one point per
optimizer step; a window's 4 steps repeat its aggregate). Hyperparameters are matched across sides --
including the two REGISTERED cross-pipeline divergences that had to be pinned explicitly (see
``QWEN_SYSTEM`` and the generation_batch_size note): legacy's qwen2_5 template injects a default
system turn where dev passes rows through untouched, and generation_batch_size counts total SAMPLES
per vLLM call on legacy but PROMPT ROWS per rollout window on dev.

Usage:
    python rl_parity.py --backend {legacy|dev} --rl_type {cpo|orpo|grpo|...} --data DATA.jsonl \
        --out RESULT.json --out_dir DIR [--steps 4] [--lr 1e-4] [--tuner lora]
        [--seed 42] [--warmup 0.0]

Writes {"backend", "rl_type", "tuner", "losses": [per-step train loss ...],
"rewards": [per-window (legacy) / per-step (dev) reward, grpo only]} to --out.

Probe-only side channel: RL_PARITY_DUMP=<path> makes the shared ORM append every scored completion
(and the prompt-side rows it can see) to <path> for text-level legacy-vs-dev diagnosis.
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


#: The structured ORM shared by BOTH sides of the grpo reward-path parity (see test_parity_legacy).
#: A completion containing the reasoning phrase 'step by step' scores 1.0, everything else 0.1.
#: The phrase sits at a MID base rate (~27% over 112 sampled base-model completions of this prompt
#: family), which is the load-bearing property: at a LOW base rate (an 'answer is' check, ~6%)
#: most rollout groups have zero reward variance, the advantage signal exists only in the rare
#: hit window, and whether 32 steps amplify it is a per-seed coin flip -- a 7-run seed grid
#: (legacy default-sync/adapter-sync, several seeds) produced both strong climbs and flatlines
#: with BOTH weight-sync paths, i.e. outcome chaos, not a pipeline divergence. At 27% most
#: groups carry variance every window, so the climb is a property of the pipeline, not the seed.
#: The optimum is still REACHABLE by a real policy shift (say the phrase -- a language pattern
#: GRPO can amplify), unlike a digit-check (math prompts make digits near-certain -> reward
#: saturates at 1.0 with zero variance) or a content-hash (an argmax no generation policy can
#: pursue, flatlining by construction). The built-in 'format'/'accuracy' are NOT shared on
#: purpose: legacy's Format anchors the whole string while twinkle's FormatReward does not (a
#: real semantic fork), and accuracy needs solvable math.
#:
#: The two pipelines hand the scorer different row shapes: legacy calls it with a list of completion
#: STRINGS, dev with trajectory rows (``messages`` + extras). One adaptive function keeps the FORMULA
#: single-sourced -- the compared quantity is the same code on both sides, which is the point.
def grpo_content_reward(completions, **kwargs):
    if isinstance(completions[0], str):
        texts = list(completions)
        prompt_side = [{'k': k, 'v': str(v)[:200]} for k, v in kwargs.items()]
    else:
        from twinkle.data_format import last_assistant_text
        texts = [last_assistant_text(row) for row in completions]
        prompt_side = [{'role': m.get('role'), 'content': str(m.get('content'))[:200]}
                       for m in completions[0].get('messages', []) if m.get('role') != 'assistant']
    _dump_completions(texts, prompt_side)
    return [_answer_phrase_score(text) for text in texts]


def _dump_completions(texts, prompt_side=None):
    """Probe-only side channel: set RL_PARITY_DUMP to record every scored completion (texts + lengths,
    plus the first call's prompt-side rows) so the legacy/dev probes can be compared at the TEXT level,
    not just the scalar path."""
    path = os.environ.get('RL_PARITY_DUMP')
    if not path:
        return
    with open(path, 'a', encoding='utf-8') as f:
        if prompt_side is not None:
            f.write(json.dumps({'prompt_side': prompt_side}, ensure_ascii=False) + chr(10))
        for text in texts:
            f.write(json.dumps({'len': len(text), 'text': text}, ensure_ascii=False) + chr(10))


def _answer_phrase_score(text: str) -> float:
    return 1.0 if 'step by step' in text.lower() else 0.1


#: The system turn carried by every parity row. The two pipelines disagree on template-default
#: injection: legacy's qwen2_5 fills in the model's DEFAULT system prompt when a row has none, dev
#: passes the row through untouched (verified empirically: legacy's window-0 completions came out
#: in a formal LaTeX style, dev's in a chatty "Sure, ..." style, from the same weights). That style
#: fork sets the base rate AND position of the reward phrase, so the reward paths were incomparable
#: until the prompt is single-sourced: both sides honor an EXPLICIT row system (legacy skips its
#: default injection, dev's to_trajectory passes the row turn through), so the data pins it.
QWEN_SYSTEM = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'


def _write_grpo_data(path, n: int = 32) -> str:
    """Prompt-only rows for the GRPO parity run; 32 rows so successive rollout windows draw fresh
    prompts (a 4-row set replays the same prompts every window and the reward path flatlines for a
    data reason, not a training one)."""
    questions = ('1 + 1 = ?', 'What is twice three?', 'How much is 10 - 4?', 'Compute 2 x 5.',
                 'What is half of twelve?', 'Add 7 and 6.', 'What is 3 squared?', 'Subtract 9 from 20.',
                 'Multiply 4 by 4.', 'What is 15 divided by 5?', 'Sum 8 and 8.', 'What is 5 plus 9?',
                 'Count 6 times 3.', 'What is 21 minus 11?', 'Double the number seven.', 'What follows 12 + 12?')
    rows = [{'messages': [
        {'role': 'system', 'content': QWEN_SYSTEM},
        {'role': 'user', 'content': f'Solve: {questions[i % len(questions)]} Show your reasoning.'}]}
        for i in range(n)]
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + chr(10))
    return str(path)


#: Legacy reward_func name after _prepare_rewards: a plain function is registered by __name__, so the
#: log_history key is deterministic on both sides.
REWARD_FUNC_NAME = grpo_content_reward.__name__
#: legacy instantiates the plugin as a CLASS, so its metric key carries the class name instead.
LEGACY_REWARD_KEY = REWARD_FUNC_NAME.title().replace('_', '')


def _run_legacy(data, out_dir, tuner, rl_type, steps, lr, seed=42, warmup=0.0):
    """legacy ``swift.rlhf_main`` (HF/RLHFArguments -> TRL trainer); per-step loss from ``log_history``."""
    # legacy's single-process trainer must see EXACTLY ONE GPU: with several visible devices its
    # device_map auto-resolution engages and GRPO+vLLM rejects the run outright ("not compatible
    # with device_map"), and LoRA placement mixes devices. Pin to the first visible device so the
    # parity file runs green under one multi-GPU pytest invocation (the disaggregated cell needs 2).
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not visible or ',' in visible:
        os.environ['CUDA_VISIBLE_DEVICES'] = visible.split(',')[0] if visible else '0'
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
        '--warmup_ratio', str(warmup),
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '1',
        '--max_steps', str(steps),
        '--seed', str(seed),
        '--data_seed', str(seed),
        '--dataset_shuffle', 'false',
        '--train_dataloader_shuffle', 'false',
        '--logging_steps', '1',
        '--split_dataset_ratio', '0.0',
        # beta is the preference-contrast temperature for cpo/dpo/orpo, but in grpo it is the KL
        # coefficient: dev's grpo config runs beta=0.0 (no reference anchor), so legacy must too --
        # at beta=0.1 the KL term drags the policy back toward the reference every step and the
        # reward path stays flat for a HYPERPARAMETER reason, breaking the same-hyperparameters-in
        # contract the comparison rests on. legacy skips the reference model entirely at beta=0.0
        # (use_ref_model = beta != 0.0), matching dev's semantics.
        '--beta', str(0.0 if rl_type == 'grpo' else BETA),
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
    rewards = None
    extra = {}
    if rl_type == 'grpo':
        key = f'rewards/{LEGACY_REWARD_KEY}/mean'
        rewards = [rec[key] for rec in history if key in rec]
    elif rl_type in ('cpo', 'orpo'):
        # per-pair fingerprints for the parity diagnosis: the sequence logps identify which
        # dataset row a step consumed, and nll_loss separates the loss's two terms.
        extra = {
            'logps_chosen': [rec['logps/chosen'] for rec in history if 'logps/chosen' in rec],
            'logps_rejected': [rec['logps/rejected'] for rec in history if 'logps/rejected' in rec],
            'nll_loss': [rec['nll_loss'] for rec in history if 'nll_loss' in rec],
        }
    return losses, rewards, extra


def _reward_plugin_path():
    """Write the shared ORM out as a one-file external plugin (legacy registers it into ``orms``).

    legacy's ``_prepare_rewards`` instantiates a registered entry as ``orms[name](args=args)`` -- a
    CLASS, not a plain function (dev takes callables; legacy does not) -- so the plugin registers a
    class whose ``__call__`` carries the same formula.
    """
    import textwrap
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_rl_parity_reward_plugin.py')
    with open(path, 'w') as f:
        f.write(textwrap.dedent(f'''
            import json
            import os

            from swift.rewards.orm import orms


            class {REWARD_FUNC_NAME.title().replace('_', '')}:
                def __init__(self, args=None, **kwargs):
                    pass

                def __call__(self, completions, **kwargs):
                    def _score(text):
                        return 1.0 if 'step by step' in text.lower() else 0.1
                    dump = os.environ.get('RL_PARITY_DUMP')
                    if dump:
                        with open(dump, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(
                                {{'prompt_side': [{{'k': k, 'v': str(v)[:200]}} for k, v in kwargs.items()]}},
                                ensure_ascii=False) + chr(10))
                            for text in completions:
                                f.write(json.dumps({{'len': len(text), 'text': text}},
                                                   ensure_ascii=False) + chr(10))
                    return [_score(text) for text in completions]


            orms['{REWARD_FUNC_NAME}'] = {REWARD_FUNC_NAME.title().replace('_', '')}
        ''').lstrip())
    return path


def _run_dev(data, out_dir, tuner, rl_type, steps, lr, seed=42, warmup=0.0, sampler_mode='colocate'):
    """dev ``run_rl`` (Ray run_dpo); per-step loss from the returned history.

    Hyperparameters mirror ``_run_legacy`` exactly (that is the point). ``tuner='full'`` is NOT supported by
    dev (it raises ``NotImplementedError``: only lora/adalora/trainable_tokens), so a full-param parity cell
    has no dev side -- ``test_parity_legacy`` records that as an honest exclusion rather than run it here.
    """
    from swift.dev.tests.feature.rl.conftest import rl_configs, run_rl
    tuner_over = {'lora_rank': 8, 'lora_alpha': 32, 'target_modules': ['all-linear']} if tuner == 'lora' else {}
    rlhf_over = {'beta': BETA, 'rpo_alpha': RPO_ALPHA}
    reward_over, generation_over = None, None
    train_over = {'learning_rate': lr, 'lr_scheduler': 'constant', 'warmup_ratio': warmup, 'seed': seed}
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
        # 'colocate' rides the driver-less plain loop; 'disaggregated' rides the StreamingDriver
        # (separate sampler actor; at async_mode='none' generation still serializes with training,
        # so a single physical GPU can host both actors for the parity-sized run).
        rollout_over['sampler_mode'] = sampler_mode
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
        dataset_over={'data_seed': seed, 'train_dataloader_shuffle': False, 'dataset_shuffle': False},
        rlhf_over=rlhf_over,
        reward_over=reward_over,
        generation_over=generation_over,
        rollout_over=rollout_over,
    )
    history = run_rl(configs)
    losses = [h['loss'] for h in history]
    rewards = None
    extra = {}
    if rl_type in ('cpo', 'orpo'):
        extra = {
            'logps_chosen': [h['logps/chosen'] for h in history if 'logps/chosen' in h],
            'logps_rejected': [h['logps/rejected'] for h in history if 'logps/rejected' in h],
        }
    if rl_type == 'grpo':
        # The driver-side CompletionRewardMetric emits train/<name>_reward per rollout window; the
        # model-side GRPOMetric emits train/approx_kl etc. The reward PATH is the former.
        key = f'train/{REWARD_FUNC_NAME}_reward'
        rewards = [h[key] for h in history if key in h]
        # StreamingDriver-only bookkeeping (absent on the colocate plain loop and on legacy): the
        # publish count proves the trained policy reached the DISAGGREGATED sampler, and the
        # version-span lag proves the run stayed on-policy.
        extra = {
            'publishes': [h['stream_publishes'] for h in history if 'stream_publishes' in h],
            'spans': [h['version_span_mean'] for h in history if 'version_span_mean' in h],
        }
    return losses, rewards, extra


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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup', type=float, default=0.0)
    parser.add_argument('--sampler_mode', choices=['colocate', 'disaggregated'], default='colocate')
    args = parser.parse_args()

    if args.backend == 'legacy':
        losses, rewards, extra = _run_legacy(
            args.data, args.out_dir, args.tuner, args.rl_type, args.steps, args.lr, seed=args.seed,
            warmup=args.warmup)
    else:
        losses, rewards, extra = _run_dev(
            args.data, args.out_dir, args.tuner, args.rl_type, args.steps, args.lr, seed=args.seed,
            warmup=args.warmup, sampler_mode=args.sampler_mode)

    result = {
        'backend': args.backend,
        'rl_type': args.rl_type,
        'tuner': args.tuner,
        'losses': losses,
        'rewards': rewards,
    }
    if extra:
        # grpo's StreamingDriver bookkeeping keeps its historical 'driver' key (the reward-path
        # streaming test reads it); the cpo/orpo per-pair fingerprints ride under 'components'.
        result['driver' if args.rl_type == 'grpo' else 'components'] = extra
    with open(args.out, 'w') as f:
        json.dump(result, f)
    print(f'RUNNER_DONE backend={args.backend} rl_type={args.rl_type} losses={losses} rewards={rewards}',
          flush=True)
    _cleanup_output_dir(args.out_dir)


if __name__ == '__main__':
    main()
