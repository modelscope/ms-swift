"""Standalone entry that runs ONE offline-preference RLHF side and writes its per-step loss trajectory.

Launched by ``test_parity_legacy.py`` as TWO separate plain-python subprocesses -- one per side (legacy
``swift.rlhf_main`` -> HF/TRL ``<TYPE>Trainer``; dev ``run_rl`` -> Ray ``PreferenceLoop``) -- so their
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
    python rl_parity.py --backend {legacy|dev} --rlhf_type {cpo|orpo|...} --data DATA.jsonl \
        --out RESULT.json --out_dir DIR [--steps 4] [--lr 1e-4] [--tuner lora]

Writes {"backend", "rlhf_type", "tuner", "losses": [per-step train loss ...]} to --out.
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


def _run_legacy(data, out_dir, tuner, rlhf_type, steps, lr):
    """legacy ``swift.rlhf_main`` (HF/RLHFArguments -> TRL trainer); per-step loss from ``log_history``."""
    from swift import rlhf_main
    argv = [
        '--rlhf_type', rlhf_type,
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
    msg = rlhf_main(argv)
    return [rec['loss'] for rec in msg['log_history'] if 'loss' in rec]


def _run_dev(data, out_dir, tuner, rlhf_type, steps, lr):
    """dev ``run_rl`` (Ray PreferenceLoop); per-step loss from the returned history.

    Hyperparameters mirror ``_run_legacy`` exactly (that is the point). ``tuner='full'`` is NOT supported by
    dev (it raises ``NotImplementedError``: only lora/adalora/trainable_tokens), so a full-param parity cell
    has no dev side -- ``test_parity_legacy`` records that as an honest exclusion rather than run it here.
    """
    from swift.dev.tests.feature.rl.conftest import rl_configs, run_rl
    tuner_over = {'lora_rank': 8, 'lora_alpha': 32, 'target_modules': ['all-linear']} if tuner == 'lora' else {}
    configs = rl_configs(
        rlhf_type=rlhf_type,
        model=_model_path(),
        model_type='qwen2',
        template='qwen2_5',
        dataset=data,
        out_dir=out_dir,
        nproc=1,
        tuner=tuner,
        tuner_over=tuner_over,
        max_steps=steps,
        train_over={'learning_rate': lr, 'lr_scheduler': 'constant', 'warmup_ratio': 0.0},
        template_over={'max_length': 1024},
        rlhf_over={'beta': BETA, 'rpo_alpha': RPO_ALPHA},
    )
    history = run_rl(configs)
    return [h['loss'] for h in history]


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
    parser.add_argument('--rlhf_type', default='cpo')
    parser.add_argument('--tuner', default='lora')
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    if args.backend == 'legacy':
        losses = _run_legacy(args.data, args.out_dir, args.tuner, args.rlhf_type, args.steps, args.lr)
    else:
        losses = _run_dev(args.data, args.out_dir, args.tuner, args.rlhf_type, args.steps, args.lr)

    with open(args.out, 'w') as f:
        json.dump(
            {
                'backend': args.backend,
                'rlhf_type': args.rlhf_type,
                'tuner': args.tuner,
                'losses': losses
            }, f)
    print(f'RUNNER_DONE backend={args.backend} rlhf_type={args.rlhf_type} losses={losses}', flush=True)
    _cleanup_output_dir(args.out_dir)


if __name__ == '__main__':
    main()
