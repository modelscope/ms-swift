"""Standalone torchrun entry that runs ONE Megatron SFT backend and writes its loss trajectory.

Launched by test_run_sft_e2e.py::test_run_sft_megatron_vs_legacy_loss as two SEPARATE torchrun
subprocesses (one per backend), because a legacy MegatronTrainer and dev-Megatron cannot coexist
in one process: the first run leaves Megatron/TransformerEngine global state (parallel_state, TE
attention backend, RNG) that NaN-poisons the second. Isolating each in its own process is the only
sound way to compare their losses.

Usage (under torchrun --nproc_per_node=N):
    python _megatron_sft_runner.py --backend {dev|legacy} --data DATA.jsonl --out RESULT.json \
        --steps S --lr LR

Writes {"backend": ..., "losses": [per-step train loss ...]} to --out on rank 0.
"""
import argparse
import json
import os

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'


def _model_path():
    from modelscope import snapshot_download
    return snapshot_download(MODEL)


def _run_dev(data_path, out_dir, steps, lr):
    """dev-Megatron via run_sft(mode='local'), no checkpoint (_save_final=False)."""
    from modelscope import snapshot_download

    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                   TrainConfig)
    from swift.dev.recipe import run_sft

    model_path = snapshot_download(MODEL)
    history = run_sft(
        # bf16 to match legacy's production dtype (Qwen2.5 config dtype). See _run_legacy.
        ModelConfig(model=model_path, torch_dtype='bfloat16'),
        TemplateConfig(template='qwen2_5', max_length=512),
        DatasetConfig(dataset=[data_path], dataset_shuffle=False),
        # bs=1/ga=1 mirrors the HF legacy-vs-dev comparison: no GA/scheduler compounding, so the
        # first-step loss (no update yet) is directly comparable across pipelines.
        # No `optim=`: it is an HF-only knob and validate_configs rejects a non-default value on the
        # Megatron backend (break-change #3). Megatron runs Adam regardless, so dropping it does not
        # change what this test trains -- it just stops the run from failing validation.
        TrainConfig(
            learning_rate=lr,
            lr_scheduler_type='constant',
            warmup_ratio=0.0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=steps,
            max_grad_norm=1.0),
        DistributedConfig(
            backend='megatron',
            bridge_backend='mcore-bridge',
            mode='local',
            nproc_per_node=int(os.environ.get('WORLD_SIZE', '2'))),
        CheckpointConfig(),
        output_dir=out_dir,
        _save_final=False)
    return [r['loss'] for r in history]


def _dev_cli_argv(data_path, out_dir, steps, lr):
    """The Megatron CLI argv shared by the dev CLI and legacy, so both parse the SAME flags.

    Mirrors _run_legacy's list exactly (that is the point: the dev CLI claims argv compatibility on
    legacy's MegatronSftArguments surface). global_batch_size=WORLD_SIZE with micro_batch_size=1
    gives one sample per DP rank -> gradient_accumulation_steps derives to 1, matching _run_dev.
    """
    world_size = os.environ.get('WORLD_SIZE', '2')
    return [
        '--model',
        _model_path(),
        '--dataset',
        data_path,
        '--template',
        'qwen2_5',
        '--max_length',
        '512',
        '--train_iters',
        str(steps),
        '--micro_batch_size',
        '1',
        '--global_batch_size',
        str(int(world_size)),
        '--lr',
        str(lr),
        '--lr_decay_style',
        'constant',
        '--min_lr',
        str(lr),
        '--tensor_model_parallel_size',
        '1',
        '--pipeline_model_parallel_size',
        '1',
        '--no_save_optim',
        'true',
        '--no_save_rng',
        'true',
        '--finetune',
        'true',
        '--dataset_shuffle',
        'false',
        '--train_dataloader_shuffle',
        'false',
        '--output_dir',
        out_dir,
        '--add_version',
        'false',
        '--save_steps',
        '100000',
        '--logging_steps',
        '1',
    ]


def _run_dev_cli(data_path, out_dir, steps, lr):
    """dev-Megatron driven through the CLI (swift.dev.cli.megatron.megatron_sft_main).

    Distinct from _run_dev, which builds Configs directly: this exercises the argv -> Config mapping
    (renames lr/train_iters/micro_batch_size, the GA derivation from global_batch_size, and the
    dev-only backend/mode/nproc_per_node) on the real MegatronSftArguments parser. A mapping bug that
    silently leaves a hyperparameter at its dev default shows up here as a loss divergence.

    Named `dev_cli` rather than replacing `dev` so the two remain separable: if only this one moves,
    the mapping is at fault; if both move, the Megatron path itself is.
    """
    from swift.dev.cli.megatron import megatron_sft_main as dev_megatron_sft_main

    history = dev_megatron_sft_main(_dev_cli_argv(data_path, out_dir, steps, lr))
    return [r['loss'] for r in history]


def _run_legacy(data_path, out_dir, steps, lr):
    """legacy megatron_sft_main; per-step loss captured by patching BaseMegatronTrainer.on_log
    (legacy exposes no returnable log_history). loss is logged as [sum, count] -> divide."""
    captured = []
    from swift.megatron.trainers.base import BaseMegatronTrainer
    orig_on_log = BaseMegatronTrainer.on_log

    def patched_on_log(self, logs, prefix=''):
        v = logs.get('loss')
        if v is not None:
            captured.append(float(v[0] / v[1]) if hasattr(v, '__len__') and len(v) == 2 else float(v))
        return orig_on_log(self, logs, prefix)

    BaseMegatronTrainer.on_log = patched_on_log
    try:
        from modelscope import snapshot_download

        from swift.megatron import megatron_sft_main
        model_path = snapshot_download(MODEL)
        megatron_sft_main([
            '--model',
            model_path,
            '--dataset',
            data_path,
            '--template',
            'qwen2_5',
            '--max_length',
            '512',
            '--train_iters',
            str(steps),
            '--micro_batch_size',
            '1',
            '--global_batch_size',
            str(int(os.environ.get('WORLD_SIZE', '2'))),
            '--lr',
            str(lr),
            '--lr_decay_style',
            'constant',
            '--min_lr',
            str(lr),
            '--tensor_model_parallel_size',
            '1',
            '--pipeline_model_parallel_size',
            '1',
            # bf16 is megatron's production dtype (and Qwen2.5's config dtype -> legacy auto-picks
            # bf16 when no precision flag is passed). This is what real users run, so the dev-vs-
            # legacy comparison is on the production path. (fp32 is not a real megatron path: legacy
            # maps --torch_dtype float32 to fp16=True; dev-Megatron's fp32 construction correctness
            # is already gated by the two-bridge bit-identical test.)
            '--no_save_optim',
            'true',
            '--no_save_rng',
            'true',
            # finetune=True: load the pretrained HF weights (converted to mcore) instead of random init.
            '--finetune',
            'true',
            # Disable BOTH shuffle knobs so legacy iterates the dataset in the same natural order as
            # dev (which passes dataset_shuffle=False). Otherwise the two sides feed different samples
            # to step-1 and the loss mismatch would be a test-setup artifact, not a construction bug.
            '--dataset_shuffle',
            'false',
            '--train_dataloader_shuffle',
            'false',
            '--output_dir',
            out_dir,
            '--add_version',
            'false',
            '--save_steps',
            '100000',
            '--logging_steps',
            '1',
        ])
    finally:
        BaseMegatronTrainer.on_log = orig_on_log
    return captured


def _cleanup_output_dir(out_dir: str) -> None:
    """Remove a finished run's output tree, best-effort.

    Megatron checkpoints are GB-scale and these comparison runs are launched in pairs, so leaving
    them behind fills the 30G /tmp within a handful of invocations -- and the resulting ENOSPC
    surfaces as errors that do not mention the disk (a CheckpointException from the writer, or an
    OSError from pytest flushing stdout). Cleanup runs AFTER the result json so a failure to delete
    can never cost an expensive measurement, and it never raises for the same reason.
    """
    import shutil

    if not out_dir or not os.path.isdir(out_dir):
        return
    try:
        shutil.rmtree(out_dir)
        print(f'  cleaned up {out_dir}', flush=True)
    except OSError as e:
        print(f'  WARNING: could not clean up {out_dir}: {e}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['dev', 'dev_cli', 'legacy'], required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()

    if args.backend == 'dev':
        losses = _run_dev(args.data, args.out_dir, args.steps, args.lr)
    elif args.backend == 'dev_cli':
        losses = _run_dev_cli(args.data, args.out_dir, args.steps, args.lr)
    else:
        losses = _run_legacy(args.data, args.out_dir, args.steps, args.lr)

    if int(os.environ.get('RANK', '0')) == 0:
        with open(args.out, 'w') as f:
            json.dump({'backend': args.backend, 'losses': losses}, f)
        print(f'RUNNER_DONE backend={args.backend} losses={losses}', flush=True)
        _cleanup_output_dir(args.out_dir)


if __name__ == '__main__':
    main()
