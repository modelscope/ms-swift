"""Standalone torchrun entry: run ONE backend on a dense.sh-shaped Megatron LoRA config and dump
its per-step loss AND learning-rate trajectory.

Why a second runner instead of extending _megatron_sft_runner.py: that one pins a deliberately
minimal comparison (bs=1/ga=1/tp=1/constant-lr/full-param) so a step-1 loss mismatch is
attributable. This one is the opposite -- it mirrors examples/megatron/lora/dense.sh, i.e. every
knob the P2-4d argv mapping has a non-trivial branch for, all at once:

    tp=2 + sequence_parallel   -> dp = world/(tp*pp*cp) = 1, the GA-derivation divisor
    micro=16, global=16        -> gradient_accumulation_steps derives to 1 (not the trivial mbs=1)
    lr_warmup_fraction         -> warmup_ratio rename (previously always None)
    lr_decay_style cosine+min_lr -> the non-injective reverse lookup + Megatron's native lr floor
    recompute full/uniform/1   -> the recompute-triple branch NOT covered by 'selective'
    tuner_type lora            -> TunerConfig dispatch (slow tier had only full-param)
    num_train_epochs, no train_iters -> max_steps=-1 normalization

What is compared, and what is actually bit-level (see the test for the assertions):
  L1  step-0 loss              -- no optimizer update yet, so a pure forward: bit-level reachable,
                                  and any mis-mapped hyperparameter (lr/batch/steps/schedule)
                                  shows up here because both sides parsed the SAME argv.
  L2  per-step learning rate    -- the real bit-level gate for the MAPPING: warmup_fraction /
                                  min_lr / decay_style all live in the lr curve, and lr is computed
                                  from the schedule rather than accumulated through bf16 matmuls,
                                  so it can match exactly for all 50 steps.
  L3  per-step loss trajectory  -- NOT bit-level by construction: bf16 has ~3 decimal digits and the
                                  two sides run different optimizer/scheduler implementations, so
                                  round-off diverges with depth. Reported as an envelope, threshold
                                  pinned from observation (never guessed).

Usage (under torchrun --nproc_per_node=2):
    python _megatron_dense_runner.py --backend {dev_cli|legacy} --out RESULT.json \
        --out_dir DIR --steps 50 [--model ID] [--dataset SPEC ...]

Writes {"backend", "losses", "lrs"} to --out on rank 0.
"""
import argparse
import json
import os

# 0.5B by default: the judgement criteria are about the argv->Config mapping, not model scale, so
# the cheap model is used to validate them; --model swaps in the 7B dense.sh uses for confirmation.
DEFAULT_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
# Named so the comparison test can compute the epoch boundary from the SAME numbers this runner uses.
# The dev and legacy sides disagree on whether to train the trailing partial batch, so where the
# epoch ends decides where a step-for-step comparison stops being meaningful.
DEFAULT_DATASET_SIZE = 500
GLOBAL_BATCH_SIZE = 16
DEFAULT_DATASET = [f'AI-ModelScope/alpaca-gpt4-data-zh#{DEFAULT_DATASET_SIZE}']


def _dense_argv(model, datasets, out_dir, steps):
    """dense.sh's flags, with only the run-length knobs replaced.

    train_iters replaces dense.sh's num_train_epochs so both backends stop at exactly --steps
    (an epoch count would make the step budget depend on dataset size / packing and the two
    trajectories would have different lengths). Everything else is dense.sh verbatim.
    """
    argv = ['--model', model]
    argv += ['--dataset'] + list(datasets)
    argv += [
        '--save_safetensors',
        'true',
        '--merge_lora',
        'false',
        '--tuner_type',
        'lora',
        '--lora_rank',
        '8',
        '--lora_alpha',
        '32',
        '--target_modules',
        'all-linear',
        '--tensor_model_parallel_size',
        '2',
        '--sequence_parallel',
        'true',
        '--micro_batch_size',
        '16',
        '--global_batch_size',
        str(GLOBAL_BATCH_SIZE),
        '--recompute_granularity',
        'full',
        '--recompute_method',
        'uniform',
        '--recompute_num_layers',
        '1',
        '--finetune',
        'true',
        '--cross_entropy_loss_fusion',
        'true',
        '--lr',
        '1e-4',
        '--lr_warmup_fraction',
        '0.05',
        '--min_lr',
        '1e-5',
        '--train_iters',
        str(steps),
        '--output_dir',
        out_dir,
        '--add_version',
        'false',
        '--save_steps',
        '100000',
        '--logging_steps',
        '1',
        '--max_length',
        '2048',
        '--system',
        'You are a helpful assistant.',
        '--dataloader_num_workers',
        '4',
        '--no_save_optim',
        'true',
        '--no_save_rng',
        'true',
        '--dataset_num_proc',
        '4',
        '--model_author',
        'swift',
        '--model_name',
        'swift-robot',
        # Both sides iterate in the same order, so step-k sees the same samples on both.
        '--dataset_shuffle',
        'false',
        '--train_dataloader_shuffle',
        'false',
    ]
    return argv


def _run_dev_cli(model, datasets, out_dir, steps):
    """dev via swift.dev.cli.megatron.megatron_sft_main (the argv -> Config mapping under test).

    lr is read straight off the Megatron optimizer after each step: dev's SFTLoop history carries
    loss/grad_norm but no lr (twinkle exposes lr only through TrainMetric, which also emits
    iters/time/speed noise). Patching _record_step here keeps the lr probe in the test scaffold
    rather than changing dev's logging format for a test's benefit.

    Set DENSE_RUNNER_TRACE_SCHED=1 to also dump the scheduler's own state (num_steps / decay /
    warmup / min_lr) next to each sampled lr -- that is what distinguishes "the schedule is wrong"
    from "the probe samples at a different moment than legacy does".
    """
    from swift.dev.cli.megatron import megatron_sft_main as dev_megatron_sft_main
    from swift.dev.recipes.sft import SFTLoop

    trace = os.environ.get('DENSE_RUNNER_TRACE_SCHED') == '1'
    lrs, sched_states = [], []
    orig_record = SFTLoop._record_step

    def patched_record(self):
        orig_record(self)
        lrs.append(_read_lr_from_model(self.model))
        if trace:
            sched_states.append(_read_sched_state(self.model))

    SFTLoop._record_step = patched_record
    try:
        history = dev_megatron_sft_main(_dense_argv(model, datasets, out_dir, steps))
    finally:
        SFTLoop._record_step = orig_record
    if trace and int(os.environ.get('RANK', '0')) == 0:
        print(f'SCHED_STATES(first3)={sched_states[:3]}', flush=True)
        print(f'SCHED_STATES(last2)={sched_states[-2:]}', flush=True)
    return [r['loss'] for r in history], lrs


def _read_sched_state(model):
    """The scheduler's own counters PLUS every candidate lr read point, for attributing an lr
    mismatch to the schedule itself vs the place the probe samples from.

    Megatron's distributed optimizer is a ChainedOptimizer: the scheduler writes lr into the param
    groups it was constructed over, which are not necessarily the ones reachable from the outer
    wrapper's .param_groups. Printing all of them at once is what settles which read point is the
    one legacy's `param_group['lr']` corresponds to.
    """
    try:
        group = model.optimizer_group[model._get_default_group()]
        s, opt = group.lr_scheduler, group.optimizer
        if s is None:
            return None
        state = {
            'num_steps': s.num_steps,
            'decay': s.lr_decay_steps,
            'warmup': s.lr_warmup_steps,
            'min_lr': s.min_lr,
            'max_lr': s.max_lr
        }
        if opt is not None:
            state['sched_get_lr'] = round(float(s.get_lr(opt.param_groups[0])), 12)
            state['outer'] = sorted({round(float(pg['lr']), 12) for pg in opt.param_groups})
            # get_lr reads max_lr/min_lr OFF THE PARAM GROUP first (optimizer_param_scheduler.py
            # get_lr: param_group.get('max_lr', self.max_lr)), so a group carrying its own bounds
            # silently overrides the scheduler's -- that is the difference between decaying to
            # min_lr and decaying to 0.
            state['pg_bounds'] = [(pg.get('max_lr'), pg.get('min_lr'), pg.get('lr_mult'), pg.get('is_decoupled_lr'))
                                  for pg in opt.param_groups]
            chained = getattr(opt, 'chained_optimizers', None)
            if chained:
                inner = {round(float(pg['lr']), 12) for co in chained for pg in co.param_groups}
                state['inner'] = sorted(inner)
        return state
    except Exception as e:
        return f'ERR:{e}'


def _read_lr_from_model(model):
    """Current lr as the SCHEDULER sees it, i.e. computed from its own state.

    Deliberately NOT read off optimizer.param_groups: Megatron's distributed optimizer is a chain
    (ChainedOptimizer), and reading the outer wrapper's param_groups yields a value that lags/differs
    from what OptimizerParamScheduler just wrote into the inner optimizers -- observed as a curve that
    tracked the right shape but with slightly wrong values and a tail decaying past min_lr to 0, while
    the scheduler's own state (num_steps/decay/warmup/min_lr) was verifiably correct.

    get_lr(param_group) is the scheduler's own accessor, so this samples exactly the quantity legacy
    logs (legacy reads param_group['lr'] right after its own scheduler wrote it, base.py:696).
    """
    try:
        group = model.optimizer_group[model._get_default_group()]
        scheduler = group.lr_scheduler
        optimizer = group.optimizer
        if scheduler is None or optimizer is None:
            return None
        # get_lr takes the param_group (for its lr_mult / is_decoupled_lr handling).
        return float(scheduler.get_lr(optimizer.param_groups[0]))
    except Exception:
        return None


def _run_legacy(model, datasets, out_dir, steps):
    """legacy megatron_sft_main; loss/lr captured by patching BaseMegatronTrainer.on_log.

    legacy logs loss as [sum, count] (divide) and lr under 'learning_rate' (base.py:696, taken from
    param_group['lr']) -- the same quantity _read_lr_from_model pulls on the dev side.
    """
    losses, lrs = [], []
    from swift.megatron.trainers.base import BaseMegatronTrainer
    orig_on_log = BaseMegatronTrainer.on_log

    def patched_on_log(self, logs, prefix=''):
        v = logs.get('loss')
        if v is not None:
            losses.append(float(v[0] / v[1]) if hasattr(v, '__len__') and len(v) == 2 else float(v))
            lr = logs.get('learning_rate')
            lrs.append(float(lr) if lr is not None else None)
        return orig_on_log(self, logs, prefix)

    BaseMegatronTrainer.on_log = patched_on_log
    try:
        from swift.megatron import megatron_sft_main
        megatron_sft_main(_dense_argv(model, datasets, out_dir, steps))
    finally:
        BaseMegatronTrainer.on_log = orig_on_log
    return losses, lrs


def _cleanup_output_dir(out_dir: str) -> None:
    """Remove a finished run's output tree, best-effort.

    Deliberately narrow and non-fatal:
      - only removes a directory this runner was told to write to, and only if it exists;
      - swallows errors, because losing the cleanup is a disk-space problem while raising here would
        lose an expensive 50-step measurement that already succeeded.
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
    parser.add_argument('--backend', choices=['dev_cli', 'legacy'], required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--dataset', nargs='+', default=DEFAULT_DATASET)
    args = parser.parse_args()

    from modelscope import snapshot_download
    model_path = snapshot_download(args.model)

    if args.backend == 'dev_cli':
        losses, lrs = _run_dev_cli(model_path, args.dataset, args.out_dir, args.steps)
    else:
        losses, lrs = _run_legacy(model_path, args.dataset, args.out_dir, args.steps)

    if int(os.environ.get('RANK', '0')) == 0:
        with open(args.out, 'w') as f:
            json.dump({'backend': args.backend, 'losses': losses, 'lrs': lrs}, f)
        print(f'RUNNER_DONE backend={args.backend} n_loss={len(losses)} n_lr={len(lrs)}', flush=True)
        print(f'  losses[:5]={losses[:5]}', flush=True)
        print(f'  lrs[:5]={lrs[:5]}', flush=True)
        # Drop the checkpoint now that the numbers we came for are safely on disk. A 50-step run of
        # this config writes a multi-GB final checkpoint, and back-to-back comparison runs filled the
        # 30G /tmp twice -- once surfacing as CheckpointException('write') and once as
        # OSError(ENOSPC) from pytest's own stdout flush, neither of which points at the disk.
        # Deleting here rather than disabling the save keeps the save path itself under test (it is
        # part of what dense.sh exercises); only the leftover bytes go away. Ordered after the json
        # dump on purpose: a failure to clean up must never cost us the measurements.
        _cleanup_output_dir(args.out_dir)


if __name__ == '__main__':
    main()
