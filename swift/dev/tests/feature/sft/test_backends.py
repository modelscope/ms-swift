"""End-to-end SFT tests: the dev happy path and a legacy-vs-dev comparison.

Two things the component-level alignment tests do NOT cover:
  1. That the full assembly (CLI -> args_to_configs -> run_sft -> SFTLoop.fit -> save) runs
     to completion on real weights and produces a usable checkpoint. Every builder is unit-
     tested in isolation, but the wired-together green path was only ever hand-run.
  2. That the dev pipeline agrees with the *actual legacy pipeline* (swift sft_main /
     Seq2SeqTrainer), not just an HF shifted-CE reference. Same argv drives both.

Comparison precision (legacy uses HF Seq2SeqTrainer: internal shift + mean-reduction + HF
scheduler; dev uses twinkle SFTLoop: encode-time shift + sum-reduction + GA lagging one
micro-step):
  - step-1 loss is compared bit-exactly under bs=1/ga=1, where GA/scheduler differences are
    not yet in play, so the first update's loss is directly comparable.
  - subsequent steps only check that dev tracks legacy step-for-step within a small relative
    band (same natural-order samples) -- exact per-step equality is not expected once
    GA/scheduler + sum-vs-mean normalization compound as weights update.

All tests are @pytest.mark.slow (real 0.5B weights + GPU); run with -m slow.
"""
import os
import pytest

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
DATASET = 'AI-ModelScope/alpaca-gpt4-data-zh#20'

requires_gpu = pytest.mark.skipif(
    'CUDA_VISIBLE_DEVICES' not in os.environ and not os.environ.get('_FORCE_GPU'),
    reason='set CUDA_VISIBLE_DEVICES to run the e2e SFT tests')


def _build_sft_args(tmp_out, **overrides):
    """Minimal legacy SftArguments shared by both pipelines (deterministic, no version dir)."""
    from swift import SftArguments
    base = dict(
        model=MODEL,
        dataset=[DATASET],
        max_length=512,
        max_steps=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        lr_scheduler_type='constant',
        warmup_ratio=0.0,
        split_dataset_ratio=0.0,
        save_steps=100,
        logging_steps=1,
        seed=42,
        data_seed=42,
        # Disable BOTH shuffle knobs so legacy and dev iterate the dataset in the same natural
        # order and thus train on the same first sample (legacy's HF loader shuffles via
        # train_dataloader_shuffle=True by default; dev's loader via dataset_shuffle).
        dataset_shuffle=False,
        train_dataloader_shuffle=False,
        tuner_type='full',
        add_version=False,
        output_dir=str(tmp_out),
        report_to=['none'],
    )
    base.update(overrides)
    return SftArguments(**base)


# ----------------------------------------------------------------------
# Gap 1: dev happy path (CLI -> run_sft -> checkpoint) runs to completion
# ----------------------------------------------------------------------


@pytest.mark.slow
@requires_gpu
def test_dev_sft_happy_path_via_cli(tmp_path):
    """The full dev SFT assembly runs end-to-end and produces a self-describing checkpoint.

    Drives the CLI mapping (sft_main -> args_to_configs -> run_sft) so this also guards the
    argv surface, not just run_sft's Python signature.
    """
    import json

    from swift.dev.cli.sft import sft_main

    out = tmp_path / 'dev_out'
    history = sft_main(_build_sft_args(out, max_steps=3))

    # loss history: one record per optimizer step, readable (normalized) magnitude
    assert len(history) == 3, f'expected 3 optimizer steps, got {len(history)}'
    losses = [h['loss'] for h in history]
    # loss == loss is the NaN test (NaN != itself); not a typo.
    assert all(loss == loss for loss in losses), f'NaN in loss history: {losses}'
    assert all(0.0 < loss < 20.0 for loss in losses), f'loss not in normalized range: {losses}'
    assert all('grad_norm' in h for h in history), 'grad_norm missing from history'

    # final checkpoint + self-describing args.json (so `swift infer <ckpt>` needs no flags)
    ckpt = out / 'checkpoint-final'
    assert ckpt.is_dir(), f'no final checkpoint at {ckpt}'
    args_json = ckpt / 'args.json'
    assert args_json.is_file(), 'args.json not written'
    meta = json.loads(args_json.read_text())
    assert meta.get('model') == MODEL
    assert meta.get('template') == 'qwen2_5'
    assert meta.get('model_type'), 'model_type missing from args.json'


# ----------------------------------------------------------------------
# Gap 2: legacy-vs-dev comparison (same argv)
# ----------------------------------------------------------------------


def _legacy_step_losses(sft_args):
    """Run legacy sft_main and return the per-logging-step train losses in order."""
    from swift import sft_main as legacy_sft_main
    msg = legacy_sft_main(sft_args)
    losses = [rec['loss'] for rec in msg['log_history'] if 'loss' in rec]
    return losses


@pytest.mark.slow
@requires_gpu
def test_legacy_vs_dev_step1_loss_bit_close(tmp_path):
    """step-1 loss agrees between legacy Seq2SeqTrainer and dev SFTLoop under bs=1/ga=1.

    At the first update GA/scheduler differences are not yet in play, so the two pipelines'
    first-step losses must match closely (allowing only fp/backend + sum-vs-mean-normalization
    noise on a single sequence). This is the strong "dev == legacy" evidence.
    """
    from swift.dev.cli.sft import sft_main as dev_sft_main

    common = dict(
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        lr_scheduler_type='constant',
        warmup_ratio=0.0)

    legacy_losses = _legacy_step_losses(_build_sft_args(tmp_path / 'legacy', **common))
    dev_history = dev_sft_main(_build_sft_args(tmp_path / 'dev', **common))
    dev_losses = [h['loss'] for h in dev_history]

    assert legacy_losses, 'legacy produced no loss log'
    assert dev_losses, 'dev produced no loss'
    l0, d0 = legacy_losses[0], dev_losses[0]
    rel = abs(l0 - d0) / max(abs(l0), 1e-8)
    assert rel < 5e-3, f'step-1 loss mismatch: legacy={l0:.6f} dev={d0:.6f} rel={rel:.2e}'


@pytest.mark.slow
@requires_gpu
def test_legacy_vs_dev_multistep_trend(tmp_path):
    """Over several steps dev loss tracks legacy step-for-step (same natural-order samples).

    With shuffle disabled both pipelines see the same sample sequence, so the per-step loss
    SHAPE must match (each step's loss reflects that step's sample). Absolute equality is NOT
    expected past step 1 (GA lag + scheduler-shape + sum-vs-mean normalization diverge as
    weights update), so this asserts every step stays close in relative terms rather than
    demanding a monotone trend -- on tiny no-shuffle data legacy itself is non-monotone.
    """
    from swift.dev.cli.sft import sft_main as dev_sft_main

    common = dict(
        max_steps=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        lr_scheduler_type='constant',
        warmup_ratio=0.0,
        learning_rate=1e-4)

    legacy_losses = _legacy_step_losses(_build_sft_args(tmp_path / 'legacy', **common))
    dev_losses = [h['loss'] for h in dev_sft_main(_build_sft_args(tmp_path / 'dev', **common))]

    assert len(legacy_losses) >= 3 and len(dev_losses) >= 3
    n = min(len(legacy_losses), len(dev_losses))
    # Per-step tracking: dev follows legacy's loss curve within a small relative band. The band
    # widens after step 1 because weight updates diverge slightly (different normalization), but
    # step 1 (no update yet) must be tight and later steps must not drift far.
    for i in range(n):
        rel = abs(dev_losses[i] - legacy_losses[i]) / max(abs(legacy_losses[i]), 1e-8)
        limit = 5e-3 if i == 0 else 0.15
        assert rel < limit, (f'step {i} loss drift: legacy={legacy_losses[i]:.4f} '
                             f'dev={dev_losses[i]:.4f} rel={rel:.2e} (limit {limit})')
