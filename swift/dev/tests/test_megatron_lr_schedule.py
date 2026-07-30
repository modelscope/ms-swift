"""The lr SCHEDULE dev builds must match legacy megatron's bit-for-bit, over a full run.

Why this is a fast (CPU-only) test even though it gates a training behaviour: the lr curve is a pure
function of the scheduler's construction arguments, so it can be replayed against
OptimizerParamScheduler directly -- no model, no GPU, no bf16 noise. That makes it the ONE bit-level
gate available for the dev-vs-legacy Megatron comparison; the loss trajectory cannot be bit-level in
bf16 (see test_run_sft_e2e.py::test_megatron_cli_vs_legacy_loss for the envelope version).

The two sides express the SAME schedule in different units, which is exactly what makes this worth
pinning:
  legacy (megatron_lm_utils.get_optimizer_param_scheduler:574-581) counts SAMPLES --
      lr_decay_steps  = lr_decay_iters * global_batch_size
      lr_warmup_steps = lr_warmup_fraction * lr_decay_steps      (kept fractional, never rounded)
      and steps the scheduler by global_batch_size per optimizer step.
  dev (swift/dev/optimizer.py) counts OPTIMIZER STEPS --
      lr_decay_steps  = num_training_steps
      lr_warmup_steps = warmup_ratio * num_training_steps
      and twinkle's lr_step() advances by increment=1.
Same curve, different unit. Two real bugs were found by comparing them at this resolution, and both
are pinned below: the warmup budget used to be int(round(...))-ed, and min_lr used to reach only the
scheduler and not the optimizer (mcore's get_lr prefers the param group's own bound).
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip('megatron.core', reason='needs megatron-core for OptimizerParamScheduler')

from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler  # noqa: E402

from swift.dev.configs import TrainConfig  # noqa: E402
from swift.dev.naming import resolve_megatron_decay_style  # noqa: E402
from swift.dev.optimizer import megatron_weight_decay_bounds  # noqa: E402

# dense.sh's schedule knobs (examples/megatron/lora/dense.sh).
LR = 1e-4
MIN_LR = 1e-5
WARMUP_FRACTION = 0.05
GLOBAL_BATCH_SIZE = 16
STEPS = 50


class _FakeOptimizer:
    """Minimal stand-in: OptimizerParamScheduler only touches param_groups.

    min_lr/max_lr are carried ON THE GROUP because that is what mcore itself does
    (optimizer/__init__.py:399 writes {'min_lr': OptimizerConfig.min_lr}) and what get_lr reads
    first -- omitting them here would hide the very interaction this file pins.
    """

    def __init__(self, *, min_lr: float, max_lr: float):
        param = torch.nn.Parameter(torch.zeros(1))
        self.param_groups = [{
            'params': [param],
            'lr': 0.0,
            'weight_decay': 0.0,
            'min_lr': min_lr,
            'max_lr': max_lr,
            'is_decoupled_lr': False,
            'wd_mult': 1.0,
            'lr_mult': 1.0,
        }]


def _curve(*, decay_steps, warmup_steps, increment, min_lr, n=STEPS, init_extra_step=False):
    """Run a schedule and return the per-step lr the trainer would see."""
    optimizer = _FakeOptimizer(min_lr=min_lr, max_lr=LR)
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=LR,
        min_lr=min_lr,
        lr_warmup_steps=warmup_steps,
        lr_decay_steps=decay_steps,
        lr_decay_style='cosine',
        start_wd=0.1,
        end_wd=0.1,
        wd_incr_steps=decay_steps,
        wd_incr_style='constant')
    if init_extra_step:
        # twinkle's first lr_step() lands with increment=0 (observed in the real run); harmless, but
        # replayed so this mirrors dev's actual call sequence rather than an idealized one.
        scheduler.step(increment=0)
    out = []
    for _ in range(n):
        scheduler.step(increment=increment)
        out.append(optimizer.param_groups[0]['lr'])
    return out


def _legacy_curve():
    """legacy's schedule, in its own SAMPLES unit."""
    return _curve(
        decay_steps=STEPS * GLOBAL_BATCH_SIZE,
        warmup_steps=WARMUP_FRACTION * STEPS * GLOBAL_BATCH_SIZE,
        increment=GLOBAL_BATCH_SIZE,
        min_lr=MIN_LR)


def _dev_curve(cfg: TrainConfig, *, num_training_steps=STEPS):
    """dev's schedule, derived from a TrainConfig exactly as configure_optimizer does."""
    warmup_steps_exact = cfg.warmup_ratio * num_training_steps
    return _curve(
        decay_steps=max(1, num_training_steps),
        warmup_steps=warmup_steps_exact,
        increment=1,
        min_lr=cfg.min_lr,
        init_extra_step=True)


def _dense_cfg(**overrides) -> TrainConfig:
    kwargs = dict(
        learning_rate=LR,
        min_lr=MIN_LR,
        warmup_ratio=WARMUP_FRACTION,
        lr_scheduler_type='cosine',
        weight_decay=0.1,
        weight_decay_incr_style='constant',
        clip_grad=1.0)
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


def test_dev_lr_curve_is_bit_identical_to_legacy_over_50_steps():
    """THE bit-level gate: every one of the 50 lr values must match exactly.

    Not a tolerance comparison on purpose -- lr comes out of a closed-form schedule, so anything but
    equality means the two sides are running different schedules, however slightly.
    """
    dev = _dev_curve(_dense_cfg())
    legacy = _legacy_curve()
    mismatches = [(i, dev[i], legacy[i]) for i in range(STEPS) if dev[i] != legacy[i]]
    assert not mismatches, (f'{len(mismatches)}/{STEPS} lr values differ from legacy; first 3: {mismatches[:3]}')


def test_warmup_budget_is_not_rounded():
    """Regression: warmup_ratio*steps is fractional (0.05*50 = 2.5) and must stay so.

    int(round(2.5)) == 2 shortened the ramp, which made every warmup lr come out 1.25x (2.5/2) too
    high and moved the peak one step earlier -- the whole curve, not just the warmup segment, since
    the decay phase starts from a different point.
    """
    legacy = _legacy_curve()
    rounded = _curve(
        decay_steps=STEPS,
        warmup_steps=int(round(WARMUP_FRACTION * STEPS)),
        increment=1,
        min_lr=MIN_LR,
        init_extra_step=True)
    assert rounded != legacy, 'rounding no longer changes the curve -- this guard is vacuous'
    assert rounded[0] != legacy[0]
    # the 1.25x signature of losing half a warmup step
    assert rounded[0] == pytest.approx(legacy[0] * 1.25, rel=1e-9)


def test_min_lr_must_reach_the_param_group_not_only_the_scheduler():
    """Regression: mcore's get_lr prefers the param group's own min_lr.

    OptimizerParamScheduler.get_lr does param_group.get('min_lr', self.min_lr), and mcore writes
    OptimizerConfig.min_lr into every group (optimizer/__init__.py:399). Passing min_lr only to
    set_lr_scheduler therefore left the group's 0.0 in charge: cosine decayed straight past the floor
    to 0. configure_optimizer now also passes min_lr to set_optimizer.
    """
    correct = _dev_curve(_dense_cfg())
    assert correct[-1] == pytest.approx(MIN_LR, rel=1e-12), 'schedule must land on min_lr'

    # Same schedule, but the param group says min_lr=0 (the pre-fix state).
    optimizer = _FakeOptimizer(min_lr=0.0, max_lr=LR)
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=LR,
        min_lr=MIN_LR,
        lr_warmup_steps=WARMUP_FRACTION * STEPS,
        lr_decay_steps=STEPS,
        lr_decay_style='cosine',
        start_wd=0.1,
        end_wd=0.1,
        wd_incr_steps=STEPS,
        wd_incr_style='constant')
    scheduler.step(increment=0)
    for _ in range(STEPS):
        scheduler.step(increment=1)
    assert optimizer.param_groups[0]['lr'] == 0.0, (
        'the group-level min_lr no longer wins -- if mcore changed this, the fix in '
        'configure_optimizer (passing min_lr to set_optimizer) may be redundant')


def test_configure_optimizer_passes_min_lr_and_fractional_warmup():
    """Pin the two values at the configure_optimizer boundary, so the fixes cannot silently regress.

    Uses a recording stub instead of a real model: what matters is WHICH kwargs reach twinkle.
    """
    from swift.dev.optimizer import configure_optimizer

    calls = {}

    class _StubMegatronModel:

        def set_optimizer(self, name, **kwargs):
            calls['optimizer'] = (name, kwargs)

        def set_lr_scheduler(self, name, **kwargs):
            calls['scheduler'] = (name, kwargs)

    import swift.dev.optimizer as optimizer_module
    original = optimizer_module._is_megatron_model
    optimizer_module._is_megatron_model = lambda _model: True
    try:
        configure_optimizer(_StubMegatronModel(), _dense_cfg(), num_training_steps=STEPS)
    finally:
        optimizer_module._is_megatron_model = original

    assert calls['optimizer'][1]['min_lr'] == MIN_LR, 'min_lr must reach the OPTIMIZER'
    assert calls['scheduler'][1]['min_lr'] == MIN_LR, 'min_lr must reach the scheduler too'
    warmup = calls['scheduler'][1]['lr_warmup_steps']
    assert warmup == WARMUP_FRACTION * STEPS == 2.5, f'warmup must stay fractional, got {warmup!r}'
    assert not isinstance(warmup, int), 'warmup must not be rounded to an int'


def test_decay_style_and_wd_bounds_match_the_config():
    """The remaining schedule knobs the Megatron branch forwards."""
    cfg = _dense_cfg()
    assert resolve_megatron_decay_style(cfg.lr_scheduler_type) == 'cosine'
    assert megatron_weight_decay_bounds(cfg) == (cfg.weight_decay, cfg.weight_decay)


@pytest.mark.parametrize('ratio', [0.0, 0.02, 0.05, 0.1, 0.5])
def test_curve_matches_legacy_across_warmup_ratios(ratio):
    """The unit conversion must hold for any warmup_ratio, not just dense.sh's 0.05.

    Guards against a fix that happens to work for one fraction (e.g. one that rounds to a value that
    coincidentally matches at 0.05).
    """
    dev = _dev_curve(_dense_cfg(warmup_ratio=ratio))
    legacy = _curve(
        decay_steps=STEPS * GLOBAL_BATCH_SIZE,
        warmup_steps=ratio * STEPS * GLOBAL_BATCH_SIZE,
        increment=GLOBAL_BATCH_SIZE,
        min_lr=MIN_LR)
    assert dev == legacy, f'warmup_ratio={ratio}: curves diverge'
