"""Guards for the TrainConfig -> optimizer/scheduler forwarding, especially the Megatron branch.

The Megatron path fails in a specific, quiet way: Megatron reads weight decay and grad clipping
from its own knobs, which twinkle defaults on its own, so a value that is not forwarded produces a
silently different training run rather than an error. dev originally had no Megatron optimizer
surface at all and force-fitted the HF-shaped fields onto it, which is how weight_decay came to be
overwritten by twinkle's 0.01 on the first optimizer step. The vs-legacy loss test cannot see this
(0.1 vs 0.01 weight decay moves the loss far less than its tolerance band), so these tests assert
the forwarding itself.

Legacy swift keeps the two surfaces separate -- MegatronArguments owns clip_grad and the
weight_decay_incr_style/start_weight_decay/end_weight_decay triple, while max_grad_norm is an HF arg
the Megatron path never reads -- and dev now mirrors that split, so these tests also pin that
per-backend fields reach the right backend and are refused on the wrong one.
"""
import pytest


def _recording_megatron_model():
    """A dev MegatronModel subclass that records set_optimizer / set_lr_scheduler kwargs.

    Subclassed rather than mocked so ``_is_megatron_model``'s isinstance check picks the Megatron
    branch for real. Megatron's own __init__ is skipped: nothing here touches the model.
    """
    from swift.dev.model.megatron.model import MegatronModel

    class _Recorder(MegatronModel):

        def __init__(self):
            self.calls = {}

        def set_optimizer(self, optimizer_cls, **kwargs):
            self.calls['optimizer'] = (optimizer_cls, kwargs)

        def set_lr_scheduler(self, scheduler_cls, **kwargs):
            self.calls['scheduler'] = (scheduler_cls, kwargs)

    return _Recorder()


def _configure(**cfg_kwargs):
    """Run configure_optimizer against a recording Megatron model; return the recorded calls."""
    from swift.dev.config import TrainConfig
    from swift.dev.optimizer import configure_optimizer

    cfg = TrainConfig(**cfg_kwargs)
    model = _recording_megatron_model()
    configure_optimizer(model, cfg, num_training_steps=10)
    return model.calls


def test_megatron_forwards_weight_decay_clip_grad_and_decay_style():
    calls = _configure(
        learning_rate=3e-5, weight_decay=0.123, clip_grad=0.456, lr_scheduler_type='linear', warmup_ratio=0.1)

    _, opt_kwargs = calls['optimizer']
    assert opt_kwargs['weight_decay'] == 0.123
    # Megatron clips inside optimizer.step() from OptimizerConfig.clip_grad; twinkle's Megatron
    # clip_grad_and_step() ignores its max_grad_norm arg, so this is the only path that works.
    assert opt_kwargs['clip_grad'] == 0.456

    _, sched_kwargs = calls['scheduler']
    # The 'constant' style derives both ends from weight_decay (legacy's post-init rule).
    assert sched_kwargs['start_wd'] == 0.123
    assert sched_kwargs['end_wd'] == 0.123
    assert sched_kwargs['wd_incr_style'] == 'constant'
    assert sched_kwargs['lr_decay_style'] == 'linear'
    assert sched_kwargs['lr_decay_steps'] == 10
    assert sched_kwargs['lr_warmup_steps'] == 1
    assert sched_kwargs['max_lr'] == 3e-5
    # wd_incr_steps must NOT be forwarded: twinkle hardcodes it and does not pop it, so passing it
    # raises a duplicate-keyword TypeError.
    assert 'wd_incr_steps' not in sched_kwargs


def test_megatron_weight_decay_survives_a_scheduler_step():
    """The decisive guard: OptimizerParamScheduler.step() rewrites param_group['weight_decay']
    every step, so forwarding weight_decay to set_optimizer alone holds for zero steps.

    Composes the real pieces -- our kwargs, twinkle's scheduler defaults, Megatron's scheduler --
    so it stays honest if twinkle changes its defaults.
    """
    import torch

    from swift.dev.model.megatron.model import MegatronModel

    calls = _configure(learning_rate=3e-5, weight_decay=0.123, lr_scheduler_type='cosine')
    _, opt_kwargs = calls['optimizer']
    _, sched_kwargs = calls['scheduler']

    param = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW([param], lr=opt_kwargs['lr'], weight_decay=opt_kwargs['weight_decay'])
    # Unbound call: _create_megatron_scheduler never touches self, and going through twinkle means
    # every param we do NOT pass keeps twinkle's real default (that is where 0.01 came from).
    scheduler = MegatronModel._create_megatron_scheduler(None, optimizer, **sched_kwargs)

    scheduler.step(increment=1)
    assert optimizer.param_groups[0]['weight_decay'] == 0.123

    # Control: the same composition minus start_wd/end_wd is the bug this guards, so if twinkle's
    # default ever stops overwriting weight_decay this assertion fails and says so, instead of the
    # test above quietly becoming vacuous.
    bare = {k: v for k, v in sched_kwargs.items() if k not in ('start_wd', 'end_wd')}
    optimizer_bare = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))],
                                       lr=opt_kwargs['lr'],
                                       weight_decay=opt_kwargs['weight_decay'])
    MegatronModel._create_megatron_scheduler(None, optimizer_bare, **bare).step(increment=1)
    assert optimizer_bare.param_groups[0]['weight_decay'] == 0.01


def test_megatron_ramped_weight_decay_uses_both_ends():
    """A non-constant wd_incr_style must forward the explicit ends, not weight_decay."""
    calls = _configure(weight_decay=0.1, weight_decay_incr_style='linear', start_weight_decay=0.0, end_weight_decay=0.2)

    _, sched_kwargs = calls['scheduler']
    assert (sched_kwargs['start_wd'], sched_kwargs['end_wd']) == (0.0, 0.2)
    assert sched_kwargs['wd_incr_style'] == 'linear'


def _validate(**cfg_kwargs):
    """Run validate_configs on a minimal Megatron config, overriding TrainConfig fields."""
    from swift.dev.config import (
        DatasetConfig,
        DistributedConfig,
        ModelConfig,
        TemplateConfig,
        TrainConfig,
        validate_configs,
    )

    validate_configs(
        ModelConfig(model='m'), TemplateConfig(), DatasetConfig(), TrainConfig(**cfg_kwargs),
        DistributedConfig(backend='megatron', mode='local', nproc_per_node=1))


def test_inconsistent_weight_decay_schedule_fails_before_anything_is_built():
    """Both halves of legacy's post-init rule, enforced in validate_configs (millisecond failure).

    Left to configure_optimizer these would surface only after the model weights are loaded.
    """
    with pytest.raises(ValueError, match='NON-constant'):
        _validate(start_weight_decay=0.0)  # ends set while the style is 'constant'
    with pytest.raises(ValueError, match='BOTH'):
        _validate(weight_decay_incr_style='cosine')  # ramping style with no ends
    _validate(weight_decay_incr_style='cosine', start_weight_decay=0.0, end_weight_decay=0.1)


def test_grad_clipping_threshold_is_one_knob_on_both_backends():
    """max_grad_norm clips on BOTH backends; clip_grad is a deprecated alias of it.

    dev briefly split the two (max_grad_norm HF-only, clip_grad Megatron-only) to mirror legacy,
    where the Megatron path never read max_grad_norm. But that silent drop was the defect, not a
    contract worth preserving: the clipping threshold means the same thing in both, so setting it
    under either name must take effect rather than raise.
    """
    from swift.dev.config import (
        DatasetConfig,
        DistributedConfig,
        ModelConfig,
        TemplateConfig,
        TrainConfig,
        validate_configs,
    )
    from swift.dev.optimizer import resolve_max_grad_norm

    # Neither name is backend-restricted any more.
    _validate(max_grad_norm=0.5)
    _validate(clip_grad=0.5)
    for field in ('max_grad_norm', 'clip_grad'):
        validate_configs(
            ModelConfig(model='m'), TemplateConfig(), DatasetConfig(), TrainConfig(**{field: 0.5}),
            DistributedConfig(backend='hf', mode='local'))

    # Either name resolves to the same threshold, and max_grad_norm wins when both are given.
    assert resolve_max_grad_norm(TrainConfig(max_grad_norm=0.5)) == 0.5
    assert resolve_max_grad_norm(TrainConfig(clip_grad=0.5)) == 0.5
    assert resolve_max_grad_norm(TrainConfig(max_grad_norm=0.5, clip_grad=9.0)) == 0.5
    # Untouched config keeps the documented default.
    assert resolve_max_grad_norm(TrainConfig()) == 1.0


def test_megatron_recompute_flag_mismatch_warns_but_contradiction_fails(monkeypatch):
    """gradient_checkpointing is unread on Megatron, so the two directions are handled differently.

    (Lives here with the other backend-surface validate_configs tests.) The default True + no
    granularity is every current Megatron run, so it can only warn -- but it must warn, because the
    config claims recompute is on while the run does not recompute. An explicit False alongside a
    configured granularity is a contradiction the user typed, so it is fatal.

    Records are collected off the module logger rather than via caplog: swift's logging setup turns
    propagation off, so nothing reaches pytest's root handler once any test has imported it.
    """
    from swift.dev.config import (
        DatasetConfig,
        DistributedConfig,
        ModelConfig,
        TemplateConfig,
        TrainConfig,
        validate_configs,
    )
    from swift.dev.config import validate as validate_mod

    warnings: list = []
    monkeypatch.setattr(validate_mod.logger, 'warning', lambda msg, *a: warnings.append(msg))

    def check(train_kwargs, dist_kwargs):
        validate_configs(
            ModelConfig(model='m'), TemplateConfig(), DatasetConfig(), TrainConfig(**train_kwargs),
            DistributedConfig(backend='megatron', mode='local', nproc_per_node=1, **dist_kwargs))

    check({}, {})  # default: flag on, nothing configured
    assert len(warnings) == 1 and 'will NOT' in warnings[0]

    warnings.clear()
    check({'gradient_checkpointing': False}, {})  # both off: consistent, silent
    check({}, {'recompute_granularity': 'selective'})  # both on: consistent, silent
    assert warnings == []

    with pytest.raises(ValueError, match='contradicts'):
        check({'gradient_checkpointing': False}, {'recompute_granularity': 'selective'})


def test_hf_optimizer_name_is_refused_on_megatron():
    """optim names a torch/HF optimizer, which the Megatron path cannot build.

    configure_optimizer always asks twinkle for Megatron's own distributed optimizer, so before this
    check optim='sgd' trained with Adam and said nothing. Megatron's own non-Adam types (sgd, muon,
    dist_muon in legacy) are not wired either, so this is the message a user asking for one gets.
    The default is legal on both backends -- and it is legacy SftArguments' default too, so a
    Megatron run through the CLI carries it without tripping.
    """
    _validate(optim='adamw_torch_fused')
    for name in ('sgd', 'muon'):
        with pytest.raises(ValueError, match='optim'):
            _validate(optim=name)


def test_learning_rate_default_lives_on_the_config():
    """No hidden lr fallback in configure_optimizer: what TrainConfig reports is what trains.

    BREAK vs legacy, deliberate: legacy resolves lr from the tuner type (1e-5 full / 1e-4 otherwise)
    in both surfaces, so a LoRA run built through the programmatic API now needs an explicit
    learning_rate=1e-4. Keeping the tuner-aware rule would have meant resolving a TrainConfig field
    from TunerConfig, i.e. a config whose reported value is not the value used.
    """
    from swift.dev.config import TrainConfig

    assert TrainConfig().learning_rate == 1e-5
    _, sched_kwargs = _configure()['scheduler']
    assert sched_kwargs['max_lr'] == TrainConfig().learning_rate


def test_megatron_optimizer_fields_are_refused_on_the_hf_backend():
    """The split cuts both ways: Megatron-only knobs must not look accepted on transformers.

    start_weight_decay is checked at 0.0 on purpose -- ramping up from no decay is the ordinary use,
    and a falsy value must still count as "the user set this" (see _is_off).
    """
    from swift.dev.config import (
        DatasetConfig,
        DistributedConfig,
        ModelConfig,
        TemplateConfig,
        TrainConfig,
        validate_configs,
    )

    for field, value in (('weight_decay_incr_style', 'linear'), ('start_weight_decay', 0.0), ('end_weight_decay', 0.1)):
        with pytest.raises(ValueError, match=field):
            validate_configs(
                ModelConfig(model='m'), TemplateConfig(), DatasetConfig(), TrainConfig(**{field: value}),
                DistributedConfig(backend='hf', mode='local'))


@pytest.mark.parametrize('swift_name,specific', [
    ('cosine_with_min_lr', {
        'min_lr': 1e-6
    }),
    ('constant_with_warmup', {}),
    ('cosine_with_restarts', {}),
    ('polynomial', {}),
    ('inverse_sqrt', {}),
])
def test_hf_backed_schedules_reproduce_hf_lr_curve(swift_name, specific):
    """dev's adapter classes must trace exactly HF's lr curve -- the point is to REUSE HF's math.

    Compares step by step against transformers.get_scheduler, so a wrong lambda (the class of bug
    that made constant_with_warmup decay to zero) shows up as a diverging curve rather than as a
    plausible-looking run.
    """
    import torch

    from swift.dev.naming import resolve_scheduler
    from transformers import get_scheduler

    def curve(build):
        param = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.AdamW([param], lr=1e-4)
        sched = build(optimizer)
        seen = []
        for _ in range(12):
            seen.append(round(optimizer.param_groups[0]['lr'], 12))
            sched.step()
        return seen

    sched_cls = resolve_scheduler(swift_name)
    assert isinstance(sched_cls, type), 'must be a CLASS: twinkle cannot resolve dev names by string'

    ours = curve(lambda opt: sched_cls(opt, num_warmup_steps=3, num_training_steps=10, **specific))
    theirs = curve(lambda opt: get_scheduler(
        swift_name, opt, num_warmup_steps=3, num_training_steps=10, scheduler_specific_kwargs=specific or None))
    assert ours == theirs, f'{swift_name}: dev={ours} hf={theirs}'


def test_twinkle_can_construct_the_adapter_classes():
    """Pin the twinkle-side resolution, which is why these are classes and not names.

    Calls construct_class exactly as set_lr_scheduler does. A string would NOT work here: twinkle
    searches only [torch.optim.lr_scheduler, twinkle.module.scheduler], and dev's module is neither,
    so it would fall through to Plugin.load_plugin. Classes also survive the Ray boundary
    (set_lr_scheduler is a @remote_function), because a module-level class pickles by reference.
    """
    import torch
    import twinkle.module.scheduler
    from torch.optim.lr_scheduler import LRScheduler
    from twinkle.utils.loader import construct_class

    from swift.dev.naming import resolve_scheduler

    param = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW([param], lr=1e-4)
    sched = construct_class(
        resolve_scheduler('cosine_with_min_lr'),
        LRScheduler, [torch.optim.lr_scheduler, twinkle.module.scheduler],
        optimizer=optimizer,
        num_warmup_steps=2,
        num_training_steps=10,
        min_lr=1e-6)

    assert isinstance(sched, LRScheduler)
    with pytest.raises(Exception):  # noqa: B017 -- twinkle raises its own plugin error type
        construct_class(
            'CosineWithMinLRScheduler',
            LRScheduler, [torch.optim.lr_scheduler, twinkle.module.scheduler],
            optimizer=optimizer)


# === transformers-path warmup rounding (anchored against HF itself) ===


def _recording_hf_model():
    """A non-Megatron model recording set_optimizer / set_lr_scheduler, to exercise the HF branch.

    Plain object on purpose: configure_optimizer picks its branch via _is_megatron_model's isinstance
    check, so anything that is not a MegatronModel takes the transformers path for real.
    """

    class _Recorder:

        def __init__(self):
            self.calls = {}

        def set_optimizer(self, optimizer_cls, **kwargs):
            self.calls['optimizer'] = (optimizer_cls, kwargs)

        def set_lr_scheduler(self, scheduler_cls, **kwargs):
            self.calls['scheduler'] = (scheduler_cls, kwargs)

    return _Recorder()


def _hf_warmup_steps(warmup_ratio, num_training_steps):
    """The warmup step count dev actually hands to the HF scheduler."""
    from swift.dev.config import TrainConfig
    from swift.dev.optimizer import configure_optimizer

    model = _recording_hf_model()
    configure_optimizer(
        model,
        TrainConfig(learning_rate=1e-4, lr_scheduler_type='linear', warmup_ratio=warmup_ratio),
        num_training_steps=num_training_steps)
    return model.calls['scheduler'][1]['num_warmup_steps']


@pytest.mark.parametrize(
    'warmup_ratio, num_training_steps',
    [
        (0.0, 50),
        (0.02, 50),
        (0.05, 50),  # 2.5 -- the divergence point: ceil=3 but round()=2 (banker's rounding)
        (0.05, 100),
        (0.1, 33),  # 3.3 -- ceil=4, round=3 (ordinary round-down disagreement, not banker's)
        (0.1, 50),
        (0.5, 33),  # 16.5 -- ceil=17, round=16
        (0.075, 33),  # 2.475 -> ceil=3
        (0.5, 50),
    ])
def test_hf_warmup_steps_match_transformers_exactly(warmup_ratio, num_training_steps):
    """dev's warmup step count must equal TrainingArguments.get_warmup_steps, not merely resemble it.

    Anchored against HF's own implementation rather than a restated formula, so a future change on
    either side shows up here. HF computes math.ceil(num_training_steps * warmup_ratio)
    (training_args.py::get_warmup_steps) and its schedules treat that as an integer step boundary.

    Why this test exists at all: dev used int(round(...)), which disagrees with HF whenever the
    product lands on .5 (Python rounds 2.5 to 2) or anywhere below .5 of a step. warmup_ratio=0.05
    over 50 steps -- dense.sh's own setting -- gave dev 2 warmup steps against HF's 3. The bug was
    found from the Megatron side, but the fix touched this SHARED path, and the earlier
    'HF is bit-identical to legacy' results were all obtained with warmup_ratio=0.0
    (_megatron_sft_runner.py, test_sft_alignment.py), where every rounding rule agrees. That blind
    spot is what this parametrisation closes; 2.5 is included deliberately.
    """
    import warnings

    from transformers import TrainingArguments

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected = TrainingArguments(
            output_dir='/tmp/_warmup_probe', warmup_ratio=warmup_ratio,
            report_to=[]).get_warmup_steps(num_training_steps)
    actual = _hf_warmup_steps(warmup_ratio, num_training_steps)
    assert actual == expected, (
        f'warmup_ratio={warmup_ratio} over {num_training_steps} steps: dev passes {actual} warmup '
        f'steps but transformers computes {expected}')


def test_hf_warmup_rounding_is_ceil_not_round():
    """Pin the direction of the rounding, so 'round' cannot creep back in and still pass by luck.

    Separate from the parametrised comparison above because that one would also pass if BOTH dev and
    a future transformers switched to round(). This states the property we rely on.
    """
    import math

    exact = 0.05 * 50
    assert exact == 2.5
    assert math.ceil(exact) == 3 and int(round(exact)) == 2, \
        'the 2.5 divergence no longer exists -- re-check whether this guard is still meaningful'
    assert _hf_warmup_steps(0.05, 50) == 3, 'dev must round the warmup budget UP on the HF path'


def test_megatron_warmup_stays_fractional_while_hf_rounds():
    """The same ratio must reach the two backends differently -- fractional vs integer.

    Megatron's OptimizerParamScheduler interpolates the ramp on lr_warmup_steps, so 2.5 there means
    two and a half steps of warmup (and legacy megatron likewise keeps it fractional). HF's schedules
    compare the step index against an integer boundary, where 2.5 is meaningless. Asserting both in
    one place keeps a future 'simplification' from unifying them.
    """
    megatron_calls = _configure(learning_rate=1e-4, lr_scheduler_type='linear', warmup_ratio=0.05)
    _, sched_kwargs = megatron_calls['scheduler']
    megatron_warmup = sched_kwargs['lr_warmup_steps']
    assert megatron_warmup == pytest.approx(0.05 * 10), \
        f'Megatron warmup must stay fractional, got {megatron_warmup!r}'

    assert _hf_warmup_steps(0.05, 10) == 1, 'HF warmup must be an integer (ceil of 0.5)'


@pytest.mark.parametrize(
    'sched_type, warmup_ratio',
    [
        ('linear', 0.05),
        ('linear', 0.1),
        ('cosine', 0.05),  # 0.05*50 = 2.5 -- the rounding divergence point
        ('cosine', 0.1),
        ('constant_with_warmup', 0.05),
        ('polynomial', 0.05),
    ])
def test_hf_lr_curve_matches_transformers_with_nonzero_warmup(sched_type, warmup_ratio):
    """The whole lr CURVE (not just the step count) must match transformers under real warmup.

    The step-count comparison above only checks what dev passes in; this drives both schedulers and
    compares every lr. Needed because dev does not reuse HF's get_scheduler -- 'cosine'/'linear' go to
    twinkle's own warmup schedulers and the rest to dev/scheduler.py adapters -- so equality of the
    curve is an assumption until asserted.

    Non-zero warmup_ratio on purpose: every previously reported 'HF is bit-identical to legacy' result
    used warmup_ratio=0.0 (the Megatron runner hardcodes it, and the HF alignment test is
    forward-only and never reaches a scheduler), so the entire warmup ramp was untested on this path.
    """
    import warnings

    import torch

    from swift.dev.naming import resolve_scheduler
    from transformers import TrainingArguments, get_scheduler

    # resolve_scheduler returns a CLASS for dev's own adapters but a twinkle NAME for cosine/linear
    # (twinkle resolves those against its own module). Resolve the name the same way twinkle does, so
    # this drives whatever the real run would drive -- including twinkle's schedulers, whose
    # equivalence to HF was itself only an assumption before this test.
    resolved = resolve_scheduler(sched_type)
    if isinstance(resolved, str):
        import twinkle.module.scheduler as twinkle_scheduler
        resolved = getattr(twinkle_scheduler, resolved)

    num_training_steps = 50
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        num_warmup_steps = TrainingArguments(
            output_dir='/tmp/_warmup_probe', warmup_ratio=warmup_ratio,
            report_to=[]).get_warmup_steps(num_training_steps)

    def _drive(scheduler, optimizer):
        out = []
        for _ in range(num_training_steps):
            out.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        return out

    lr = 1e-4
    dev_param = torch.nn.Parameter(torch.zeros(1))
    dev_opt = torch.optim.AdamW([dev_param], lr=lr)
    dev_curve = _drive(
        resolved(dev_opt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps), dev_opt)

    hf_param = torch.nn.Parameter(torch.zeros(1))
    hf_opt = torch.optim.AdamW([hf_param], lr=lr)
    hf_curve = _drive(
        get_scheduler(sched_type, hf_opt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps),
        hf_opt)

    mismatches = [(i, dev_curve[i], hf_curve[i]) for i in range(num_training_steps)
                  if dev_curve[i] != pytest.approx(hf_curve[i], rel=1e-12, abs=1e-15)]
    assert not mismatches, (f'{sched_type} @ warmup_ratio={warmup_ratio}: {len(mismatches)}/{num_training_steps} lr '
                            f'values differ from transformers; first 3: {mismatches[:3]}')
    # The ramp must actually be exercised, or this would pass vacuously on a flat curve. With
    # num_warmup_steps>=2 the first lr is strictly below the last warmup lr; at exactly 1 the ramp is
    # a single point, so only the sub-peak start is checkable.
    assert num_warmup_steps >= 1, f'no warmup at all (ratio={warmup_ratio})'
    if num_warmup_steps >= 2:
        assert dev_curve[0] < dev_curve[num_warmup_steps - 1], \
            f'no warmup ramp: curve[:{num_warmup_steps}]={dev_curve[:num_warmup_steps]}'
    assert dev_curve[0] < lr, f'warmup must start below the peak lr, got {dev_curve[0]}'


def test_constant_with_warmup_holds_instead_of_decaying():
    """The regression that motivated the adapter, asserted on the curve itself.

    The old mapping (twinkle's LinearWarmupScheduler) decayed linearly to 0 after warmup, so a user
    asking for a constant lr silently got an annealed one.
    """
    import torch

    from swift.dev.naming import resolve_scheduler

    param = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW([param], lr=1e-4)
    sched = resolve_scheduler('constant_with_warmup')(optimizer, num_warmup_steps=2, num_training_steps=10)
    lrs = []
    for _ in range(8):
        lrs.append(optimizer.param_groups[0]['lr'])
        sched.step()

    assert lrs[0] < lrs[2], 'no warmup ramp'
    assert lrs[2:] == [pytest.approx(1e-4)] * len(lrs[2:]), f'lr did not stay constant: {lrs}'


def test_megatron_cosine_with_min_lr_needs_no_extra_schedule():
    """On Megatron the floor is a first-class scheduler arg, so 'cosine' + min_lr is the whole story."""
    calls = _configure(learning_rate=3e-5, lr_scheduler_type='cosine_with_min_lr', min_lr=1e-6)

    _, sched_kwargs = calls['scheduler']
    assert sched_kwargs['lr_decay_style'] == 'cosine'
    assert sched_kwargs['min_lr'] == 1e-6


def test_megatron_cosine_with_min_lr_requires_a_floor():
    """Accepting the name without min_lr would run plain cosine -- the name silently not honoured."""
    with pytest.raises(ValueError, match='needs TrainConfig.min_lr'):
        _validate(lr_scheduler_type='cosine_with_min_lr')
    _validate(lr_scheduler_type='cosine_with_min_lr', min_lr=1e-6)


def test_scheduler_kwargs_accepts_cli_json_and_rejects_garbage():
    """The CLI can only deliver a string, so JSON must be parsed here rather than forwarded."""
    from swift.dev.naming import parse_scheduler_kwargs

    assert parse_scheduler_kwargs('{"min_lr": 1e-6}') == {'min_lr': 1e-6}
    assert parse_scheduler_kwargs({'min_lr': 2e-6}) == {'min_lr': 2e-6}
    assert parse_scheduler_kwargs(None) == {}
    with pytest.raises(ValueError, match='not valid JSON'):
        parse_scheduler_kwargs('min_lr=1e-6')
    with pytest.raises(ValueError, match='must be a JSON object'):
        parse_scheduler_kwargs('[1, 2]')


def test_unsupported_lr_scheduler_type_fails_fast_on_both_backends():
    """A name neither backend can run must raise, not silently decay as cosine.

    'reduce_lr_on_plateau' is a real HF SchedulerType (so legacy swift accepts it, since it forwards
    lr_scheduler_type straight to HF Trainer) that neither backend can drive here: it steps on a
    metric rather than a step count, which is not what configure_optimizer wires up. That makes it
    the right shape of config to fail loudly rather than train a different schedule.
    """
    from swift.dev.naming import resolve_megatron_decay_style, resolve_scheduler

    with pytest.raises(NotImplementedError, match='Megatron'):
        resolve_megatron_decay_style('reduce_lr_on_plateau')
    with pytest.raises(NotImplementedError, match='Transformers'):
        resolve_scheduler('reduce_lr_on_plateau')
    with pytest.raises(NotImplementedError):
        _configure(lr_scheduler_type='reduce_lr_on_plateau')


def test_backends_may_support_different_scheduler_sets():
    """The two maps are NOT required to agree, and the error must say which backend refused.

    An equal-key-sets invariant would cap dev at the intersection: the Transformers path can run any
    LambdaLR-shaped HF schedule (see dev/scheduler.py), while Megatron accepts only its own
    OptimizerParamScheduler's decay styles. 'polynomial' is the asymmetry -- and the message has to
    name the refusing backend, or "not supported" reads like a dev-wide limitation.
    """
    from swift.dev.naming import resolve_megatron_decay_style, resolve_scheduler

    assert resolve_scheduler('polynomial') is not None
    with pytest.raises(NotImplementedError, match='not supported on the Megatron backend'):
        resolve_megatron_decay_style('polynomial')


def test_megatron_decay_styles_are_values_megatron_accepts():
    """Every mapped value must be a legal Megatron lr_decay_style.

    Mirrors the Literal legacy swift accepts in MegatronArguments.lr_decay_style; a typo here
    (e.g. HF's 'inverse_sqrt' spelling instead of Megatron's 'inverse-square-root') would only
    surface deep inside OptimizerParamScheduler.get_lr as a silently skipped decay branch.
    """
    from swift.dev.naming import _MEGATRON_DECAY_STYLE_MAP, resolve_megatron_decay_style

    legal = {'constant', 'linear', 'cosine', 'inverse-square-root', 'WSD'}
    assert set(_MEGATRON_DECAY_STYLE_MAP.values()) <= legal
    assert resolve_megatron_decay_style('inverse_sqrt') == 'inverse-square-root'
