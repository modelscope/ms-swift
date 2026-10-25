# Copyright (c) ModelScope Contributors. All rights reserved.
"""Async dimension (RL_PLAN §四/⑤): how far the rollout may run AHEAD of training -- the staleness ladder
that turns the synchronous loop into an overlapped one -- and the invariant that BOUNDS it.

The overlapped regimes ride a per-sample ``StreamingDriver`` and differ only in ``max_staleness``
(``async_mode='one_step_off'`` -> 1, ``'fully_async'`` -> the configured value). At staleness > 0 the sampler
runs ahead on its OWN GPUs (``sampler_mode='disaggregated'`` is mandatory there -- a colocated sampler
time-shares one DeviceGroup and cannot overlap), a publish overwrites the live weights under in-flight
generations (``in_place`` + ``allow_partial_rollout`` aborts and resumes each), and a sample may be trained up
to ``max_staleness`` versions after it was generated; the driver counts each publish (``stream_publishes``)
and stamps each sample's version skew (``version_span_mean``). The failure this dimension kills is the
staleness bound being violated -- a slow trajectory that spans several publishes being consumed at a lag
GREATER than ``max_staleness`` (the in-flight staleness gate missing, so only the ready buffer is scanned), or
a version pin released early.

A COLOCATE run is the synchronous 0-rung, but on a DIFFERENT code path: a colocated sampler cannot overlap
with the trainer (one device, time-shared), so ``run_grpo`` runs it as a driver-less plain loop
(``_colocate_loop`` -- wake the sampler, ``sync_weights(merge_and_sync=True)``, generate, hand the device back,
train, every window). There is no StreamingDriver, hence no ``stream_publishes`` / ``version_span_mean`` /
off-policy telemetry at all; the run is strictly on-policy by construction (generation and training alternate,
never concurrent), so there is no version skew to measure or bound.

PLAN table -- every async corner case, mapped to its owner (the two fail-loudly deployment gates are pure
config rejections already anchored in ``test_infeasible_combos.py`` and are cross-referenced, not rewritten,
per RL_PLAN §二 "只补缺口，不重写"):

| # | case | input | expected (contract) | where covered | failure it kills |
|---|------|-------|---------------------|---------------|------------------|
| 1 | staleness=0 synchronous colocate (driver-less) | ``async_mode='none'`` colocate GRPO | trains strictly on-policy; resolved regime is colocate + in_place; the StreamingDriver-only telemetry (``stream_publishes`` / ``version_span_mean`` / ``off_policy_consumed`` / ``partial_ratio``) is ABSENT (no driver on the plain loop) | **THIS FILE** ``test_sync_colocate_is_driverless_and_on_policy`` | colocate silently put back on the driver / a version skew creeping into a regime that alternates generation and training, or a downgrade that trivially lacks the telemetry |
| 2 | one_step_off bounded overlap (staleness 1) | ``async_mode='one_step_off'`` disaggregated GRPO, in_place + allow_partial_rollout + IS mode | trains on disjoint GPUs; ``version_span_mean <= 1`` and ``max_partial_span <= 1`` on EVERY step; policy published at least once | **THIS FILE** ``test_one_step_off_disaggregated_staleness_bounded`` | straggler consumed at lag > max_staleness (in-flight staleness gate missing) / version pin released early |
| 3 | fully_async bounded overlap (staleness 2) | ``async_mode='fully_async'`` ``max_staleness=2`` disaggregated GRPO, in_place + partial + IS mode | trains; ``version_span_mean <= 2`` and ``max_partial_span <= 2`` on EVERY step | **THIS FILE** ``test_fully_async_disaggregated_staleness_bounded`` | the deep-buffer staleness bound enforced only at depth 1 (a straggler consumed at lag 3) |
| 4 | colocate x overlap refused | ``async_mode='one_step_off'`` + ``sampler_mode='colocate'`` | ``ValueError`` ("needs the sampler on its own GPUs") | ``test_infeasible_combos.py::test_colocate_async_overlap_rejected`` | a single-card ColocateHandover that cannot overlap silently degrading instead of refusing |
| 5 | async GRPO without off-policy correction refused | ``async_mode='one_step_off'``, no ``rollout_importance_sampling_mode`` | ``ValueError`` (requires the IS mode) | ``test_infeasible_combos.py::test_async_grpo_requires_importance_sampling`` | staleness > 0 trains GRPO on raw stale tokens with no importance-sampling correction, silently wrong |

Why rows 1-3 are the e2e gaps here: rows 4/5 are pure config rejections (no run to drive) already owned by
the infeasible-combination suite. The staleness LADDER itself -- 0 (synchronous colocate) / 1 (one_step_off) /
2 (fully_async) -- is what this dimension adds. Rows 2/3 drive the REAL overlapping streaming driver (partial
rollout abort+resume, in-place weight overwrite under in-flight generation, an NCCL publish per window) end to
end and pin the version-span bound to the configured ``max_staleness``. Row 1 is the 0-rung on the OTHER code
path: a colocated sampler cannot overlap, so ``run_grpo`` runs it driver-less (``_colocate_loop``), and the
strictly-on-policy guarantee is structural rather than a measured span -- the row pins that regime (it trains,
it resolves to colocate + in_place, and it emits NO driver telemetry), which is the contrast that makes rows
2/3's ``span <= max_staleness`` bound meaningful.

Honest oracle note (testcase-planning step 4/8): whether ``version_span_mean`` actually RISES above 0 within
a 3-window tiny run is timing-dependent (it needs a generation slow enough to be consumed a version later),
so rows 2/3 assert the CEILING -- ``span <= max_staleness`` on every step -- which is exactly the invariant
the staleness gate guarantees, NOT a forced nonzero observation (that would be flaky). The bound is the
adversarial assertion: remove the in-flight staleness gate and a straggler is consumed at lag > max_staleness
-> the bound goes red. ``in_place`` is mandatory for rows 2/3 to be meaningful: under ``adapter_snapshot``
the driver reports ``version_span`` as a constant 0 by design, so the bound would be vacuous.

Reverse-verification: row 1 goes red if colocate is put back on the driver (any of ``stream_publishes`` /
``version_span_mean`` / ``off_policy_consumed`` / ``partial_ratio`` shows up in the history), if the resolved
regime is not the synchronous colocate one, or if the run does not train on-policy (NaN loss / zero ``lora_B``).
Rows 2/3 go red if the staleness gate is removed (a straggler consumed at lag > max_staleness -> bound
violated), if the overlapping regime silently downgrades to sync/colocate (``_assert_overlapping_regime``), or
if the in_place + partial-rollout machinery corrupts training (NaN loss / zero ``lora_B`` -> ``assert_rl_trained``
red). Run with ``CUDA_VISIBLE_DEVICES=<cards> pytest swift/dev/tests/feature/rl/test_async_e2e.py -m slow``.
"""
import pytest

from swift.dev.tests.feature.rl.conftest import rl_configs, run_rl
from twinkle.data_format import last_assistant_text


@pytest.mark.slow
@pytest.mark.accel(1)
def test_sync_colocate_is_driverless_and_on_policy(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """Row 1: a colocate ``async_mode='none'`` run is the synchronous 0-rung -- driver-less and on-policy.

    A colocated sampler time-shares ONE device with the trainer, so generation and training cannot overlap;
    ``run_grpo`` therefore runs it as a driver-less plain loop (``_colocate_loop``: wake the sampler,
    ``sync_weights(merge_and_sync=True)``, generate, hand the device back, train -- every window, with no
    StreamingDriver). The strictly-on-policy guarantee is STRUCTURAL (the two phases alternate, never run
    concurrently), not a measured version span, so this row pins three things rather than a numeric zero:

    * it really trains on-policy -- finite normalized loss + a non-zero ``lora_B`` read back from disk, with
      the GRPO policy-gradient fingerprint (``train/approx_kl`` / ``train/clip_ratio``) present, so it rolled
      out and took a real policy-gradient step rather than degrading to a non-rollout loop;
    * the resolved config is the synchronous colocate regime (``sampler_mode='colocate'``, ``async_mode='none'``,
      ``weight_sync_strategy`` derived to ``in_place``), so the run cannot have been silently downgraded to
      something that trivially lacks the telemetry;
    * the StreamingDriver-only telemetry rows 2/3 read (``stream_publishes`` / ``version_span_mean`` /
      ``off_policy_consumed`` / ``partial_ratio``) is ABSENT -- there is no driver on the colocate path to emit
      it. Asserting the absence pins the "no driver, no version face" design: a regression that put colocate
      back on the driver would surface here, and so would a version skew creeping into a synchronous run.

    The geometry (``num_generations=2``, ``max_steps=6``) spans three windows, so the on-policy training and the
    telemetry absence are checked across multiple sync boundaries rather than a single step. This is the 0-rung
    contrast that gives rows 2/3's ``span <= max_staleness`` bound its meaning.
    """
    configs = rl_configs(
        rl_type='grpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=6),
        out_dir=str(tmp_path / 'out_async_none'),
        nproc=1,
        max_steps=6,
        rlhf_over={'num_generations': 2}, reward_over={'orm': [_varied_content_reward]},
        rollout_over={'sampler_mode': 'colocate'},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)

    rc = configs['rollout_config']
    assert rc.sampler_mode == 'colocate', f'expected the synchronous colocate regime, got {rc.sampler_mode!r}'
    assert rc.async_mode == 'none', f'expected async_mode none (staleness 0), got {rc.async_mode!r}'
    assert rc.weight_sync_strategy == 'in_place', (
        f'colocate derives in_place (the only publication whose sync_fn hands the device back each window), '
        f'got {rc.weight_sync_strategy!r}')

    assert_rl_trained(
        history, configs, 'grpo', max_loss=20.0, require_keys=('train/approx_kl', 'train/clip_ratio'))

    for key in ('stream_publishes', 'version_span_mean', 'off_policy_consumed', 'partial_ratio'):
        assert all(key not in r for r in history), (
            f'colocate rides the driver-less plain loop, so the StreamingDriver-only {key!r} must be ABSENT '
            f'(its presence would mean colocate was put back on the driver, or a version skew leaked into a '
            f'strictly synchronous run): {[r.get(key) for r in history]}')


@pytest.mark.slow
@pytest.mark.accel(2)
def test_one_step_off_disaggregated_staleness_bounded(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """Row 2: ``one_step_off`` overlaps one window on a disaggregated sampler and BOUNDS the version span to 1.

    ``async_mode='one_step_off'`` pins ``max_staleness=1``: the driver trains version ``v`` while admitting
    ``v+1``. ``validate`` forces the overlapping regime onto ``sampler_mode='disaggregated'`` (a colocated
    sampler cannot overlap), ``weight_sync_strategy='in_place'`` + ``allow_partial_rollout`` (a publish
    overwrites the sampler's single live weight copy under the in-flight generations, so each is aborted and
    resumed on the fresh weights), and -- because GRPO trains on raw sampled tokens -- a
    ``rollout_importance_sampling_mode`` to correct the off-policy ratio (``old`` recomputed from the live
    policy minus ``rollout`` the sampler's own logprobs). The oracle is the staleness BOUND, an independent
    contract (the configured ``max_staleness``): ``version_span_mean`` and ``max_partial_span`` must never
    exceed 1 on any step. Remove the in-flight staleness gate and a trajectory that spans two publishes is
    consumed at lag 2 -> the bound goes red. ``assert_rl_trained`` (finite normalized loss + a non-zero
    ``lora_B`` read back from disk) proves the abort/resume + in-place overwrite did not corrupt training,
    and ``stream_publishes >= 1`` proves the trained policy really reached the separate-GPU sampler.
    """
    configs = rl_configs(
        rl_type='grpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=8),
        out_dir=str(tmp_path / 'out_async_one_step_off'),
        nproc=1,
        max_steps=6,
        rlhf_over={
            'num_generations': 2,
            'rollout_importance_sampling_mode': 'token_truncate',
        },
        reward_over={'orm': [_varied_content_reward]},
        rollout_over={
            'sampler_mode': 'disaggregated',
            'async_mode': 'one_step_off',
            'weight_sync_strategy': 'in_place',
            'allow_partial_rollout': True,
        },
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    _assert_overlapping_regime(configs, async_mode='one_step_off', max_staleness=1)
    assert_rl_trained(
        history,
        configs,
        'grpo',
        max_loss=20.0,
        require_keys=('version_span_mean', 'max_partial_span', 'stream_publishes'))

    _assert_staleness_bound(history, max_staleness=1)
    publishes = [r['stream_publishes'] for r in history]
    assert publishes[-1] >= 1, (
        f'stream_publishes never advanced ({publishes}): the overlapping run never pushed a trained policy '
        f'into the disaggregated sampler, so it rolled out on stale disk-loaded weights the whole time')


@pytest.mark.slow
@pytest.mark.accel(2)
def test_fully_async_disaggregated_staleness_bounded(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """Row 3: ``fully_async`` at ``max_staleness=2`` runs the sampler two versions ahead and bounds the span
    to 2 -- the deep-buffer regime the single-lookahead ``one_step_off`` structurally cannot reach.

    Same overlapping machinery as row 2 (disaggregated + in_place + partial rollout + IS correction), but the
    driver admits up to ``max_staleness=2`` versions ahead, so a ready buffer of in-flight trajectories
    absorbs a deeper version skew (``buffer_depth`` scales with ``max_staleness + 1``). The bound oracle
    scales with the knob: ``version_span_mean`` and ``max_partial_span`` must never exceed 2. This is the row
    that catches a deep-buffer staleness bound enforced only at depth 1 -- a straggler consumed at lag 3 goes
    red here, which row 2's single-lookahead window can never produce.
    """
    configs = rl_configs(
        rl_type='grpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=8),
        out_dir=str(tmp_path / 'out_async_fully'),
        nproc=1,
        max_steps=6,
        rlhf_over={
            'num_generations': 2,
            'rollout_importance_sampling_mode': 'token_truncate',
        },
        reward_over={'orm': [_varied_content_reward]},
        rollout_over={
            'sampler_mode': 'disaggregated',
            'async_mode': 'fully_async',
            'max_staleness': 2,
            'weight_sync_strategy': 'in_place',
            'allow_partial_rollout': True,
        },
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    _assert_overlapping_regime(configs, async_mode='fully_async', max_staleness=2)
    assert_rl_trained(
        history,
        configs,
        'grpo',
        max_loss=20.0,
        require_keys=('version_span_mean', 'max_partial_span', 'stream_publishes'))

    _assert_staleness_bound(history, max_staleness=2)


def _assert_overlapping_regime(configs, *, async_mode, max_staleness):
    """Anti-degeneracy: the RESOLVED config really is the overlapping disaggregated regime we asked for.

    The bound assertions in rows 2/3 would also pass on a silently-downgraded synchronous / colocate run
    (span 0 <= any bound), so pin the regime from the post-validation config: ``async_mode``, the mandatory
    disaggregated placement, the in_place + allow_partial_rollout publish mechanism the overlapping regime
    requires, and the staleness depth. ``validate`` already REFUSES the run outright if any of these is an
    illegal combination, so reaching here with them intact means the overlapping path was really taken.
    """
    rc = configs['rollout_config']
    assert rc.async_mode == async_mode, f'async_mode resolved to {rc.async_mode!r}, expected {async_mode!r}'
    assert rc.sampler_mode == 'disaggregated', (
        f'an overlapping regime must place the sampler on its own GPUs, but sampler_mode resolved to '
        f'{rc.sampler_mode!r} (a colocate downgrade would make the staleness bound vacuous)')
    assert rc.weight_sync_strategy == 'in_place', (
        f'overlapping in_place publish expected (the only strategy with a real version_span), got '
        f'{rc.weight_sync_strategy!r}')
    assert rc.allow_partial_rollout, (
        'overlapping in_place requires allow_partial_rollout: a publish aborts and resumes each in-flight '
        'generation on the fresh weights')
    assert rc.max_staleness == max_staleness, (
        f'max_staleness resolved to {rc.max_staleness}, expected {max_staleness}')


def _assert_staleness_bound(history, *, max_staleness):
    """Assert the version span never exceeds the configured staleness on ANY step (the bound invariant).

    This is the adversarial assertion for the overlapping rows: the staleness gate -- checked at BOTH the
    in-flight collect and the post-publish ready-buffer scan -- guarantees no sample is consumed more than
    ``max_staleness`` versions after it was generated. A missing in-flight gate lets a straggler that spans
    several publishes be consumed at a larger lag, which shows up here as a span over the bound.
    """
    for key in ('version_span_mean', 'max_partial_span'):
        series = [r[key] for r in history]
        assert all(v <= max_staleness for v in series), (
            f'{key} exceeded the staleness bound {max_staleness} on some step -- a trajectory was consumed '
            f'more than max_staleness versions after it was generated (the in-flight staleness gate is '
            f'missing, or a version pin was released early): {series}')


def _varied_content_reward(trajectories, **kwargs):
    """A deterministic ORM reward that VARIES with completion CONTENT (char-sum mod 11, scaled to [0,1)).

    GRPO's advantage is ``(r - group_mean) / group_std``, so a constant reward across a prompt's group gives
    std=0 -> advantage 0 -> zero gradient -> ``lora_B`` never moves and ``assert_rl_trained`` would fail for
    a reason unrelated to the async regime. Keying the reward on content gives most groups a non-zero std so
    the run really trains (mirrors ``test_placement_e2e`` / ``test_samplers_e2e``).
    """
    return [float(sum(ord(ch) for ch in last_assistant_text(c)) % 11) / 11.0 for c in trajectories]
