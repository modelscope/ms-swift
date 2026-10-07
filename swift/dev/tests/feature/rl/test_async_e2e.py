# Copyright (c) ModelScope Contributors. All rights reserved.
"""Async dimension (RL_PLAN §四/⑤): how far the rollout may run AHEAD of training -- the staleness ladder
that turns the synchronous loop into an overlapped one -- and the invariant that BOUNDS it.

Every online regime rides the SAME per-sample ``StreamingDriver`` and differs only in ``max_staleness``
(``async_mode='none'`` -> 0, ``'one_step_off'`` -> 1, ``'fully_async'`` -> the configured value). At
staleness 0 the driver DRAINS each window before publishing (a colocated sampler is serialized by that drain
barrier + device hand-over), so nothing is ever in flight at a publish and no sample is trained against a
policy newer than the one that generated it. At staleness > 0 the sampler runs ahead on its OWN GPUs
(``vllm_mode='disaggregated'`` is mandatory there -- a colocated sampler time-shares one DeviceGroup and
cannot overlap), a publish overwrites the live weights under in-flight generations (``in_place`` +
``allow_partial_rollout`` aborts and resumes each), and a sample may be trained up to ``max_staleness``
versions after it was generated. The failure this dimension kills is the staleness bound being violated -- a
slow trajectory that spans several publishes being consumed at a lag GREATER than ``max_staleness`` (the
in-flight staleness gate missing, so only the ready buffer is scanned), or a version pin released early --
and, at the synchronous end, off-policy signal leaking into a regime that must be strictly on-policy.

PLAN table -- every async corner case, mapped to its owner (the two fail-loudly deployment gates are pure
config rejections already anchored in ``test_infeasible_combos.py`` and are cross-referenced, not rewritten,
per RL_PLAN §二 "只补缺口，不重写"):

| # | case | input | expected (contract) | where covered | failure it kills |
|---|------|-------|---------------------|---------------|------------------|
| 1 | staleness=0 synchronous hard-zero | ``async_mode='none'`` colocate GRPO | every off-policy signal EXACTLY 0 on every step (``version_span_mean`` / ``max_partial_span`` / ``partial_ratio`` / ``off_policy_consumed``) | **THIS FILE** ``test_sync_staleness_zero_hard_invariant`` | synchronous semantics polluted by async -- a sample trained against a newer policy, a partial-rollout span, or an off-policy consumption leaking into the drain-barrier regime |
| 2 | one_step_off bounded overlap (staleness 1) | ``async_mode='one_step_off'`` disaggregated GRPO, in_place + allow_partial_rollout + IS mode | trains on disjoint GPUs; ``version_span_mean <= 1`` and ``max_partial_span <= 1`` on EVERY step; policy published at least once | **THIS FILE** ``test_one_step_off_disaggregated_staleness_bounded`` | straggler consumed at lag > max_staleness (in-flight staleness gate missing) / version pin released early |
| 3 | fully_async bounded overlap (staleness 2) | ``async_mode='fully_async'`` ``max_staleness=2`` disaggregated GRPO, in_place + partial + IS mode | trains; ``version_span_mean <= 2`` and ``max_partial_span <= 2`` on EVERY step | **THIS FILE** ``test_fully_async_disaggregated_staleness_bounded`` | the deep-buffer staleness bound enforced only at depth 1 (a straggler consumed at lag 3) |
| 4 | colocate x overlap refused | ``async_mode='one_step_off'`` + ``vllm_mode='colocate'`` | ``ValueError`` ("needs the sampler on its own GPUs") | ``test_infeasible_combos.py::test_colocate_async_overlap_rejected`` | a single-card ColocateHandover that cannot overlap silently degrading instead of refusing |
| 5 | async GRPO without off-policy correction refused | ``async_mode='one_step_off'``, no ``rollout_importance_sampling_mode`` | ``ValueError`` (requires the IS mode) | ``test_infeasible_combos.py::test_async_grpo_requires_importance_sampling`` | staleness > 0 trains GRPO on raw stale tokens with no importance-sampling correction, silently wrong |

Why rows 1-3 are the e2e gaps here: rows 4/5 are pure config rejections (no run to drive) already owned by
the infeasible-combination suite. The staleness LADDER itself -- 0 (hard-zero, colocate) / 1 (one_step_off) /
2 (fully_async) -- is what this dimension adds: it drives the REAL overlapping streaming driver (partial
rollout abort+resume, in-place weight overwrite under in-flight generation, an NCCL publish per window) end
to end and pins the version-span bound to the configured ``max_staleness``. Row 1 is colocate on purpose --
its drain barrier + per-cycle device hand-over (``serialize_generation``) is a DIFFERENT code path from the
disaggregated staleness-0 run in ``test_placement_e2e.py`` (no drain, no hand-over, NCCL instead), and it
asserts the full hard-zero off-policy set rather than ``test_samplers_e2e`` case 3's publish cadence.

Honest oracle note (testcase-planning step 4/8): whether ``version_span_mean`` actually RISES above 0 within
a 3-window tiny run is timing-dependent (it needs a generation slow enough to be consumed a version later),
so rows 2/3 assert the CEILING -- ``span <= max_staleness`` on every step -- which is exactly the invariant
the staleness gate guarantees, NOT a forced nonzero observation (that would be flaky). The bound is the
adversarial assertion: remove the in-flight staleness gate and a straggler is consumed at lag > max_staleness
-> the bound goes red. ``in_place`` is mandatory for rows 2/3 to be meaningful: under ``adapter_snapshot``
the driver reports ``version_span`` as a constant 0 by design, so the bound would be vacuous.

Reverse-verification: row 1 goes red if any off-policy signal is nonzero at staleness 0 (e.g. the drain
barrier removed so a publish overlaps an in-flight generation -> partial span > 0). Rows 2/3 go red if the
staleness gate is removed (a straggler consumed at lag > max_staleness -> bound violated), if the overlapping
regime silently downgrades to sync/colocate (``_assert_overlapping_regime``), or if the in_place +
partial-rollout machinery corrupts training (NaN loss / zero ``lora_B`` -> ``assert_rl_trained`` red). Run
with ``CUDA_VISIBLE_DEVICES=<cards> pytest swift/dev/tests/feature/rl/test_async_e2e.py -m slow``.
"""
import pytest

from swift.dev.tests.feature.rl.conftest import rl_configs, run_rl


@pytest.mark.slow
@pytest.mark.accel(1)
def test_sync_staleness_zero_hard_invariant(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """Row 1: at staleness 0 EVERY off-policy signal is exactly zero -- the synchronous regime is not
    polluted by the async machinery it now shares.

    ``async_mode='none'`` colocate GRPO rides the same StreamingDriver at ``max_staleness=0``: each pull
    first passes the drain barrier (assembly is not ready while any trajectory is in flight), the device is
    handed over, generation runs, and the window is trained and published with nothing left in flight. So no
    sample can be trained against a policy newer than the one that generated it (``version_span_mean == 0``),
    no generation is ever interrupted at a publish (``max_partial_span == 0`` / ``partial_ratio == 0``), and
    nothing is consumed off-policy (``off_policy_consumed == 0``). ``weight_sync_strategy`` is left unset so
    the run takes the production-derived colocate default (``in_place``), exercising the real sync path.

    The geometry (``num_generations=2``, one group per window, ``2 / (per_device_train_batch_size * ga * dp)
    = 2`` optimizer steps per window, ``max_steps=6``) spans three windows, so the invariant is checked
    across multiple publish boundaries rather than a single step. This is the control that gives rows 2/3's
    ``span <= max_staleness`` bound its meaning -- the ladder is 0 here, <=1 and <=2 there.
    """
    configs = rl_configs(
        rlhf_type='grpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=6),
        out_dir=str(tmp_path / 'out_async_none'),
        nproc=1,
        max_steps=6,
        rlhf_over={'num_generations': 2, 'orm': [_varied_content_reward]},
        rollout_over={'vllm_mode': 'colocate'},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(
        history,
        configs,
        'grpo',
        max_loss=20.0,
        require_keys=('version_span_mean', 'max_partial_span', 'partial_ratio', 'off_policy_consumed'))

    for key in ('version_span_mean', 'max_partial_span', 'partial_ratio', 'off_policy_consumed'):
        series = [r[key] for r in history]
        assert all(v == 0 for v in series), (
            f'a staleness-0 synchronous run must emit ZERO {key} on every step -- the drain barrier leaves '
            f'nothing in flight at a publish, so there is no version skew, no interrupted generation and no '
            f'off-policy consumption -- but got {series}')


@pytest.mark.slow
@pytest.mark.accel(2)
def test_one_step_off_disaggregated_staleness_bounded(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """Row 2: ``one_step_off`` overlaps one window on a disaggregated sampler and BOUNDS the version span to 1.

    ``async_mode='one_step_off'`` pins ``max_staleness=1``: the driver trains version ``v`` while admitting
    ``v+1``. ``validate`` forces the overlapping regime onto ``vllm_mode='disaggregated'`` (a colocated
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
        rlhf_type='grpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=8),
        out_dir=str(tmp_path / 'out_async_one_step_off'),
        nproc=1,
        max_steps=6,
        rlhf_over={
            'num_generations': 2,
            'orm': [_varied_content_reward],
            'rollout_importance_sampling_mode': 'token_truncate',
        },
        rollout_over={
            'vllm_mode': 'disaggregated',
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
        rlhf_type='grpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=8),
        out_dir=str(tmp_path / 'out_async_fully'),
        nproc=1,
        max_steps=6,
        rlhf_over={
            'num_generations': 2,
            'orm': [_varied_content_reward],
            'rollout_importance_sampling_mode': 'token_truncate',
        },
        rollout_over={
            'vllm_mode': 'disaggregated',
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
    assert rc.vllm_mode == 'disaggregated', (
        f'an overlapping regime must place the sampler on its own GPUs, but vllm_mode resolved to '
        f'{rc.vllm_mode!r} (a colocate downgrade would make the staleness bound vacuous)')
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


def _varied_content_reward(completions, **kwargs):
    """A deterministic ORM reward that VARIES with completion CONTENT (char-sum mod 11, scaled to [0,1)).

    GRPO's advantage is ``(r - group_mean) / group_std``, so a constant reward across a prompt's group gives
    std=0 -> advantage 0 -> zero gradient -> ``lora_B`` never moves and ``assert_rl_trained`` would fail for
    a reason unrelated to the async regime. Keying the reward on content gives most groups a non-zero std so
    the run really trains (mirrors ``test_placement_e2e`` / ``test_samplers_e2e``).
    """
    return [float(sum(ord(ch) for ch in c) % 11) / 11.0 for c in completions]
