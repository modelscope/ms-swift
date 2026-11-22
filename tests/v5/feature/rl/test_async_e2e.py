# Copyright (c) ModelScope Contributors. All rights reserved.
"""Async dimension (RL_PLAN §四/⑤): how far the rollout may run AHEAD of training -- the staleness ladder
that turns the synchronous loop into an overlapped one -- and the invariant that BOUNDS it.

The overlapped regimes ride a per-sample ``StreamingDriver`` and differ only in ``max_staleness``
(``async_mode='one_step_off'`` -> 1, ``'fully_async'`` -> the configured value). At staleness > 0 the sampler
runs ahead on its OWN GPUs (``sampler_mode='disaggregated'`` is mandatory there -- a colocated sampler
time-shares one DeviceGroup and cannot overlap), a publish overwrites the live weights under in-flight
generations and pauses then resumes each on the new weights (this abort/resume is now UNCONDITIONAL
behavior -- 78a1a1c55 retired the ``weight_sync_strategy`` / ``allow_partial_rollout`` knobs), and a sample
may be trained up
to ``max_staleness`` versions after it was generated; the driver tracks the live policy version (``policy_version``)
and stamps each sample's version skew (``version_span_mean``). The failure this dimension kills is the
staleness bound being violated -- a slow trajectory that spans several publishes being consumed at a lag
GREATER than ``max_staleness`` (the in-flight staleness gate missing, so only the ready buffer is scanned), or
a version pin released early.

A COLOCATE run is the synchronous 0-rung: a colocated sampler cannot overlap with the trainer (one device,
time-shared), so generation and training alternate, never concurrent. ``run_grpo`` builds the SAME
``StreamingDriver`` for colocate and disaggregated (the earlier driver-less ``_colocate_loop`` was unified
away), so the driver telemetry IS present -- but the run is strictly on-policy, expressed as a measured ZERO
version skew (``version_span_mean == 0`` and ``max_partial_span == 0`` on every step) rather than as the absence
of telemetry. The 0-rung pins span == 0; the overlapping regimes pin span <= max_staleness.

PLAN table -- every async corner case, mapped to its owner (the two fail-loudly deployment gates are pure
config rejections already anchored in ``test_infeasible_combos.py`` and are cross-referenced, not rewritten,
per RL_PLAN §二 "只补缺口，不重写"):

| # | case | input | expected (contract) | where covered | failure it kills |
|---|------|-------|---------------------|---------------|------------------|
| 1 | staleness=0 synchronous colocate (zero skew) | ``async_mode='none'`` colocate GRPO | trains strictly on-policy; resolved regime is colocate + async none; ``version_span_mean == 0`` and ``max_partial_span == 0`` on EVERY step (colocate rides the unified driver but reports zero skew) | **THIS FILE** ``test_sync_colocate_is_on_policy_zero_skew`` | a version skew creeping into a regime that alternates generation and training (colocate silently overlapping), or a silent switch to an overlapping regime |
| 2 | one_step_off bounded overlap (staleness 1) | ``async_mode='one_step_off'`` disaggregated GRPO, unconditional pause/resume publish + IS mode | trains on disjoint GPUs; ``version_span_mean <= 1`` and ``max_partial_span <= 1`` on EVERY step; policy published at least once | **THIS FILE** ``test_one_step_off_disaggregated_staleness_bounded`` | straggler consumed at lag > max_staleness (in-flight staleness gate missing) / version pin released early |
| 3 | fully_async bounded overlap (staleness 2) | ``async_mode='fully_async'`` ``max_staleness=2`` disaggregated GRPO, unconditional pause/resume publish + IS mode | trains; ``version_span_mean <= 2`` and ``max_partial_span <= 2`` on EVERY step | **THIS FILE** ``test_fully_async_disaggregated_staleness_bounded`` | the deep-buffer staleness bound enforced only at depth 1 (a straggler consumed at lag 3) |
| 4 | colocate x overlap refused | ``async_mode='one_step_off'`` + ``sampler_mode='colocate'`` | ``ValueError`` ("needs the sampler on its own GPUs") | ``test_infeasible_combos.py::test_colocate_async_overlap_rejected`` | a single-card ColocateHandover that cannot overlap silently degrading instead of refusing |
| 5 | async GRPO without off-policy correction refused | ``async_mode='one_step_off'``, no ``rollout_importance_sampling_mode`` | ``ValueError`` (requires the IS mode) | ``test_infeasible_combos.py::test_async_grpo_requires_importance_sampling`` | staleness > 0 trains GRPO on raw stale tokens with no importance-sampling correction, silently wrong |

Why rows 1-3 are the e2e gaps here: rows 4/5 are pure config rejections (no run to drive) already owned by
the infeasible-combination suite. The staleness LADDER itself -- 0 (synchronous colocate) / 1 (one_step_off) /
2 (fully_async) -- is what this dimension adds. Rows 2/3 drive the REAL overlapping streaming driver (partial
rollout abort+resume, in-place weight overwrite under in-flight generation, an NCCL publish per window) end to
end and pin the version-span bound to the configured ``max_staleness``. Row 1 is the 0-rung: a colocated
sampler cannot overlap, so although it rides the same unified ``StreamingDriver``, the strictly-on-policy
guarantee shows up as a measured ZERO span -- the row pins that regime (it trains, it resolves to colocate +
async none, and ``version_span_mean`` / ``max_partial_span`` are 0 on every step), which is the contrast that
makes rows 2/3's ``span <= max_staleness`` bound meaningful.

Honest oracle note (testcase-planning step 4/8): whether ``version_span_mean`` actually RISES above 0 within
a 3-window tiny run is timing-dependent (it needs a generation slow enough to be consumed a version later),
so rows 2/3 assert the CEILING -- ``span <= max_staleness`` on every step -- which is exactly the invariant
the staleness gate guarantees, NOT a forced nonzero observation (that would be flaky). The bound is the
adversarial assertion: remove the in-flight staleness gate and a straggler is consumed at lag > max_staleness
-> the bound goes red. The bound is only meaningful on the overlapping regime: a sync/colocate downgrade
reports ``version_span`` 0 by construction, so ``_assert_overlapping_regime`` pins the disaggregated
placement + publication behavior to keep the ceiling from going vacuous.

Reverse-verification: row 1 goes red if a version skew creeps into the synchronous colocate run
(``version_span_mean`` or ``max_partial_span`` rises above 0 on any step -- generation overlapped training),
if the resolved regime is not the synchronous colocate one, or if the run does not train on-policy (NaN loss /
zero ``lora_B``).
Rows 2/3 go red if the staleness gate is removed (a straggler consumed at lag > max_staleness -> bound
violated), if the overlapping regime silently downgrades to sync/colocate (``_assert_overlapping_regime``), or
if the pause/resume publish machinery corrupts training (NaN loss / zero ``lora_B`` -> ``assert_rl_trained``
red). Run with ``CUDA_VISIBLE_DEVICES=<cards> pytest tests/v5/feature/rl/test_async_e2e.py -m slow``.
"""
import pytest

from tests.v5.feature.rl.conftest import rl_configs, run_rl
from twinkle.data_format import last_assistant_text


@pytest.mark.slow
@pytest.mark.accel(1)
def test_sync_colocate_is_on_policy_zero_skew(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """Row 1: a colocate ``async_mode='none'`` run is the synchronous 0-rung -- strictly on-policy (zero skew).

    A colocated sampler time-shares ONE device with the trainer, so generation and training cannot overlap.
    ``run_grpo`` now builds the SAME ``StreamingDriver`` for colocate and disaggregated (the earlier
    driver-less ``_colocate_loop`` was unified away), so the driver telemetry IS present on this path -- the
    strictly-on-policy guarantee is no longer "no driver, no version face" but a measured ZERO version skew:
        ``version_span_mean`` and ``max_partial_span`` are 0 on EVERY step (no sample is trained against a
    newer policy than it was generated under). This row pins three things:

    * it really trains on-policy -- finite normalized loss + a non-zero ``lora_B`` read back from disk, with
      the GRPO policy-gradient fingerprint (``train/approx_kl`` / ``train/clip_ratio``) present, so it rolled
      out and took a real policy-gradient step rather than degrading to a non-rollout loop;
    * the resolved config is the synchronous colocate regime (``sampler_mode='colocate'``, ``async_mode='none'``),
      so the run cannot have been silently switched to an overlapping regime;
    * the version skew is ZERO on every step (``version_span_mean == 0`` and ``max_partial_span == 0``). A
      regression that let colocate overlap generation with training would consume a sample a version late and
      push the span above 0 -> red. This is the 0-rung contrast that gives rows 2/3's ``span <= max_staleness``
      bound its meaning: colocate pins span == 0, the overlapping regimes pin span <= max_staleness.

    The geometry (``num_generations=2``, ``max_steps=6``) spans three windows, so the on-policy training and the
    zero skew are checked across multiple sync boundaries rather than a single step.
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

    assert_rl_trained(
        history, configs, 'grpo', max_loss=20.0, require_keys=('train/approx_kl', 'train/clip_ratio'))

    for key in ('version_span_mean', 'max_partial_span'):
        series = [r.get(key) for r in history]
        assert all(v == 0 for v in series), (
            f'colocate async_mode none is strictly on-policy, so {key} must be 0 on every step (no sample '
            f'trained against a newer policy); a non-zero span means generation overlapped training and a '
            f'version skew leaked into the synchronous run: {series}')


@pytest.mark.slow
@pytest.mark.accel(2)
def test_one_step_off_disaggregated_staleness_bounded(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """Row 2: ``one_step_off`` overlaps one window on a disaggregated sampler and BOUNDS the version span to 1.

    ``async_mode='one_step_off'`` pins ``max_staleness=1``: the driver trains version ``v`` while admitting
    ``v+1``. ``validate`` forces the overlapping regime onto ``sampler_mode='disaggregated'`` (a colocated
    sampler cannot overlap); the in-place publish that overwrites the sampler's single live weight copy under
    the in-flight generations -- pausing each and resuming it on the fresh weights -- is now UNCONDITIONAL
    behavior (78a1a1c55 retired the ``weight_sync_strategy`` / ``allow_partial_rollout`` knobs), and -- because
    GRPO trains on raw sampled tokens -- a
    ``rollout_importance_sampling_mode`` to correct the off-policy ratio (``old`` recomputed from the live
    policy minus ``rollout`` the sampler's own logprobs). The oracle is the staleness BOUND, an independent
    contract (the configured ``max_staleness``): ``version_span_mean`` and ``max_partial_span`` must never
    exceed 1 on any step. Remove the in-flight staleness gate and a trajectory that spans two publishes is
    consumed at lag 2 -> the bound goes red. ``assert_rl_trained`` (finite normalized loss + a non-zero
    ``lora_B`` read back from disk) proves the abort/resume + in-place overwrite did not corrupt training,
    and ``policy_version`` advancing beyond its initial value proves the trained policy really reached the separate-GPU sampler.
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
        },
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    _assert_overlapping_regime(configs, history, async_mode='one_step_off', max_staleness=1)
    assert_rl_trained(
        history,
        configs,
        'grpo',
        max_loss=20.0,
        require_keys=('version_span_mean', 'max_partial_span', 'policy_version'))

    _assert_staleness_bound(history, max_staleness=1)


@pytest.mark.slow
@pytest.mark.accel(2)
def test_fully_async_disaggregated_staleness_bounded(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """Row 3: ``fully_async`` at ``max_staleness=2`` runs the sampler two versions ahead and bounds the span
    to 2 -- the deep-buffer regime the single-lookahead ``one_step_off`` structurally cannot reach.

    Same overlapping machinery as row 2 (disaggregated + unconditional pause/resume publish + IS correction), but the
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
        },
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    _assert_overlapping_regime(configs, history, async_mode='fully_async', max_staleness=2)
    assert_rl_trained(
        history,
        configs,
        'grpo',
        max_loss=20.0,
        require_keys=('version_span_mean', 'max_partial_span', 'policy_version'))

    _assert_staleness_bound(history, max_staleness=2)


def _assert_overlapping_regime(configs, history, *, async_mode, max_staleness):
    """Anti-degeneracy: the RESOLVED config really is the overlapping disaggregated regime we asked for,
    and the publication BEHAVIOR (pause + resume of in-flight requests) actually ran.

    The bound assertions in rows 2/3 would also pass on a silently-downgraded synchronous / colocate run
    (span 0 <= any bound), so pin the regime from the post-validation config: ``async_mode``, the mandatory
    disaggregated placement, and the staleness depth. ``validate`` already REFUSES the run outright if any
    of these is an illegal combination, so reaching here with them intact means the overlapping path was
    really taken.

    The in-place publish + partial-rollout abort/resume became UNCONDITIONAL behavior (78a1a1c55 deleted the
    ``weight_sync_strategy`` / ``allow_partial_rollout`` knobs -- a sync now always pauses the sampler's
    running requests and lets them continue on the new weights), so this pins the BEHAVIOR rather than a
    config value that no longer exists: a publication really reached the separate-GPU sampler (the driver's
    ``policy_version`` counter advanced beyond its initial value -- the product reports the live policy version,
    not a separate ``stream_publishes`` tally) and the partial-rollout face is reported (``max_partial_span``
    present), which is what proves in-flight requests were paused and resumed rather than discarded. Restoring
    the deleted fields to go green is explicitly forbidden.
    """
    rc = configs['rollout_config']
    assert rc.async_mode == async_mode, f'async_mode resolved to {rc.async_mode!r}, expected {async_mode!r}'
    assert rc.sampler_mode == 'disaggregated', (
        f'an overlapping regime must place the sampler on its own GPUs, but sampler_mode resolved to '
        f'{rc.sampler_mode!r} (a colocate downgrade would make the staleness bound vacuous)')
    assert rc.max_staleness == max_staleness, (
        f'max_staleness resolved to {rc.max_staleness}, expected {max_staleness}')
    versions = [r['policy_version'] for r in history]
    assert versions[-1] > versions[0], (
        f'policy_version never advanced ({versions}): the overlapping run never pushed a trained policy '
        f'into the disaggregated sampler, so it rolled out on stale disk-loaded weights the whole time')
    assert all('max_partial_span' in r for r in history), (
        'the partial-rollout face (max_partial_span) is absent from some step: a publication must pause and '
        'resume each in-flight generation on the fresh weights, and that abort/resume is what this telemetry '
        'reports -- its absence means the unconditional pause/resume behavior did not run')


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
