# Copyright (c) ModelScope Contributors. All rights reserved.
"""Dimension ⑧ -- loss regression vs legacy: dev's RL objective must restore ``swift.rlhf_main``'s loss.

``test_backends_e2e`` proves the dev *transformers* and dev *megatron* backends agree, and its docstring
explicitly defers DEEP per-step parity to here ("step 2 drifts apart ... which is why deep per-step parity
is anchored against legacy in ``test_parity_legacy``, not here"). This file is that anchor: it drives the
REAL legacy pipeline (``swift.rlhf_main`` -> HF/RLHFArguments -> TRL ``<TYPE>Trainer``) and the REAL dev
pipeline (``run_rl`` -> Ray ``run_dpo`` -> twinkle loss) on the SAME cached Qwen2.5-0.5B, the SAME
preference rows and MATCHED deterministic hyperparameters, and asserts dev restores legacy's per-step loss
trajectory. The contract is an INDEPENDENT oracle (legacy's own trainer), not a self-consistency round-trip.

Mechanism (RL_PLAN section 8: isolate a subprocess per side and compare per-step loss, mirroring
``megatron_sft.py``): each side runs in its OWN plain-python subprocess (``_runners/rl_parity.py``), because
legacy builds a TRL trainer on the HF Trainer/accelerate stack while dev spins up a Ray session + the twinkle
runtime, and the two cannot safely share one interpreter. Offline preference training is single-card with no
rollout, so -- unlike ``megatron_sft.py`` -- this needs no torchrun, just one process per side.

Why the REFERENCE-FREE variants (cpo/orpo) carry the parity oracle, not DPO (bands below are MEASURED on
probes at this cell's exact config -- 4 steps, lr 1e-4, seed 42, system-pinned rows -- deterministic
across runs, not guessed):

  - **cpo/orpo step-1 is DATA-DEPENDENT and restores to the logging floor.** These objectives are
    reference-free: with the identity adapter (``lora_B``=0 at init) the step-1 loss is the BASE model's
    own NLL / odds-ratio contrast on chosen-vs-rejected (probe: cpo 2.95896, orpo 2.82758 -- far from
    ln2), a deterministic function of the weights + which tokens are scored. So step-1 is a STRONG
    discriminant: a label shift, a wrong completion mask, or an encode bug moves it by O(10%). Measured
    step-1 rel gap: cpo 1.5e-6, orpo 8e-7 (dominated by dev's 5-decimal loss logging); band 1e-3 keeps
    ~1000x margin.
  - **later steps are BOUNDED, not exact** (probe worst rel: cpo 5.2e-2, orpo 5.6e-2, at step 4). Past
    step-1 the update consumes the LoRA-A init, which legacy peft and dev twinkle draw at DIFFERENT
    RNG-stream positions, so the two runs are valid-but-distinct trajectories -- the same reason the SFT
    parity grid bounds its later steps. Bands: cpo 0.15 (~3x margin), orpo 0.15 (~2.7x margin). The
    step-1 anchor, not the later band, is the load-bearing gate.

Per-step parity rests on TWO preconditions the RLHF-recipe rework had silently broken (the 2026-10-24
registered gap, dev ~3.3-3.8 vs legacy ~8.9-9.2 rel ~0.6, resolved 2026-10-26 -- it was NOT an
objective/normalization divergence, both objectives compute the same formula):

  - **same read order.** ``--train_dataloader_shuffle false`` used to be DEAD on the legacy RLHF path:
    the SFT trainers consume it in DataLoaderMixin's BatchSamplerShard, but the CPO/ORPO/KTO/Reward
    trainer family never reaches that mixin, so HF Trainer's default RandomSampler permuted the data
    off the global RNG (first draw after set_seed == randperm(data_seed)) while dev read rows
    sequentially -- per-step "parity" compared different rows. Fixed in
    ``RLHFTrainerMixin._get_train_sampler`` (SequentialSampler when the flag is false;
    shuffle=True behavior is untouched).
  - **same system turn.** legacy's qwen2_5 template injects the model's DEFAULT system prompt into
    rows that lack one; dev passes the row through untouched. The rows here carry an EXPLICIT system
    (``QWEN_SYSTEM``, single-sourced in ``rl_parity.py``) so both sides encode identical content -- the
    same fix the GRPO reward-path cells already apply.

Honest exclusions (each asserted as a body-level ``pytest.skip`` with its probe evidence, mirroring the SFT
``test_parity_grid`` exclusions -- NOT a vacuous "it trained" test, and NOT silently omitted):

| # | cell | contract / why excluded | failure it kills (or why unattainable) |
|---|------|-------------------------|----------------------------------------|
| 1 | cpo, orpo per-step loss | dev restores legacy: step-1 rel<1e-3 AND step-1 far from ln2 (data-dependent); later rel<band + net window decrease | dev loss formula / label shift / completion mask / encode drifts from legacy |
| 2 | dpo deep parity | EXCLUDED: step-1 is the degenerate ln2 (policy==ref cancels to 0 for ANY masking -- cannot catch a shift bug; already anchored in test_algorithms_e2e / test_backends_e2e), and step2+ is chaotic (probe rel 7.3% -> 43.5% -> 41.7%) because the adapter-dominated loss -> 0 divides the gap | mistaking a self-cancelling ln2 for parity signal |
| 3 | grpo per-step loss (advantage/clip/KL) | EXCLUDED: dev grpo.py and legacy grpo_trainer.py BOTH import ``rl_core.advantage.compute_advantages`` (shared code -> comparing it is a self-consistency round-trip); the policy loss is already unit-anchored vs a hand-computed PPO-clip oracle in test_recipes.py; and each side runs an independent temperature>0 vLLM rollout so the completions -- and every per-step loss -- differ by construction. The comparable face -- the REWARD PATH under the same ORM (baseline -> climb -> saturation, 6/6 seed-grid stable) -- is asserted by ``test_grpo_reward_path_parity_vs_legacy`` with a zero-lr control | a fake "parity" that only proves shared code equals itself |
| 4 | full-param parity | EXCLUDED: dev raises ``NotImplementedError`` for ``tuner='full'`` (only lora/adalora/trainable_tokens), so there is no dev side to compare | assuming a dev full-param path that does not exist |
| 5 | rm / seq_cls random head | EXCLUDED: the scalar reward head is freshly initialized, and the two pipelines consume different RNG before that init, so step-1 loss is init-dominated, not data-dominated (same root cause the SFT parity grid excludes seq_cls/reranker for); dev rm training is proven e2e in test_algorithms_e2e / test_backends_e2e | a vacuous "anything goes" band on an init-dominated loss |

Run (offline, single card): ``CUDA_VISIBLE_DEVICES=<c> pytest
swift/dev/tests/feature/rl/test_parity_legacy.py -m slow``.
"""
import json
import math
import os
import sys

import pytest

from swift.dev.tests._runners import Runners
from swift.dev.tests._runners.rl_parity import QWEN_SYSTEM

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

_LN2 = math.log(2.0)
_STEPS = 4
_LR = 1e-4
# step-1 is the load-bearing anchor: identity adapter (lora_B=0) -> pure base-model forward, independent of
# the LoRA-A init RNG stream and of any optimizer update. Probe-measured rel cpo 1.5e-6 / orpo 8e-7 (dev
# logs loss at 5 decimals, which is the floor); 1e-3 keeps ~1000x margin yet still trips on a real
# shift/mask/encode drift.
_STEP1_BAND = 1e-3

#: cell -> later-step relative band (probe-measured worst + margin, at this cell's exact config; see the
#: module docstring). The measured worsts (cpo 5.2e-2, orpo 5.6e-2, both at step 4) come from the
#: LoRA-A init RNG-stream fork amplifying under lr 1e-4; the bands carry ~3x margin for library drift.
_LATER_BAND = {'cpo': 0.15, 'orpo': 0.15}


def _run_parity_side(backend, rl_type, data_path, tmp_path, *, steps=_STEPS, lr=_LR, timeout=1800,
                     sampler_mode=None):
    """Run ONE side (legacy|dev) of the parity pair in its own plain-python subprocess; return its result.

    Plain python, not torchrun: offline preference training is single-card with no rollout, so each side is
    one process (contrast ``megatron_sft.py``, which needs torchrun for world_size>=2). Isolation is still
    required -- see the module docstring. ``Runners.launch`` bounds the run and kills the whole process
    group on timeout; a crashed side leaves no result file, which is turned into an AssertionError carrying
    the subprocess stdout/stderr tail (the result itself is read back from disk, never trusted in-memory).

    Returns the runner's full result dict (``losses`` always; ``rewards`` for the grpo reward-path
    contract) so the caller picks the trajectory its cell compares.
    """
    runner = Runners.path('rl_parity')
    result_path = str(tmp_path / f'{backend}_{rl_type}.json')
    cmd = [
        sys.executable, runner,
        '--backend', backend,
        '--rl_type', rl_type,
        '--data', data_path,
        '--out', result_path,
        '--out_dir', str(tmp_path / f'{backend}_{rl_type}_out'),
        '--steps', str(steps),
        '--lr', str(lr),
        '--tuner', 'lora',
    ]
    if sampler_mode is not None:
        cmd += ['--sampler_mode', sampler_mode]
    proc = Runners.launch(cmd, timeout=timeout)
    if not os.path.exists(result_path):
        raise AssertionError(
            f'{backend}/{rl_type} parity runner produced no result (exit={proc.returncode}). '
            f'stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-3000:]}')
    with open(result_path) as f:
        return json.load(f)


def _pin_system(data_path, tmp_path):
    """Prepend the shared ``QWEN_SYSTEM`` turn to every preference row.

    legacy's qwen2_5 template injects the model's DEFAULT system prompt when a row has none; dev's
    to_trajectory passes the row through untouched. Without an explicit system the two sides encode
    DIFFERENT content for the same row and step-1 diverges by ~20% (probe: same row dev 3.80 vs legacy
    2.96). An explicit row system is honored by both, single-sourcing the prompt (see the module
    docstring's "same system turn" precondition).
    """
    out = tmp_path / 'preference_system.jsonl'
    system = {'role': 'system', 'content': QWEN_SYSTEM}
    with open(data_path) as fin, open(out, 'w') as fout:
        for line in fin:
            row = json.loads(line)
            row['messages'] = [system] + row['messages']
            fout.write(json.dumps(row, ensure_ascii=False) + chr(10))
    return str(out)


@pytest.mark.parametrize('rl_type', sorted(_LATER_BAND))
def test_reference_free_loss_parity_vs_legacy(rl_type, rl_data, tmp_path):
    """Row 1: dev's ``<cpo|orpo>`` per-step loss restores legacy ``swift.rlhf_main``'s (same model/data/hparams).

    Both sides ride the SAME cached Qwen2.5-0.5B + the SAME 8 preference rows + matched deterministic
    hyperparameters (constant LR / no warmup / seed 42 / shuffle off / beta 0.1 / rpo_alpha 0 / LoRA
    r8-a32-all-linear / bf16), each in its own subprocess. The reference-free objective makes step-1
    data-dependent (the base model's NLL / odds contrast), so step-1 is a strong discriminant for a
    shift/mask/encode bug -- the exact signal DPO's degenerate ln2 step-1 cannot provide.

    Reverse-verify: the step-1 gate has two independent knobs, each proven to trip it -- (a) flip the sign
    of dev's chosen/rejected log-probability difference (or shift the completion labels by one token) in
    the twinkle loss and step-1 moves off legacy's ~2.96/~2.83 by >>1e-3; (b) revert the legacy sampler
    fix and legacy resumes consuming rows in randperm(data_seed) order, so step-1 becomes a DIFFERENT
    row's loss (rel ~2). Dropping the ``not ln2`` guard would let a silent degradation to a policy==ref
    cancel (both sides -> ln2) pass as a vacuous "parity".
    """
    data_path = _pin_system(rl_data.preference(n=8), tmp_path)
    dev = _run_parity_side('dev', rl_type, data_path, tmp_path)['losses']
    legacy = _run_parity_side('legacy', rl_type, data_path, tmp_path)['losses']
    print(f'\n{rl_type} dev-vs-legacy parity: dev={dev} legacy={legacy}')

    assert len(dev) == len(legacy) == _STEPS, \
        f'{rl_type}: expected {_STEPS} per-step losses, dev={dev} legacy={legacy}'
    # loss == loss is the NaN test (NaN != itself); not a typo.
    assert all(x == x and abs(x) != float('inf') for x in dev + legacy), \
        f'{rl_type}: non-finite loss dev={dev} legacy={legacy}'

    # Non-degeneracy guard FIRST: the legacy reference step-1 must be data-dependent, not the ln2 cancel.
    # If it sat at ln2 this cell would carry no more signal than DPO's step-1 (both sides self-cancel), and
    # the rel check below could pass vacuously -- so prove the reference-free objective is really computed.
    assert abs(legacy[0] - _LN2) > 1.0, (
        f'{rl_type}: legacy step-1 loss {legacy[0]:.5f} ~= ln2 {_LN2:.5f} -- the reference-free objective '
        'degenerated to a policy==ref cancel (data-dependent NLL/odds term missing), so this is not a real '
        'parity signal')

    # step-1: the load-bearing, RNG-independent anchor (identity adapter -> pure base-model forward).
    rel1 = abs(dev[0] - legacy[0]) / max(abs(legacy[0]), 1e-8)
    assert rel1 < _STEP1_BAND, (
        f'{rl_type} step-1 dev-vs-legacy mismatch: dev={dev[0]:.6f} legacy={legacy[0]:.6f} rel={rel1:.2e} '
        f'(band {_STEP1_BAND}) -- the reference-free objective (logp-diff sign / completion mask / label '
        'shift / encode) diverges from legacy')

    # net window decrease: the rows differ in difficulty, so per-step monotonicity is not this data's
    # property (probe: legacy cpo steps 1-4 = [2.96, 3.10, 8.65, 1.23]); the sound sign-error oracle is
    # that the whole window decreases on BOTH sides (a flipped preference gradient climbs instead).
    assert dev[-1] < dev[0] and legacy[-1] < legacy[0], (
        f'{rl_type}: loss did not decrease over the window (a sign error would climb): dev={dev} '
        f'legacy={legacy}')
    later_band = _LATER_BAND[rl_type]
    for i in range(1, _STEPS):
        rel = abs(dev[i] - legacy[i]) / max(abs(legacy[i]), 1e-8)
        assert rel < later_band, (
            f'{rl_type} step {i + 1} dev-vs-legacy divergence: dev={dev[i]:.6f} legacy={legacy[i]:.6f} '
            f'rel={rel:.2e} (band {later_band})')


def test_dpo_deep_parity_excluded_chaotic():
    """GAP(exclusion): DPO deep per-step parity vs legacy is not a sound contract.

    DPO's step-1 is the degenerate ln2: with the identity adapter the policy equals its own disabled-adapter
    reference, so the chosen/rejected log-ratio difference is exactly 0 and ``-log sigmoid(beta*0) == ln2``
    for ANY completion mask or label alignment -- it cannot catch the shift/mask/encode bug a parity cell
    exists to catch (that ln2 anchor is already asserted in test_algorithms_e2e and test_backends_e2e). Past
    step-1 DPO's adapter-dominated loss decays toward 0, which divides the legacy-vs-dev gap: a reference
    probe measured rel 7.3% (step2) -> 43.5% (step3) -> 41.7% (step4) -- chaotic, driven by the LoRA-A init
    RNG-stream difference amplified by the shrinking denominator, not a dev defect. The reference-free
    cpo/orpo cell above carries the data-dependent step-1 anchor DPO cannot, so DPO deep parity is excluded
    rather than band-fudged into a vacuous pass.
    """
    pytest.skip('DPO step-1 is the degenerate ln2 (policy==ref cancels for any masking -- cannot catch a '
                'shift/mask bug; already anchored in test_algorithms_e2e/test_backends_e2e) and step2+ is '
                'chaotic (probe rel 7.3%->43.5%->41.7%: adapter-dominated loss->0 divides the LoRA-A-init '
                'RNG gap). cpo/orpo carry the data-dependent step-1 parity anchor instead.')


def test_grpo_parity_excluded_shared_code_and_rollout():
    """GAP(exclusion): GRPO per-step LOSS parity vs legacy is not an independent-oracle comparison.

    RL_PLAN section 5 Row 2 asks for "dev vs legacy grpo_algorithm, advantage/clip/KL numerically equal".
    That is unattainable as a NON-circular test for three reasons: (1) dev ``recipe/grpo.py`` and legacy
    ``grpo_trainer.py`` BOTH ``import rl_core.advantage.compute_advantages`` -- the advantage is SHARED code,
    so comparing dev's advantage to legacy's is a self-consistency round-trip (the skill's forbidden
    pattern), not an oracle; (2) the GRPO policy loss (clip/KL) is already unit-anchored against a
    hand-computed PPO-clip oracle in test_recipes.py, which is the independent ground truth; (3) a full e2e
    GRPO loss comparison is incomparable because each side drives an independent temperature>0 vLLM rollout,
    so the sampled completions -- and therefore every per-step loss -- differ by construction (the same
    non-determinism test_backends_e2e documents for its megatron GRPO run).

    What IS comparable -- the reward PATH -- is covered by ``test_grpo_reward_path_parity_vs_legacy``
    below: the y-axis is the same single-sourced ORM on both sides, window noise is averaged, and reward
    is GRPO's first-class optimization target, so "both pipelines climb from the same baseline to
    saturation on the same window budget" is a real cross-pipeline contract without comparing per-step
    losses.
    """
    pytest.skip('GRPO advantage is SHARED code (dev grpo.py and legacy grpo_trainer.py both import '
                'rl_core.advantage.compute_advantages), so a dev-vs-legacy advantage comparison is a '
                'self-consistency round-trip, not an oracle; the clip/KL policy loss is already unit-anchored '
                'vs a hand-computed PPO-clip oracle in test_recipes.py; and independent temperature>0 vLLM '
                'rollouts make per-step GRPO loss incomparable by construction. The comparable reward PATH '
                'is asserted by test_grpo_reward_path_parity_vs_legacy.')


#: The GRPO reward-path contract (probe-derived 2026-10-26, 6-run seed grid: 3 legacy + 3 dev seeds, all
#: climbing -- window points below are one per ROLLOUT WINDOW, 4 prompts x 4 generations = 16 samples =
#: 4 optimizer steps at per-device batch 4):
#:   legacy s42 [0.381, 0.831, 0.887, 1.0, 1.0, 1.0, 1.0, 1.0]
#:   legacy s43 [0.213, 0.381, 0.606, 1.0, 0.944, 1.0, 0.944, 1.0]
#:   legacy s44 [0.55, 1.0, 0.944, 1.0, 0.944, 1.0, 1.0, 1.0]
#:   dev    s42 [0.325, 0.831, 1.0, 0.944, 0.944, 0.831, 1.0, 1.0]
#:   dev    s43 [0.269, 0.719, 0.55, 0.944, 1.0, 1.0, 1.0, 1.0]
#:   dev    s44 [0.213, 0.156, 0.269, 0.719, 1.0, 0.944, 1.0, 1.0]
_GRPO_STEPS = 32
_GRPO_LR = 3e-4
#: window cadence: legacy logs one rewards/<name>/mean per generation cycle (8 points); dev's driver-side
#: CompletionRewardMetric is read once per optimizer step, so 32 entries carry each window aggregate 4x
#: and collapse to 8 points. Both cadences are asserted below so a change fails loudly, not silently.
_GRPO_WINDOWS = 8
_DEV_POINTS_PER_WINDOW = 4
#: window-0 (pre-update base policy) mean reward: the shared ORM's phrase has a MID base rate (~27% of
#: 112 sampled base-model completions), so window-0 sits well inside (0.15, 0.65) on every grid run
#: (observed 0.213..0.55). This band is the reward-channel sanity gate: a broken scorer wiring (missing
#: ORM, key drift, constant reward) puts window-0 at a sentinel/0/1, far outside.
_GRPO_BASELINE_BAND = (0.15, 0.65)
#: PEAK climb: max window point >= 0.75. The gate is the PEAK, not the terminal mean: at a constant
#: lr 3e-4 both pipelines show a "climb then late collapse" mode (a 7th probe run dev reached 1.0 by
#: window 2 then fell to 0.1 in the last two windows; legacy shows the same shape in the low-base-rate
#: grid), so terminal-window asserts are flaky. The peak is the stable discriminant: every observed
#: LEARNING run peaked >= 0.887 (7 runs across seeds/backends/rewards, including the collapsing ones),
#: while an untrained policy at the ~27% base rate reaching 0.75 needs 12/16 hits in one window
#: (P ~ 2e-5) -- see the zero-lr control.
_GRPO_CLIMB_MIN = 0.75
#: the climb must come from TRAINING, not from data/sampling drift: peak minus window-0 >= 0.25
#: (observed minimum 0.34).
_GRPO_RISE_MIN = 0.25


def _grpo_window_points(result, backend):
    """One reward point per rollout window from a runner result (see the cadence note above)."""
    rewards = result['rewards']
    assert rewards, f'{backend}/grpo reward path came back empty: {result}'
    if backend == 'dev':
        assert len(rewards) == _GRPO_STEPS, (
            f'dev/grpo expected {_GRPO_STEPS} per-step reward entries (one per optimizer step), got '
            f'{len(rewards)} -- the window cadence changed, re-derive the collapse')
        return [rewards[i] for i in range(0, len(rewards), _DEV_POINTS_PER_WINDOW)]
    assert len(rewards) == _GRPO_WINDOWS, (
        f'legacy/grpo expected {_GRPO_WINDOWS} per-generation-cycle reward entries, got {len(rewards)} '
        '-- the log cadence changed, re-derive the bands')
    return rewards


def test_grpo_reward_path_parity_vs_legacy(tmp_path):
    """Row 3b: dev and legacy drive the SAME reward up the SAME kind of path (not per-step losses).

    GRPO's per-step loss is excluded above (shared advantage code + independent rollouts), but the
    REWARD PATH is a sound cross-pipeline contract: both sides roll out with the SAME colocated vLLM
    setup (temperature 1.0, groups of 4) on the SAME prompt rows (explicit system turn -- see the
    runner's QWEN_SYSTEM note: legacy injects a default system where dev passes rows through, so the
    data pins it) against the SAME single-sourced ORM (``grpo_content_reward``: a phrase check at a MID
    base rate), with matched hyperparameters (beta 0, lr 3e-4 constant, LoRA r8-a32-all-linear). A
    broken learning loop on either side -- advantage sign flipped, reward channel cut, completion mask
    misaligned, weight sync never reaching the sampler -- flatlines its path at the baseline; a healthy
    one climbs to saturation within the window budget.

    Assertions (bands measured on the seed grid above, both sides):
      1. window-0 (base policy) inside the baseline band on BOTH sides -- the reward channel is wired
         and reads the same formula;
      2. PEAK window >= 0.75 on BOTH sides -- each pipeline actually drives the reward up (the peak,
         not the terminal level: at constant lr 3e-4 a trained run can collapse late, but every
         learning run observed -- 7 across seeds/backends -- peaked >= 0.887, while an untrained
         policy needs a P~2e-5 window to touch 0.75);
      3. peak minus window-0 >= 0.25 on BOTH sides -- the climb is trained, not baseline drift.

    Per-window equality is NOT asserted: independent rollouts sample different completions, so the
    paths interleave (grid: legacy s44 saturates at window 1, dev s44 only at window 4). The contract
    is the shape (baseline -> steep rise), stable 7/7 on learning runs.

    Reverse-verify: ``test_grpo_reward_path_control_zero_lr_stays_flat`` runs the same dev cell with
    lr=0 -- the climb assertions go red (terminal stays ~baseline), proving this test's climb gate
    detects a non-learning loop instead of passing on data/sampling drift. History note: the MID base
    rate is load-bearing -- with a ~6% phrase the outcome was seed-chaotic (a 7-run grid produced both
    strong climbs and flatlines on BOTH weight-sync paths); at ~27% the climb is a pipeline property.
    """
    from swift.dev.tests._runners import rl_parity
    data_path = rl_parity._write_grpo_data(str(tmp_path / 'grpo_parity_prompts.jsonl'))
    dev = _grpo_window_points(
        _run_parity_side('dev', 'grpo', data_path, tmp_path, steps=_GRPO_STEPS, lr=_GRPO_LR), 'dev')
    legacy = _grpo_window_points(
        _run_parity_side('legacy', 'grpo', data_path, tmp_path, steps=_GRPO_STEPS, lr=_GRPO_LR),
        'legacy')
    print(f'\ngrpo reward path: dev={dev} legacy={legacy}')

    for backend, points in (('dev', dev), ('legacy', legacy)):
        assert len(points) == _GRPO_WINDOWS, (
            f'{backend}/grpo expected {_GRPO_WINDOWS} window points, got {len(points)}: {points}')
        assert all(x == x and abs(x) != float('inf') for x in points), \
            f'{backend}/grpo non-finite window reward: {points}'
        lo, hi = _GRPO_BASELINE_BAND
        assert lo < points[0] < hi, (
            f'{backend}/grpo window-0 reward {points[0]:.4f} outside the baseline band ({lo}, {hi}) -- '
            'the shared ORM is not wired the same on both sides (missing scorer / key drift / constant '
            'reward all land outside this band)')
        peak = max(points)
        assert peak >= _GRPO_CLIMB_MIN, (
            f'{backend}/grpo peak window reward {peak:.4f} < {_GRPO_CLIMB_MIN} '
            f'(points={points}) -- the reward path never climbs; the learning loop is broken '
            '(advantage sign / reward channel / mask alignment / weight sync)')
        assert peak - points[0] >= _GRPO_RISE_MIN, (
            f'{backend}/grpo climb {peak - points[0]:.4f} < {_GRPO_RISE_MIN} '
            f'(points={points}) -- the peak is baseline drift, not training')


def test_grpo_reward_path_control_zero_lr_stays_flat(tmp_path):
    """Control for the reward-path cell: with lr=0 the same dev run must NOT climb.

    The red side of ``test_grpo_reward_path_parity_vs_legacy``: if the peak gate passed because the
    reward happened to drift up for a data or sampling reason, this control (identical cell, learning
    rate 0 -> every optimizer step is a no-op) would drift too and expose that. Measured: every window
    samples the base policy's ~27% phrase rate, so window means sit ~0.34 and the PEAK stays far below
    the main cell's 0.75 gate. A pipeline bug that kills the gradient (advantage zeroed, channel cut,
    sync frozen) reproduces exactly this shape -- which is why the main cell's peak gate is a real
    detector, not a tautology.
    """
    from swift.dev.tests._runners import rl_parity
    data_path = rl_parity._write_grpo_data(str(tmp_path / 'grpo_parity_prompts.jsonl'))
    points = _grpo_window_points(
        _run_parity_side('dev', 'grpo', data_path, tmp_path, steps=_GRPO_STEPS, lr=0.0), 'dev')
    print(f'\ngrpo reward path (lr=0 control): {points}')
    peak = max(points)
    assert peak < _GRPO_CLIMB_MIN, (
        f'lr=0 control peaked at {peak:.4f} (points={points}) -- the main cell\'s peak gate does not '
        'discriminate learning from drift; re-derive the bands')



@pytest.mark.slow
@pytest.mark.accel(2)
def test_grpo_reward_path_streaming_driver(tmp_path):
    """Row 3c: the StreamingDriver (disaggregated) branch climbs the same reward path, metric wired.

    The colocate cell above pins the cross-pipeline contract on the driver-less plain loop; this cell
    pins the SAME reward-path contract on dev's OTHER online-RL execution path --
    ``sampler_mode='disaggregated'`` rides the StreamingDriver over a separate-GPU sampler (structural:
    ``plan_device_groups`` appends a disjoint SAMPLER_GROUP, so this cell needs 2 cards where the rest
    of the file needs 1; a single visible GPU fails the ResourceManager reservation loudly).

    Beyond the colocate cell's shape gates, two driver-path-specific faces are asserted:

      1. the reward METRIC is wired on this branch too. Probe 2026-10-26: the disaggregated pull loop
         originally left the CompletionRewardMetric empty (product gap); it now records every pulled
         batch through the same ``_record_reward_metric`` helper the colocate loop uses. Red side:
         revert that call and ``rewards`` comes back empty -> this test dies on its first assertion.
      2. the run really rode the driver and really reached the separate sampler:
         ``stream_publishes`` is present on every step and advances (probe: 31 over 32 steps; the gate
         asks for >= 8, one per window), and ``version_span_mean`` stays 0 on every step
         (async_mode='none' pins max_staleness=0, so a non-zero span would mean the collect gate
         leaked an off-policy trajectory).

    Cadence note: unlike colocate (one window aggregate repeated across its 4 optimizer steps), the
    driver streams pulls, so per-step reward entries differ WITHIN a window; the baseline estimator is
    therefore the mean of the first 4 entries (probe: 0.213 / 0.55 -- the same statistic, sample size
    and band as the colocate window-0). The peak/rise gates are the colocate cell's, measured here at
    peak 1.0 on both probe seeds.
    """
    from swift.dev.tests._runners import rl_parity
    data_path = rl_parity._write_grpo_data(str(tmp_path / 'grpo_parity_prompts.jsonl'))
    result = _run_parity_side(
        'dev', 'grpo', data_path, tmp_path, steps=_GRPO_STEPS, lr=_GRPO_LR,
        sampler_mode='disaggregated')
    rewards = result['rewards']
    assert rewards, (
        'dev/grpo disaggregated reward path came back empty: the StreamingDriver pull loop does not '
        'feed the CompletionRewardMetric (the _record_reward_metric call is missing on that branch)')
    assert len(rewards) == _GRPO_STEPS, (
        f'dev/grpo disaggregated expected {_GRPO_STEPS} per-step reward entries, got {len(rewards)}: '
        f'{rewards} -- the pull cadence changed, re-derive the baseline estimator')
    assert all(x == x and abs(x) != float('inf') for x in rewards), \
        f'dev/grpo disaggregated non-finite reward entry: {rewards}'

    baseline = sum(rewards[:4]) / 4
    lo, hi = _GRPO_BASELINE_BAND
    assert lo < baseline < hi, (
        f'dev/grpo disaggregated baseline (mean of first 4 steps) {baseline:.4f} outside the band '
        f'({lo}, {hi}) -- the shared ORM is not wired the same on the driver path')
    peak = max(rewards)
    assert peak >= _GRPO_CLIMB_MIN, (
        f'dev/grpo disaggregated peak reward {peak:.4f} < {_GRPO_CLIMB_MIN} (rewards={rewards}) -- '
        'the driver-path learning loop is broken (advantage sign / reward channel / mask alignment '
        '/ publish never reaching the sampler)')
    assert peak - baseline >= _GRPO_RISE_MIN, (
        f'dev/grpo disaggregated climb {peak - baseline:.4f} < {_GRPO_RISE_MIN} '
        f'(rewards={rewards}) -- the peak is baseline drift, not training')

    driver = result.get('driver') or {}
    publishes, spans = driver.get('publishes'), driver.get('spans')
    assert publishes and len(publishes) == _GRPO_STEPS, (
        f'stream_publishes missing or not per-step ({publishes}): the run did not ride the '
        'StreamingDriver face -- expected the driver-only bookkeeping on every step')
    assert publishes[-1] >= _GRPO_WINDOWS, (
        f'stream_publishes never advanced ({publishes}): the trained policy never reached the '
        'disaggregated sampler, so it rolled out on the initially-loaded weights the whole time -- '
        'the climb above would then be an artifact, not learning')
    assert spans and all(s == 0 for s in spans), (
        f'version_span_mean non-zero on a staleness-0 run ({spans}): the collect gate admitted an '
        'off-policy trajectory (async_mode=none pins max_staleness=0)')


def test_full_param_parity_excluded_dev_unsupported():
    """GAP(exclusion): a full-parameter parity cell has no dev side.

    dev rejects ``tuner='full'`` outright (``NotImplementedError: tuner='full' is not supported by dev;
    supported: lora, adalora, trainable_tokens``), confirmed by a reference probe whose dev full-param DPO
    side raised exactly that while the legacy side trained fine. With no dev full-param path there is
    nothing to compare, so the cell is excluded rather than stubbed. LoRA parity -- the tuning mode dev
    actually ships for RL -- is covered by the cpo/orpo cell above.
    """
    pytest.skip("dev raises NotImplementedError for tuner='full' (only lora/adalora/trainable_tokens are "
                'implemented), so a full-parameter legacy-vs-dev parity cell has no dev side to compare. '
                'LoRA parity is covered by the cpo/orpo cell.')


def test_reward_model_parity_excluded_random_head():
    """GAP(exclusion): rm / seq_cls loss parity vs legacy is not attainable -- the head init is independent.

    A reward model rides a freshly-initialized num_labels=1 scalar head. The legacy and dev pipelines consume
    different amounts of RNG before that head init (different model-load / template / dataset-prep order), so
    the heads -- and therefore the step-1 loss, which is init-dominated rather than data-dominated -- cannot
    be made to agree; no band short of "anything goes" would pass, which would be vacuous. This is the SAME
    root cause the SFT parity grid excludes seq_cls/reranker for (see test_parity_grid.py). dev's rm training
    is proven end to end in test_algorithms_e2e and test_backends_e2e (both backends, task_type='seq_cls'
    force-loaded into args.json); loss parity against legacy is excluded, not silently omitted.
    """
    pytest.skip('rm/seq_cls has a randomly-initialized scalar reward head whose init RNG-stream position '
                'differs between the legacy and dev pipelines, so step-1 loss is init-dominated and not '
                'comparable (same root cause the SFT parity grid excludes seq_cls/reranker for). Not a dev '
                'defect; test_algorithms_e2e/test_backends_e2e cover dev rm training.')
