# Copyright (c) ModelScope Contributors. All rights reserved.
"""Dimension ⑧ -- loss regression vs legacy: dev's RL objective must restore ``swift.rlhf_main``'s loss.

``test_backends_e2e`` proves the dev *transformers* and dev *megatron* backends agree, and its docstring
explicitly defers DEEP per-step parity to here ("step 2 drifts apart ... which is why deep per-step parity
is anchored against legacy in ``test_parity_legacy``, not here"). This file is that anchor: it drives the
REAL legacy pipeline (``swift.rlhf_main`` -> HF/RLHFArguments -> TRL ``<TYPE>Trainer``) and the REAL dev
pipeline (``run_rl`` -> Ray ``PreferenceLoop`` -> twinkle loss) on the SAME cached Qwen2.5-0.5B, the SAME
preference rows and MATCHED deterministic hyperparameters, and asserts dev restores legacy's per-step loss
trajectory. The contract is an INDEPENDENT oracle (legacy's own trainer), not a self-consistency round-trip.

Mechanism (RL_PLAN section 8: isolate a subprocess per side and compare per-step loss, mirroring
``megatron_sft.py``): each side runs in its OWN plain-python subprocess (``_runners/rl_parity.py``), because
legacy builds a TRL trainer on the HF Trainer/accelerate stack while dev spins up a Ray session + the twinkle
runtime, and the two cannot safely share one interpreter. Offline preference training is single-card with no
rollout, so -- unlike ``megatron_sft.py`` -- this needs no torchrun, just one process per side.

Why the REFERENCE-FREE variants (cpo/orpo) carry the parity oracle, not DPO (bands below are MEASURED on a
reference probe, deterministic across two runs -- 0.0 self-drift -- not guessed):

  - **cpo/orpo step-1 is DATA-DEPENDENT and restores to ~1e-7 rel.** These objectives are reference-free:
    with the identity adapter (``lora_B``=0 at init) the step-1 loss is the BASE model's own
    NLL / odds-ratio contrast on chosen-vs-rejected (probe: cpo 9.2321, orpo 8.8866 -- far from ln2), a
    deterministic function of the weights + which tokens are scored. So step-1 is a STRONG discriminant: a
    label shift, a wrong completion mask, or an encode bug moves it by O(10%). Measured step-1 rel gap:
    cpo 4.7e-7, orpo 4.9e-8 (dominated by dev's 5-decimal loss logging); band 1e-3 keeps ~2000x margin.
  - **later steps are BOUNDED, not exact** (probe worst rel: cpo 1.02e-2, orpo 4.57e-2). Past step-1 the
    update consumes the LoRA-A init, which legacy peft and dev twinkle draw at DIFFERENT RNG-stream
    positions, so the two runs are valid-but-distinct trajectories -- the same reason the SFT parity grid
    bounds its later steps. Bands: cpo 0.05 (~5x margin), orpo 0.10 (~2x margin). The step-1 anchor, not
    the later band, is the load-bearing gate.

Honest exclusions (each asserted as a body-level ``pytest.skip`` with its probe evidence, mirroring the SFT
``test_parity_grid`` exclusions -- NOT a vacuous "it trained" test, and NOT silently omitted):

| # | cell | contract / why excluded | failure it kills (or why unattainable) |
|---|------|-------------------------|----------------------------------------|
| 1 | cpo, orpo per-step loss | dev restores legacy: step-1 rel<1e-3 AND step-1 far from ln2 (data-dependent); later rel<band + both decreasing | dev loss formula / label shift / completion mask / encode drifts from legacy |
| 2 | dpo deep parity | EXCLUDED: step-1 is the degenerate ln2 (policy==ref cancels to 0 for ANY masking -- cannot catch a shift bug; already anchored in test_algorithms_e2e / test_backends_e2e), and step2+ is chaotic (probe rel 7.3% -> 43.5% -> 41.7%) because the adapter-dominated loss -> 0 divides the gap | mistaking a self-cancelling ln2 for parity signal |
| 3 | grpo advantage/clip/KL | EXCLUDED: dev grpo.py and legacy grpo_trainer.py BOTH import ``rl_core.advantage.compute_advantages`` (shared code -> comparing it is a self-consistency round-trip); the policy loss is already unit-anchored vs a hand-computed PPO-clip oracle in test_recipes.py; and each side runs an independent temperature>0 vLLM rollout so the completions -- and every per-step loss -- differ by construction | a fake "parity" that only proves shared code equals itself |
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

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

_LN2 = math.log(2.0)
_STEPS = 4
_LR = 1e-4
# step-1 is the load-bearing anchor: identity adapter (lora_B=0) -> pure base-model forward, independent of
# the LoRA-A init RNG stream and of any optimizer update. Probe-measured rel ~5e-7 (dev logs loss at 5
# decimals, which is the floor); 1e-3 keeps ~2000x margin yet still trips on a real shift/mask/encode drift.
_STEP1_BAND = 1e-3

#: cell -> later-step relative band (probe-measured worst + margin; see module docstring). Deterministic
#: across two probe runs (0.0 self-drift), so the margin guards against library-version drift, not flakiness.
_LATER_BAND = {'cpo': 0.05, 'orpo': 0.10}


def _run_parity_side(backend, rlhf_type, data_path, tmp_path, *, steps=_STEPS, lr=_LR, timeout=900):
    """Run ONE side (legacy|dev) of the parity pair in its own plain-python subprocess; return its losses.

    Plain python, not torchrun: offline preference training is single-card with no rollout, so each side is
    one process (contrast ``megatron_sft.py``, which needs torchrun for world_size>=2). Isolation is still
    required -- see the module docstring. ``Runners.launch`` bounds the run and kills the whole process
    group on timeout; a crashed side leaves no result file, which is turned into an AssertionError carrying
    the subprocess stdout/stderr tail (the loss itself is read back from disk, never trusted in-memory).
    """
    runner = Runners.path('rl_parity')
    result_path = str(tmp_path / f'{backend}_{rlhf_type}.json')
    cmd = [
        sys.executable, runner,
        '--backend', backend,
        '--rlhf_type', rlhf_type,
        '--data', data_path,
        '--out', result_path,
        '--out_dir', str(tmp_path / f'{backend}_{rlhf_type}_out'),
        '--steps', str(steps),
        '--lr', str(lr),
        '--tuner', 'lora',
    ]
    proc = Runners.launch(cmd, timeout=timeout)
    if not os.path.exists(result_path):
        raise AssertionError(
            f'{backend}/{rlhf_type} parity runner produced no result (exit={proc.returncode}). '
            f'stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-3000:]}')
    with open(result_path) as f:
        return json.load(f)['losses']


@pytest.mark.parametrize('rlhf_type', sorted(_LATER_BAND))
def test_reference_free_loss_parity_vs_legacy(rlhf_type, rl_data, tmp_path):
    """Row 1: dev's ``<cpo|orpo>`` per-step loss restores legacy ``swift.rlhf_main``'s (same model/data/hparams).

    Both sides ride the SAME cached Qwen2.5-0.5B + the SAME 8 preference rows + matched deterministic
    hyperparameters (constant LR / no warmup / seed 42 / shuffle off / beta 0.1 / rpo_alpha 0 / LoRA
    r8-a32-all-linear / bf16), each in its own subprocess. The reference-free objective makes step-1
    data-dependent (the base model's NLL / odds contrast), so step-1 is a strong discriminant for a
    shift/mask/encode bug -- the exact signal DPO's degenerate ln2 step-1 cannot provide.

    Reverse-verify: flip the sign of dev's chosen/rejected log-probability difference (or shift the
    completion labels by one token) in the twinkle loss and step-1 moves off legacy's ~9.23/~8.89 by >>1e-3
    -> the step-1 rel assertion goes red; restore it -> green. Dropping the ``not ln2`` guard would let a
    silent degradation to a policy==ref cancel (both sides -> ln2) pass as a vacuous "parity".
    """
    data_path = rl_data.preference(n=8)
    dev = _run_parity_side('dev', rlhf_type, data_path, tmp_path)
    legacy = _run_parity_side('legacy', rlhf_type, data_path, tmp_path)
    print(f'\n{rlhf_type} dev-vs-legacy parity: dev={dev} legacy={legacy}')

    assert len(dev) == len(legacy) == _STEPS, \
        f'{rlhf_type}: expected {_STEPS} per-step losses, dev={dev} legacy={legacy}'
    # loss == loss is the NaN test (NaN != itself); not a typo.
    assert all(x == x and abs(x) != float('inf') for x in dev + legacy), \
        f'{rlhf_type}: non-finite loss dev={dev} legacy={legacy}'

    # Non-degeneracy guard FIRST: the legacy reference step-1 must be data-dependent, not the ln2 cancel.
    # If it sat at ln2 this cell would carry no more signal than DPO's step-1 (both sides self-cancel), and
    # the rel check below could pass vacuously -- so prove the reference-free objective is really computed.
    assert abs(legacy[0] - _LN2) > 1.0, (
        f'{rlhf_type}: legacy step-1 loss {legacy[0]:.5f} ~= ln2 {_LN2:.5f} -- the reference-free objective '
        'degenerated to a policy==ref cancel (data-dependent NLL/odds term missing), so this is not a real '
        'parity signal')

    # step-1: the load-bearing, RNG-independent anchor (identity adapter -> pure base-model forward).
    rel1 = abs(dev[0] - legacy[0]) / max(abs(legacy[0]), 1e-8)
    assert rel1 < _STEP1_BAND, (
        f'{rlhf_type} step-1 dev-vs-legacy mismatch: dev={dev[0]:.6f} legacy={legacy[0]:.6f} rel={rel1:.2e} '
        f'(band {_STEP1_BAND}) -- the reference-free objective (logp-diff sign / completion mask / label '
        'shift / encode) diverges from legacy')

    # later steps: bounded trajectory agreement (the LoRA-A init RNG stream differs between pipelines, so
    # these are two valid-but-distinct runs, not an exact contract) + monotone decrease (a sign error climbs).
    later_band = _LATER_BAND[rlhf_type]
    for i in range(1, _STEPS):
        assert dev[i] < dev[i - 1] and legacy[i] < legacy[i - 1], (
            f'{rlhf_type}: loss not strictly decreasing (a sign error would climb): dev={dev} legacy={legacy}')
        rel = abs(dev[i] - legacy[i]) / max(abs(legacy[i]), 1e-8)
        assert rel < later_band, (
            f'{rlhf_type} step {i + 1} dev-vs-legacy divergence: dev={dev[i]:.6f} legacy={legacy[i]:.6f} '
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
    """GAP(exclusion): GRPO advantage/clip/KL parity vs legacy is not an independent-oracle comparison.

    RL_PLAN section 5 Row 2 asks for "dev vs legacy grpo_algorithm, advantage/clip/KL numerically equal".
    That is unattainable as a NON-circular test for three reasons: (1) dev ``recipe/grpo.py`` and legacy
    ``grpo_trainer.py`` BOTH ``import rl_core.advantage.compute_advantages`` -- the advantage is SHARED code,
    so comparing dev's advantage to legacy's is a self-consistency round-trip (the skill's forbidden
    pattern), not an oracle; (2) the GRPO policy loss (clip/KL) is already unit-anchored against a
    hand-computed PPO-clip oracle in test_recipes.py, which is the independent ground truth; (3) a full e2e
    GRPO loss comparison is incomparable because each side drives an independent temperature>0 vLLM rollout,
    so the sampled completions -- and therefore every per-step loss -- differ by construction (the same
    non-determinism test_backends_e2e documents for its megatron GRPO run). GRPO training itself is proven
    e2e in test_algorithms_e2e / test_backends_e2e / test_router_replay_e2e.
    """
    pytest.skip('GRPO advantage is SHARED code (dev grpo.py and legacy grpo_trainer.py both import '
                'rl_core.advantage.compute_advantages), so a dev-vs-legacy advantage comparison is a '
                'self-consistency round-trip, not an oracle; the clip/KL policy loss is already unit-anchored '
                'vs a hand-computed PPO-clip oracle in test_recipes.py; and independent temperature>0 vLLM '
                'rollouts make per-step GRPO loss incomparable by construction.')


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
