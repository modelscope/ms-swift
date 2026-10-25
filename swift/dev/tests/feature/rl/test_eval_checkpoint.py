# Copyright (c) ModelScope Contributors. All rights reserved.
"""Dimension ⑨ -- eval / checkpoint-storage contracts (RL_PLAN section 5 test_eval_checkpoint).

The RL loops own three storage/eval behaviours that a green "it trained" test never touches: whether a
requested periodic evaluation is honoured or refused, whether a resumed run continues the SAME trajectory an
uninterrupted run would have, and whether ``save_total_limit`` rotates old checkpoints without eating the one
just written. Each is asserted against an INDEPENDENT oracle (the fail-loudly guard's message, an
uninterrupted control run, the on-disk checkpoint set read back after the run), not a self-consistency
round-trip, and each is reverse-verifiable (delete the guard / break the rotation -> red).

Corner cases (RL_PLAN section 5 rows; row 4 belongs to ``test_infeasible_combos`` and is not duplicated):

| # | contract | input | expected | failure it kills |
|---|----------|-------|----------|------------------|
| 1 | offline RL periodic eval is HONOURED (val-loss), not refused | dpo + ``split_dataset_ratio=0.5`` (eval split + derived ``eval_steps``) | the run trains AND the inlined ``evaluate_loss`` returns a finite ``eval_loss`` (spied at ``run_dpo``'s bound name) | a configured eval split silently ignored (``run_dpo`` never evaluating), or the old W24 refusal regressing |
| 1b | forward_only val-loss eval / megatron ``calculate_loss`` guard | N/A for RL | -- | grpo/ppo call no ``evaluate``; the offline preference val-loss path is row 1. Standalone ``swift eval`` is covered by ``test_eval_lora_example_e2e`` -- not re-tested here |
| 2a | resume continues the trajectory | dpo: save@step2 -> resume -> run to 4 | step counter continues at 3 AND losses match an uninterrupted control's tail within a measured band | a resume that ignores the checkpoint step count, or restores wrong weights / optimizer / data position |
| 2b | PPO policy+value step alignment | ``_check_policy_value_step_alignment(step=5, value_state=3)`` | ``ValueError``; matched (5,5) and ``value_state=None`` accepted | resuming a half-written PPO checkpoint whose policy and value sit at different completed steps |
| 3 | ``save_total_limit`` rotation + current protection | dpo ``save_steps=1``, 3 steps + final; limit in {None,1,2} | None -> all 4 kept; 1 -> {final}; 2 -> {checkpoint-3, final}; the just-saved current ALWAYS survives | over-deleting the current save, or under-rotating (limit ignored) |

Rows 1 / 2a / 3 drive the real ``run_rl`` lifecycle (a real Ray session, a real LoRA policy on the tiny
Qwen2.5, real checkpoints read back from disk), gated ``slow`` + ``accel(1)``. Row 2b exercises the extracted
pure guard ``_check_policy_value_step_alignment`` (run_ppo.py) directly -- it reads only the completed-step
count, so it needs no GPU -- the real atomic policy+value checkpoint it protects is proven e2e in
``test_algorithms_e2e``.

Run: ``CUDA_VISIBLE_DEVICES=<c> pytest swift/dev/tests/feature/rl/test_eval_checkpoint.py -m slow``.
"""
import os
from contextlib import contextmanager

import pytest

from swift.dev.tests.feature.rl.conftest import rl_configs, run_rl

#: Resume-vs-control relative band. MEASURED on the fixed-seed tiny run: correct resume reproduces the
#: control tail with rel gaps [2.1e-3 (step 3), 1.1e-2 (step 4)] -- a small resume perturbation that DPO's
#: chaotic sensitivity then compounds step over step, so it is tighter than the dev-vs-legacy parity cell
#: (43%) but not bit-exact. Band sits ~2.7x above the observed worst gap to absorb run-to-run jitter while
#: still failing hard on a wrong restore (fresh weights sit at ln2; a dropped data-skip trains on the wrong
#: preference rows), both of which push the gap well past this.
_RESUME_BAND = 3e-2


@contextmanager
def _captured_eval_results():
    """Spy the offline preference run's periodic val-loss eval, recording each REAL result dict.

    ``run_dpo``'s flat offline layer no longer builds a capturable ``Evaluator``: eval is inlined as an
    ``evaluate_loss`` call whose result feeds a discarded ``eval_history`` closure (``_run_supervised``
    returns only the training history). So this wraps the module-level ``evaluate_loss`` that ``run_dpo``
    imported at load time, recording its return while calling the REAL function through -- it observes the
    actual eval output (a finite ``eval_loss`` produced by ``forward_only`` + ``calculate_metric`` over the
    val split, each batch encoded through the SAME ``PreferenceBatchEncoder.__call__`` seam training uses),
    not a stub. Patching the recipe module's OWN bound name is what makes the spy land: ``run_dpo`` calls
    ``evaluate_loss`` from its namespace, so replacing ``swift.dev.recipe.utils.evaluate_loss`` alone would
    be missed. ``importlib.import_module`` (not ``import ... as``) fetches the MODULE from ``sys.modules``:
    ``swift.dev.recipe.rlhf.__init__`` rebinds the name ``run_dpo`` to the FUNCTION, so an attribute-style
    import would hand back the function and miss the module's bound ``evaluate_loss``.
    """
    import importlib
    run_dpo_mod = importlib.import_module('swift.dev.recipe.rlhf.run_dpo')
    recorded = []
    original = run_dpo_mod.evaluate_loss

    def _spy(*args, **kwargs):
        result = original(*args, **kwargs)
        if result is not None:
            recorded.append(result)
        return result

    run_dpo_mod.evaluate_loss = _spy
    try:
        yield recorded
    finally:
        run_dpo_mod.evaluate_loss = original


@pytest.mark.slow
@pytest.mark.accel(1)
def test_offline_preference_honours_periodic_eval(tiny_qwen2_5, rl_data, tmp_path):
    """Row 1: an offline preference run HONOURS a periodic val-loss eval instead of refusing it.

    ``split_dataset_ratio>0`` carves off a val split, so ``_derive_eval_schedule`` inherits
    ``eval_strategy='steps'`` / ``eval_steps=save_steps`` and ``build_dataset`` returns a non-None
    ``eval_dataloader``. ``run_dpo``'s flat offline loop then runs the whole cadence INCLUDING periodic
    eval inline: its ``evaluate`` closure calls :func:`~swift.dev.recipe.utils.evaluate_loss` over the eval
    dataloader, encoding each eval batch through the SAME ``PreferenceBatchEncoder.__call__`` seam training
    uses (interleave chosen/rejected + one reference ``forward_only``), so the DPO loss is computable on the
    eval split too. The old W24 ``raise NotImplementedError`` guard is gone (record 决策二). The assertion
    reads a finite ``eval_loss`` off the spied ``evaluate_loss`` results -- a real number produced by
    ``forward_only`` + ``calculate_metric``, not merely a green run.

    Reverse-verify: reinstate a run_dpo refusal (or drop the ``split_dataset_ratio`` eval wiring) and the
    run never calls ``evaluate_loss`` -> the spy records nothing -> the ``eval_results`` assertion goes red;
    or make ``evaluate_loss`` feed the raw eval rows to ``forward_only`` instead of ``encoder(unit,
    owns_ga=True)`` (``PreferenceBatchEncoder.__call__``) and the DPO eval loss is absent/non-finite -> the
    finite-``eval_loss`` assertion goes red. Remove either sabotage -> green.
    """
    configs = rl_configs(
        rl_type='dpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.preference(n=8),
        out_dir=str(tmp_path / 'out_eval_honour'),
        max_steps=2,
        dataset_over={'split_dataset_ratio': 0.5},
        ckpt_over={'save_steps': 1},
    )
    with _captured_eval_results() as eval_results:
        history = run_rl(configs)

    assert history, 'the preference run produced no optimizer steps (it should train, not refuse)'
    assert eval_results, ('periodic eval never ran: evaluate_loss was never called, so the val split was '
                          'silently dropped instead of scored (the offline eval path regressed, or a W24-style '
                          'refusal was reinstated)')
    eval_losses = [r['eval_loss'] for r in eval_results if 'eval_loss' in r]
    assert eval_losses, f'eval ran but recorded no eval_loss: {eval_results}'
    assert all(x == x and abs(x) != float('inf') for x in eval_losses), \
        f'non-finite eval_loss: {eval_losses} (the eval batch was not encoded through PreferenceBatchEncoder.__call__)'


@pytest.mark.slow
@pytest.mark.accel(1)
def test_offline_preference_resume_continues_trajectory(tiny_qwen2_5, rl_data, tmp_path):
    """Row 2a: resuming from a checkpoint CONTINUES the trajectory an uninterrupted run would have taken.

    Three real ``run_rl`` DPO runs on the same tiny policy, the same preference rows and matched
    deterministic hyperparameters (seed / data_seed fixed, shuffle off):

    * Phase A trains 2 steps and persists ``checkpoint-2`` (``save_steps=2`` lands a periodic save there).
    * Control trains 4 steps uninterrupted -- the ORACLE for what steps 3 and 4 must be.
    * Phase B resumes from ``checkpoint-2`` and trains to 4.

    Because Phase A's first two steps are bit-identical to the control's (same init, same data, same seed),
    ``checkpoint-2`` IS the control's post-step-2 state; a correct resume therefore reproduces steps 3-4.
    The load-bearing assertion is the step counter: a resume that ignored the checkpoint's ``cur_step``
    would restart at step 1 and record ``[1,2,3,4]``, not ``[3,4]``. The trajectory match then catches a
    resume that restored the wrong weights, optimizer moments, RNG or dataloader position. This is a
    dev-vs-dev comparison (no LoRA-A init gap, unlike the dev-vs-legacy DPO parity cell), so the band is far
    tighter than that cell's chaotic 43%; it is MEASURED (correct resume: rel gaps [2.1e-3, 1.1e-2], a small
    perturbation DPO compounds step over step), not assumed bit-exact.

    Reverse-verify: point ``resume_from_checkpoint`` at ``checkpoint-2`` but drop the loop's step restore
    (``cur_step -> 0``) and the resumed run restarts at step 1, recording ``[1,2,3,4]`` instead of ``[3,4]``
    -> the load-bearing step-counter assertion goes red; restore -> green. The loss band is the SECONDARY
    guard, against a catastrophic weight restore (NaN / wildly wrong weights): this tiny DPO run's losses
    all sit within ~2.6% of ln2 (0.675-0.693), so an absolute band cannot discriminate a subtle
    data-position slip -- the step counter is the strong contract, the band catches a gross restore failure.
    """
    data = rl_data.preference(n=8)

    configs_a = rl_configs(
        rl_type='dpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=data,
        out_dir=str(tmp_path / 'phase_a'),
        max_steps=2,
        ckpt_over={'save_steps': 2},
    )
    run_rl(configs_a)
    ckpt2 = os.path.join(configs_a['checkpoint_config'].output_dir, 'checkpoint-2')
    assert os.path.isdir(ckpt2), f'Phase A did not persist the resume checkpoint at {ckpt2}'

    configs_c = rl_configs(
        rl_type='dpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=data,
        out_dir=str(tmp_path / 'control'),
        max_steps=4,
        ckpt_over={'save_steps': 4},
    )
    control = run_rl(configs_c)

    configs_b = rl_configs(
        rl_type='dpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=data,
        out_dir=str(tmp_path / 'phase_b'),
        max_steps=4,
        ckpt_over={'save_steps': 4, 'resume_from_checkpoint': ckpt2},
    )
    resumed = run_rl(configs_b)

    assert len(control) == 4, f'control produced {len(control)} optimizer steps, expected 4'
    # Load-bearing: the resumed run CONTINUES the optimizer-step counter (does not restart at 1).
    assert [r['step'] for r in resumed] == [3, 4], \
        f'resume did not continue the step counter: {[r["step"] for r in resumed]} (expected [3, 4])'

    dev = [r['loss'] for r in resumed]
    ref = [r['loss'] for r in control[2:]]
    # loss == loss is the NaN test (NaN != itself); not a typo.
    assert all(x == x and abs(x) != float('inf') for x in dev + ref), \
        f'non-finite loss: resumed={dev} control[2:]={ref}'
    gaps = [abs(d - c) / max(abs(c), 1e-8) for d, c in zip(dev, ref)]
    print(f'\nresume trajectory: resumed={dev} control[2:]={ref} rel_gaps={gaps}')
    for i, rel in enumerate(gaps):
        assert rel < _RESUME_BAND, (
            f'resumed step {i + 3} loss {dev[i]:.6f} diverges from the uninterrupted control {ref[i]:.6f} '
            f'(rel {rel:.2e} > band {_RESUME_BAND}) -- resume restored wrong weights / optimizer / RNG / '
            'dataloader position, so the continuation is not the trajectory an unbroken run would take')


def test_ppo_resume_rejects_policy_value_step_mismatch():
    """Row 2b: PPO resume refuses a policy/value checkpoint pair at DIFFERENT completed-step counts.

    PPO saves the policy and the critic (value model) as two components under one checkpoint dir. A crash
    between the two writes leaves them a step apart; resuming such a pair would train a policy whose rollout
    history no longer matches its critic's value baseline. The flat run body restores ``global_step`` from
    the policy state (``resume_training``), reloads the critic state, and cross-checks the two through the
    pure guard ``_check_policy_value_step_alignment`` (run_ppo.py) -- so a direct call exercises the exact
    invariant with no GPU / Ray build (the same faithful-unit pattern ``test_infeasible_combos`` uses for
    ``MegatronModel``). The real atomic policy+value save this protects is proven e2e in
    ``test_algorithms_e2e::test_ppo_e2e_policy_and_value_atomic``.

    Reverse-verify: delete the mismatch check inside ``_check_policy_value_step_alignment`` and the (5, 3)
    pair is accepted -> ``pytest.raises`` goes red; restore -> refused again.
    """
    from swift.dev.recipe.ppo.run_ppo import _check_policy_value_step_alignment

    # A pair a step apart (policy at 5, critic at 3) is refused.
    with pytest.raises(ValueError, match='different completed-step counts'):
        _check_policy_value_step_alignment(5, {'consumed_train_samples': 3})

    # An aligned pair is accepted (the guard is a no-op); it validates only, never mutates.
    _check_policy_value_step_alignment(5, {'consumed_train_samples': 5})

    # value_state=None is the single-component resume (no critic to cross-check): accepted.
    _check_policy_value_step_alignment(7, None)

    # A critic state missing the counter reads as 0, so it mismatches a non-zero policy step (refused) --
    # the safe direction: an absent count is never treated as aligned with real progress.
    with pytest.raises(ValueError, match='different completed-step counts'):
        _check_policy_value_step_alignment(4, {})

    # ... and a missing counter at a zero policy step is genuinely aligned (both 0): accepted.
    _check_policy_value_step_alignment(0, {})


@pytest.mark.slow
@pytest.mark.accel(1)
@pytest.mark.parametrize('limit,expected', [
    (None, {'checkpoint-1', 'checkpoint-2', 'checkpoint-3', 'checkpoint-final'}),
    (1, {'checkpoint-final'}),
    (2, {'checkpoint-3', 'checkpoint-final'}),
],
                         ids=['none', 'one', 'two'])
def test_save_total_limit_rotates_and_protects_current(limit, expected, tiny_qwen2_5, rl_data, tmp_path):
    """Row 3: ``save_total_limit`` keeps the newest N checkpoints and NEVER deletes the one just written.

    A DPO run with ``save_steps=1`` over 3 steps writes ``checkpoint-1/-2/-3`` then ``checkpoint-final``;
    every save calls ``rotate_checkpoints`` (twinkle model/base.py:34), which sorts the checkpoint dirs by
    ``(is_current, mtime, name)`` -- so the just-saved CURRENT dir always sorts last and survives -- and
    deletes ``[:-limit]`` (the oldest). Read back from the RESOLVED ``output_dir`` (the add_version subdir
    ``process_and_validate_configs`` mutates it to), the surviving set is:

    * ``None`` -> rotation is a no-op, all four survive.
    * ``1``    -> only the current ``checkpoint-final`` survives (every older one pruned as it was written).
    * ``2``    -> the newest numbered ``checkpoint-3`` plus ``checkpoint-final``.

    Reverse-verify: make ``rotate_checkpoints`` return early (no-op) and the limit=1/2 runs keep all four
    dirs -> red; or invert the ``is_current`` sort key so the current sorts first and gets pruned -> limit=1
    deletes ``checkpoint-final`` -> red. Restore either -> green.
    """
    configs = rl_configs(
        rl_type='dpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.preference(n=8),
        out_dir=str(tmp_path / f'out_limit_{limit}'),
        max_steps=3,
        ckpt_over={'save_steps': 1, 'save_total_limit': limit},
    )
    run_rl(configs)

    out_dir = configs['checkpoint_config'].output_dir
    survived = {d for d in os.listdir(out_dir) if d.startswith('checkpoint-')}
    assert survived == expected, (
        f'save_total_limit={limit}: surviving checkpoints {sorted(survived)} != expected {sorted(expected)} '
        '-- rotation either ignored the limit or deleted the wrong (newest/current) checkpoint')
    # The just-saved current checkpoint must ALWAYS survive, whatever the limit.
    final = os.path.join(out_dir, 'checkpoint-final')
    assert os.path.isdir(final), f'save_total_limit={limit}: the current checkpoint-final was pruned'
    assert any(f.endswith('.safetensors') for f in os.listdir(final)), \
        f'save_total_limit={limit}: checkpoint-final has no weights: {sorted(os.listdir(final))}'
