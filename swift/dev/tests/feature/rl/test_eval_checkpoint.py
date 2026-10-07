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
| 1 | offline RL periodic eval is REFUSED, not silently dropped | dpo + ``split_dataset_ratio=0.5`` (eval split + derived ``eval_steps``) | ``NotImplementedError`` raised after the model is built but BEFORE any optimizer step | a configured eval split silently ignored (``PreferenceLoop.fit`` never evaluates) |
| 1b | forward_only val-loss eval / megatron ``calculate_loss`` guard | N/A for RL | -- | RL never reaches the SFT ``TrainLoop`` eval path (offline rejects pre-fit; grpo/ppo call no ``evaluate``). Standalone ``swift eval`` is covered by ``test_eval_lora_example_e2e`` -- not re-tested here |
| 2a | resume continues the trajectory | dpo: save@step2 -> resume -> run to 4 | step counter continues at 3 AND losses match an uninterrupted control's tail within a measured band | a resume that ignores the checkpoint step count, or restores wrong weights / optimizer / data position |
| 2b | PPO policy+value step alignment | ``PPOLoop.resume(policy=5, value=3)`` | ``ValueError``; matched (5,5) and ``value_state=None`` accepted | resuming a half-written PPO checkpoint whose policy and value sit at different completed steps |
| 3 | ``save_total_limit`` rotation + current protection | dpo ``save_steps=1``, 3 steps + final; limit in {None,1,2} | None -> all 4 kept; 1 -> {final}; 2 -> {checkpoint-3, final}; the just-saved current ALWAYS survives | over-deleting the current save, or under-rotating (limit ignored) |

Rows 1 / 2a / 3 drive the real ``run_rl`` lifecycle (a real Ray session, a real LoRA policy on the tiny
Qwen2.5, real checkpoints read back from disk), gated ``slow`` + ``accel(1)``. Row 2b exercises the bare
``PPOLoop.resume`` guard with ``__new__`` (it reads/sets only ``global_step``), so it needs no GPU -- the real
atomic policy+value checkpoint it protects is proven e2e in ``test_algorithms_e2e``.

Run: ``CUDA_VISIBLE_DEVICES=<c> pytest swift/dev/tests/feature/rl/test_eval_checkpoint.py -m slow``.
"""
import os

import pytest

from swift.dev.tests.feature.rl.conftest import rl_configs, run_rl

#: Resume-vs-control relative band. MEASURED on the fixed-seed tiny run: correct resume reproduces the
#: control tail with rel gaps [2.1e-3 (step 3), 1.1e-2 (step 4)] -- a small resume perturbation that DPO's
#: chaotic sensitivity then compounds step over step, so it is tighter than the dev-vs-legacy parity cell
#: (43%) but not bit-exact. Band sits ~2.7x above the observed worst gap to absorb run-to-run jitter while
#: still failing hard on a wrong restore (fresh weights sit at ln2; a dropped data-skip trains on the wrong
#: preference rows), both of which push the gap well past this.
_RESUME_BAND = 3e-2


@pytest.mark.slow
@pytest.mark.accel(1)
def test_offline_preference_rejects_periodic_eval(tiny_qwen2_5, rl_data, tmp_path):
    """Row 1: an offline preference run REFUSES a periodic eval split instead of silently ignoring it.

    ``split_dataset_ratio>0`` carves off a val split, so ``_derive_eval_schedule`` inherits
    ``eval_strategy='steps'`` / ``eval_steps=save_steps`` and ``build_dataset`` returns a non-None
    ``eval_dataloader``. But ``PreferenceLoop.fit`` never evaluates -- it stores the eval loader and drops
    it -- so ``run_dpo`` fails loudly (run_dpo.py:173-177) rather than train without the eval that was asked
    for. The guard sits AFTER ``build_model`` and BEFORE the loop's ``fit``, so this exercises the real
    GPU-built model on the exact refusal path ``swift rlhf`` hits, with no optimizer step run.

    Reverse-verify: delete the run_dpo.py:173-177 guard and the split is silently ignored -- the run trains
    and returns a history instead of raising, so ``pytest.raises`` goes red; restore it -> refused again.
    """
    configs = rl_configs(
        rlhf_type='dpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.preference(n=8),
        out_dir=str(tmp_path / 'out_eval_reject'),
        max_steps=1,
        dataset_over={'split_dataset_ratio': 0.5},
    )
    with pytest.raises(NotImplementedError, match='Periodic evaluation is not implemented'):
        run_rl(configs)


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
        rlhf_type='dpo',
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
        rlhf_type='dpo',
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
        rlhf_type='dpo',
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

    PPO saves the policy and the critic (value model) as two components under one checkpoint dir
    (run_ppo.py:731-760). A crash between the two writes leaves them a step apart; resuming such a pair
    would train a policy whose rollout history no longer matches its critic's value baseline.
    ``PPOLoop.resume`` (run_ppo.py:762-766) sets ``global_step`` from the policy state and raises unless the
    value state reports the same count. ``resume`` reads/sets ONLY ``global_step``, so a bare ``__new__``
    instance exercises the exact guard with no GPU / Ray build -- the same faithful-unit pattern
    ``test_infeasible_combos`` uses for ``MegatronModel``. The real atomic policy+value save this protects
    is proven e2e in ``test_algorithms_e2e::test_ppo_e2e_policy_and_value_atomic``.

    Reverse-verify: delete the run_ppo.py:765-766 mismatch check and the (5,3) pair resumes silently ->
    ``pytest.raises`` goes red; restore -> refused again.
    """
    from swift.dev.recipe.run_ppo import PPOLoop

    mismatched = PPOLoop.__new__(PPOLoop)  # __init__ not run: resume only reads/sets global_step
    with pytest.raises(ValueError, match='different completed-step counts'):
        mismatched.resume({'consumed_train_samples': 5}, value_state={'consumed_train_samples': 3})

    aligned = PPOLoop.__new__(PPOLoop)
    aligned.resume({'consumed_train_samples': 5}, value_state={'consumed_train_samples': 5})
    assert aligned.global_step == 5, f'aligned policy/value resume set global_step={aligned.global_step}'

    # value_state=None is the single-component resume (no critic to cross-check): accepted, counter set.
    policy_only = PPOLoop.__new__(PPOLoop)
    policy_only.resume({'consumed_train_samples': 7})
    assert policy_only.global_step == 7, f'policy-only resume set global_step={policy_only.global_step}'


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
        rlhf_type='dpo',
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
