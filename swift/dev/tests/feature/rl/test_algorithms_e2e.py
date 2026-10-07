# Copyright (c) ModelScope Contributors. All rights reserved.
"""Dimension ① -- every alignment algorithm trained end to end (RL_PLAN §四 test_algorithms_e2e).

One to two real optimizer steps per algorithm through the production CLI lifecycle (``run_rl``): real Ray
actors, a real vLLM sampler for the online family, a real frozen reference/teacher, LoRA on a tiny (or real
0.5B) checkpoint. No stubs, no degenerate ``group=None`` / single-sample shortcuts -- a green test means the
algorithm's real integration path ran and persisted a trained, self-describing checkpoint.

Post-conditions come from ``assert_rl_trained`` (history non-empty, loss finite + normalized, checkpoint read
back from disk, ``lora_B`` non-zero == the optimizer really moved the adapter). The DPO ``ln2`` start is an
INDEPENDENT oracle, not a self-consistency round-trip: with an identity adapter (``lora_B``=0) the policy
equals its own disabled-adapter reference, so the chosen/rejected log-ratio difference is exactly 0 and
``-log(sigmoid(beta*0)) == ln2`` for any beta -- a wrong reduction or a non-identity init trips it.

Gated ``slow`` (a real Ray session per test); the multi-GPU / vLLM-heavy ones also carry ``accel``. Run on a
free card with ``CUDA_VISIBLE_DEVICES=<cards> pytest swift/dev/tests/feature/rl/test_algorithms_e2e.py -m slow``.
"""
import math
import os

import pytest

from swift.dev.tests.feature.rl.conftest import rl_configs, run_rl

_LN2 = math.log(2.0)


def _varied_content_reward(completions, **kwargs):
    """A deterministic ORM reward that VARIES with completion CONTENT (char-sum mod 11, scaled to [0,1)).

    GRPO's advantage is ``(r - group_mean) / group_std``, so a constant reward across a prompt's
    ``num_generations`` completions gives std=0 -> advantage 0 -> zero gradient -> ``lora_B`` never moves
    (the run trains nothing). A tiny random-weight policy rolled out at temperature>0 emits DIFFERENT
    gibberish per sample, so keying the reward on content gives most groups a non-zero std and the run
    really trains; the mod still lands some groups on std=0, exercising GRPO's divide-by-zero guard. A
    LENGTH-based reward would be constant here -- a tiny model never emits EOS, so every completion runs
    to ``max_completion_length`` -- which is exactly the degenerate case this avoids.
    """
    return [float(sum(ord(ch) for ch in completion) % 11) / 11.0 for completion in completions]


#: Per-algorithm converged-loss ceiling for the normalized-loss guard. The offline family splits by whether
#: the loss carries a full-vocabulary SFT/NLL term (~ln(151k) ~= 11.9 at init): cpo/orpo/simpo do, so they
#: get the SFT-style <20 ceiling; dpo/kto/rm contrast log-ratios or a scalar head and start near ln2/0.5.
_OFFLINE_MAX_LOSS = {
    'dpo': 2.0,
    'kto': 5.0,
    'cpo': 20.0,
    'orpo': 20.0,
    'simpo': 20.0,
    'rm': 10.0,
}


@pytest.mark.slow
@pytest.mark.accel(1)
def test_dpo_loss_starts_at_ln2_identity_adapter(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """ORACLE: DPO's first loss is ln2 because an identity adapter makes policy == reference exactly.

    This is the failure-surface anchor for the whole DPO family: the loss must be the reference-CONTRASTIVE
    ``-log(sigmoid(beta*(logratio_c - logratio_r)))``, and at init both log-ratios vanish (the adapter-disabled
    reference is bitwise the identity-adapter policy, so each ``policy_logp - reference_logp`` is exactly 0),
    giving ``-log(sigmoid(beta*0)) == ln2`` independent of beta, of the chosen/rejected sequences, and of the
    reduction -- a DETERMINISTIC value, not a data-dependent one, so the band is tight (2e-3).

    The tight band is load-bearing, not cosmetic: dropping the reference subtraction (contrasting raw logps,
    cpo-style) leaves only ``policy_chosen_logp - policy_rejected_logp``, which on a tiny model over these
    short rows is small -- it moves step-1 to ~0.705, a mere 1.2e-2 off ln2. A loose 0.05 band would NOT catch
    that (measured: it passed with the reference contrast removed); 2e-3 does. A raw-sum reduction, which
    scales with the token count, moves it far more. Reverse-verify: drop the reference contrast in
    ``DPOLoss.forward`` -> step-1 leaves the 2e-3 band -> red; restore -> ln2 again.
    """
    out_dir = str(tmp_path / 'out_dpo_ln2')
    configs = rl_configs(
        rlhf_type='dpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.preference(n=4),
        out_dir=out_dir,
        max_steps=1,
    )
    history = run_rl(configs)
    losses = assert_rl_trained(history, configs, 'dpo', max_loss=2.0)
    assert abs(losses[0] - _LN2) < 2e-3, \
        f'DPO first loss {losses[0]:.6f} != ln2 {_LN2:.6f} (off by {abs(losses[0] - _LN2):.2e} > 2e-3): the ' \
        'identity-adapter policy/reference contrast is not cancelling to exactly 0 -- a wrong loss form ' \
        '(e.g. raw-logp contrast with the reference dropped), a non-identity init, or a raw-sum reduction'


@pytest.mark.slow
@pytest.mark.accel(1)
@pytest.mark.parametrize('rlhf_type', sorted(_OFFLINE_MAX_LOSS))
def test_offline_preference_family_e2e(rlhf_type, tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """Each offline preference algorithm (dpo/kto/cpo/orpo/simpo/rm) trains + persists end to end.

    All six ride the SAME ``run_dpo``/``PreferenceLoop`` path on the transformers backend under
    ``mode='ray'`` (RL is always Ray), differing only in the loss ``configure_rlhf_loss`` picks and whether
    a reference is consulted. ``rm`` alone is a ``seq_cls`` reward head (no labels, no logps), so it asserts
    the ``task_type='seq_cls'`` force_load key that keeps ``swift infer`` from downgrading it to causal_lm.

    KTO gets ``per_device_train_batch_size=2``: its loss anchors on a mismatched KL batch that
    ``_encode_kto_batch`` builds by ROTATING the completions within a batch, and rotating a one-row batch
    is a no-op (``rejected_response == response``, which the template rejects). Batch >= 2 is intrinsic to
    the unpaired objective, not a test shortcut -- a size-1 KTO batch cannot form a KL point.
    """
    out_dir = str(tmp_path / f'out_{rlhf_type}')
    data = rl_data.kto(n=8) if rlhf_type == 'kto' else rl_data.preference(n=8)
    train_over = {'per_device_train_batch_size': 2} if rlhf_type == 'kto' else None
    configs = rl_configs(
        rlhf_type=rlhf_type,
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=data,
        out_dir=out_dir,
        max_steps=2,
        train_over=train_over,
    )
    history = run_rl(configs)
    assert_rl_trained(
        history,
        configs,
        rlhf_type,
        max_loss=_OFFLINE_MAX_LOSS[rlhf_type],
        expect_task_type='seq_cls' if rlhf_type == 'rm' else None,
    )


@pytest.mark.slow
@pytest.mark.accel(1)
def test_grpo_e2e_colocate_vllm(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """GRPO end to end: a real vLLM rollout (colocate) feeds a real policy-gradient step.

    This is the on-policy anchor -- the whole online family rides this rollout + weight-sync + group-
    normalized-advantage path. It drives ``run_rlhf`` under ``mode='ray'`` with a colocated vLLM sampler
    that GENERATES ``num_generations`` completions per prompt (temperature>0 so they differ), scores them
    with a content-varied ORM, and takes real optimizer steps. Post-conditions beyond the universal ones:
    the history carries the streaming rollout driver's fingerprint (``stream_publishes`` /
    ``version_span_mean`` -- absent if GRPO degraded to a non-rollout loop) and ``lora_B`` moved. The
    ``lora_B`` movement IS the reward/advantage oracle: GRPO's policy-gradient term is ``advantage * ratio``,
    and at init the policy equals its reference (KL ~ 0), so the ONLY thing that can move the adapter is a
    non-zero group-relative advantage back-propagating -- a ``completion_mask`` off-by-one (wrong tokens
    credited) or an advantage ``std=0`` divide-by-zero (NaN, or all-zero advantage) trips the finite-loss /
    lora_B guards.
    """
    out_dir = str(tmp_path / 'out_grpo')
    configs = rl_configs(
        rlhf_type='grpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=4),
        out_dir=out_dir,
        max_steps=2,
        rlhf_over={'num_generations': 4, 'orm': [_varied_content_reward]},
        # temperature>0 so the group's completions differ (greedy would make them identical -> std=0 ->
        # no gradient); a short completion budget keeps the rollout cheap on one card.
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(
        history, configs, 'grpo', max_loss=20.0, require_keys=('stream_publishes', 'version_span_mean'))


@pytest.mark.slow
@pytest.mark.accel(1)
def test_rft_best_of_n_e2e(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """RFT (rejection-sampling FT) end to end: roll out, score with ``orm accuracy``, keep best-of-n, SFT.

    RFT reuses GRPO's rollout + reward machinery but sets the plain SFT ``cross_entropy`` loss on the KEPT
    completions -- no advantage, no reference, no importance ratio. ``best_of_n`` always keeps exactly one
    completion per prompt, so a round never comes up empty and the run always takes real steps (the
    empty-kept-set fail-loudly path is a ``threshold`` selector corner, covered separately). The kept-set
    SFT gradient moves ``lora_B`` regardless of the (all-zero, on a tiny model) accuracy reward, which is
    exactly the point: RFT trains on the selected sequences, not on a reward-weighted objective.
    """
    out_dir = str(tmp_path / 'out_rft')
    configs = rl_configs(
        rlhf_type='rft',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.rft_math(n=4),
        out_dir=out_dir,
        max_steps=2,
        rlhf_over={'orm': ['accuracy'], 'rft_num_samples': 2, 'rft_select': 'best_of_n', 'rft_iterations': 1},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(history, configs, 'rft', max_loss=20.0)


@pytest.mark.slow
# accel(2): a separate frozen teacher is a Ray actor on its OWN disjoint DeviceGroup (auxiliary weights
# never contend with the trainer/sampler), planned AFTER the trainer+sampler ranks -- so GKD needs the
# policy card plus one teacher card. This is the design (plan §四 placement row 4), not a test shortcut.
@pytest.mark.accel(2)
def test_gkd_e2e_separate_teacher(tiny_qwen2_5, tiny_qwen2_5_teacher, rl_data, tmp_path, assert_rl_trained):
    """GKD (on-policy distillation) end to end against a SEPARATE frozen teacher actor.

    The student rolls out on-policy (vLLM), a frozen teacher -- a distinct tiny checkpoint, so its logits
    differ from the identity-adapter student's -- scores those completions with ``forward_only``, and the
    generalized-JSD pulls the student toward it. A self teacher would give zero divergence at init (no
    gradient), so this uses ``tiny_qwen2_5_teacher``: the non-zero student/teacher gap is what makes
    ``lora_B`` move, proving the teacher forward really ran and fed the loss (not a silent no-op).
    """
    out_dir = str(tmp_path / 'out_gkd')
    configs = rl_configs(
        rlhf_type='gkd',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=4),
        out_dir=out_dir,
        max_steps=2,
        rlhf_over={'teacher_model': [tiny_qwen2_5_teacher], 'lmbda': 1.0},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(history, configs, 'gkd', max_loss=20.0)


@pytest.mark.slow
@pytest.mark.accel(1)
def test_opsd_e2e_privileged_self_teacher(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """OPSD (on-policy self-distillation) end to end: the teacher is the student on a PRIVILEGED prompt.

    OPSD needs no separate teacher model -- the same policy scores each rollout twice, once on the plain
    prompt and once on the row's privileged ``teacher_prompt`` (extra information), and the student is
    pulled toward that privileged view. Because the two views differ, the divergence is non-zero even
    though the weights are shared, so ``lora_B`` moves. This is the ``teacher_model=None`` self-distillation
    branch (the 'OPSD teacher None ambiguity' failure the plan targets): None must resolve to the
    privileged-prompt self teacher, not crash or silently skip the teacher forward.
    """
    out_dir = str(tmp_path / 'out_opsd')
    configs = rl_configs(
        rlhf_type='opsd',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.opsd(n=4),
        out_dir=out_dir,
        max_steps=2,
        rlhf_over={'lmbda': 1.0, 'opsd_reverse': True},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(history, configs, 'opsd', max_loss=20.0)


@pytest.mark.slow
# accel(2): MOPD always builds K separate frozen teachers on the shared 'teacher' DeviceGroup (disjoint
# from the trainer/sampler), so it needs the policy card plus one teacher card regardless of K.
@pytest.mark.accel(2)
def test_mopd_e2e_multi_teacher(tiny_qwen2_5, tiny_qwen2_5_teacher, tiny_qwen2_5_teacher2, rl_data, tmp_path,
                                assert_rl_trained):
    """MOPD (multi-teacher on-policy distillation) end to end: a weighted blend of frozen teachers.

    Two teacher actors -- DISTINCT tiny checkpoints, so the blend is a genuine mixture, not one teacher
    duplicated -- score each student rollout and ``MOPDLoss`` fuses their channels by ``teacher_weights``
    (normalized to sum 1). Both differ from the identity-adapter student, so the divergence and its gradient
    are non-zero and ``lora_B`` moves. This exercises the multi-teacher DeviceGroup planning, the K-channel
    weighted fusion, and the per-teacher Ray-actor naming a single-teacher GKD run never touches: all K
    teachers share ONE 'teacher' group built from one call site, so each needs a distinct ``instance_id`` or
    twinkle's ``{group}-{class}-{caller}-{rank}`` actor name collides (``ActorAlreadyExistsError``) on the
    second -- the K>1 placement path this test drives.
    """
    out_dir = str(tmp_path / 'out_mopd')
    configs = rl_configs(
        rlhf_type='mopd',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=4),
        out_dir=out_dir,
        max_steps=2,
        rlhf_over={
            'teacher_model': [tiny_qwen2_5_teacher, tiny_qwen2_5_teacher2],
            'teacher_weights': [0.6, 0.4],
            'lmbda': 1.0,
        },
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(history, configs, 'mopd', max_loss=20.0)


@pytest.mark.slow
# accel(2): the frozen reward model is a seq_cls Ray actor on its OWN disjoint DeviceGroup (the critic
# shares the policy mesh and a LoRA reference is the adapter-disabled policy, so neither plans a group) --
# so PPO needs the policy+critic+sampler card plus one reward-model card.
@pytest.mark.accel(2)
def test_ppo_e2e_policy_and_value_atomic(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """PPO end to end: policy + critic + reference + reward model + per-token GAE, atomically checkpointed.

    PPO is the heaviest online path -- a trainable ``seq_cls`` critic (value head) alongside the policy, a
    frozen reference for the KL penalty, and a frozen reward model scoring each rollout. The reward model
    here is the tiny checkpoint loaded as a ``seq_cls num_labels=1`` scorer: its randomly-initialised head
    gives DIFFERENT scalars for different completions, so the per-token GAE advantage is non-degenerate and
    both the policy and the critic really train. Post-conditions beyond the universal ones: the history
    carries PPO's ``reward`` and ``value_loss`` (the critic's own objective -- absent unless the value head
    ran), and ``checkpoint-final`` holds the policy adapter AND a ``value_model`` subdir written together
    (the atomic policy+value save a step-aligned resume depends on). This is the path the plan flags as
    'still does not run' (GAE called with positional args against a kw-only signature).
    """
    out_dir = str(tmp_path / 'out_ppo')
    configs = rl_configs(
        rlhf_type='ppo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=4),
        out_dir=out_dir,
        max_steps=2,
        rlhf_over={'reward_model': [tiny_qwen2_5], 'num_generations': 2},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(
        history, configs, 'ppo', max_loss=50.0, require_keys=('reward', 'value_loss'))
    # Atomic policy+value: the critic is saved under checkpoint-final/value_model alongside the policy
    # adapter, so a resume can step-align them (run_ppo.resume rejects mismatched step counts).
    value_dir = os.path.join(configs['checkpoint_config'].output_dir, 'checkpoint-final', 'value_model')
    assert os.path.isdir(value_dir), f'ppo: no value_model subdir under checkpoint-final ({value_dir})'
    assert any(f.endswith('.safetensors') for f in os.listdir(value_dir)), \
        f'ppo: value_model checkpoint has no weights: {sorted(os.listdir(value_dir))}'
