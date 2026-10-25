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
from twinkle.data_format import last_assistant_text

_LN2 = math.log(2.0)


def _varied_content_reward(trajectories, **kwargs):
    """A deterministic ORM reward that VARIES with completion CONTENT (char-sum mod 11, scaled to [0,1)).

    GRPO's advantage is ``(r - group_mean) / group_std``, so a constant reward across a prompt's
    ``num_generations`` completions gives std=0 -> advantage 0 -> zero gradient -> ``lora_B`` never moves
    (the run trains nothing). A tiny random-weight policy rolled out at temperature>0 emits DIFFERENT
    gibberish per sample, so keying the reward on content gives most groups a non-zero std and the run
    really trains; the mod still lands some groups on std=0, exercising GRPO's divide-by-zero guard. A
    LENGTH-based reward would be constant here -- a tiny model never emits EOS, so every completion runs
    to ``max_completion_length`` -- which is exactly the degenerate case this avoids.
    """
    return [float(sum(ord(ch) for ch in last_assistant_text(completion)) % 11) / 11.0 for completion in trajectories]


def _constant_reward(trajectories, **kwargs):
    """A CONSTANT ORM reward (same scalar for every completion).

    Used as the rule half of a mixed rule+RM channel so the rule contributes NO within-group variance: any
    non-zero GRPO advantage -- hence any ``lora_B`` movement -- must then come from the OTHER scorer (the
    scalar reward model). That is what gives the multi-reward test teeth: if the model half silently degrades
    (e.g. a plain causal LM misresolved to a generative judge whose random-weight verdicts parse to None, which
    ``weighted_reward``'s nansum drops), the folded reward collapses to this constant, every group's std is 0,
    the advantage is 0, and ``lora_B`` never moves -- tripping ``assert_rl_trained``'s non-zero-adapter check.
    """
    return [0.5 for _ in trajectories]


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
        rl_type='dpo',
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
@pytest.mark.parametrize('rl_type', sorted(_OFFLINE_MAX_LOSS))
def test_offline_preference_family_e2e(rl_type, tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """Each offline preference algorithm (dpo/kto/cpo/orpo/simpo/rm) trains + persists end to end.

    All six ride the SAME ``run_dpo``/``PreferenceBatchEncoder`` path on the transformers backend under
    ``mode='ray'`` (RL is always Ray), differing only in the loss ``configure_rl_loss`` picks and whether
    a reference is consulted. ``rm`` alone is a ``seq_cls`` reward head (no labels, no logps), so it asserts
    the ``task_type='seq_cls'`` force_load key that keeps ``swift infer`` from downgrading it to causal_lm.

    KTO gets ``per_device_train_batch_size=2``: its loss anchors on a mismatched KL batch that
    ``_encode_kto_batch`` builds by ROTATING the completions within a batch, and rotating a one-row batch
    is a no-op (``rejected_response == response``, which the template rejects). Batch >= 2 is intrinsic to
    the unpaired objective, not a test shortcut -- a size-1 KTO batch cannot form a KL point.
    """
    out_dir = str(tmp_path / f'out_{rl_type}')
    data = rl_data.kto(n=8) if rl_type == 'kto' else rl_data.preference(n=8)
    train_over = {'per_device_train_batch_size': 2} if rl_type == 'kto' else None
    configs = rl_configs(
        rl_type=rl_type,
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
        rl_type,
        max_loss=_OFFLINE_MAX_LOSS[rl_type],
        expect_task_type='seq_cls' if rl_type == 'rm' else None,
    )


@pytest.mark.slow
@pytest.mark.accel(1)
def test_grpo_e2e_colocate_vllm(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """GRPO end to end: a real vLLM rollout (colocate) feeds a real policy-gradient step.

    This is the on-policy anchor -- the whole online family rides this rollout + weight-sync + group-
    normalized-advantage path. It drives ``run_rl`` under ``mode='ray'`` with a colocated vLLM sampler
    that GENERATES ``num_generations`` completions per prompt (temperature>0 so they differ), scores them
    with a content-varied ORM, and takes real optimizer steps. Colocate is the PLAIN-loop branch
    (``run_grpo._colocate_loop``; deep_refactor.md 75-86: no StreamingDriver, that is the disaggregated
    branch), so the stream-driver fingerprints (``stream_publishes`` / ``version_span_mean``) do NOT exist
    here -- they are pinned by the disaggregated async suite (``test_async_e2e``). What proves this did not
    degrade to a non-rollout loop is the GRPO surrogate telemetry the model-registered GRPOMetric emits:
    ``train/approx_kl`` and ``train/clip_ratio`` both need the rollout-time behaviour logps (``old_logp``)
    contrasted against the training forward, so they are absent unless a real on-policy rollout fed the
    policy gradient. And ``lora_B`` movement IS the reward/advantage oracle: GRPO's policy-gradient term is
    ``advantage * ratio``, and at init the policy equals its reference (KL ~ 0), so the ONLY thing that can
    move the adapter is a non-zero group-relative advantage back-propagating -- a ``completion_mask``
    off-by-one (wrong tokens credited) or an advantage ``std=0`` divide-by-zero (NaN, or all-zero
    advantage) trips the finite-loss / lora_B guards.
    """
    out_dir = str(tmp_path / 'out_grpo')
    configs = rl_configs(
        rl_type='grpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=4),
        out_dir=out_dir,
        max_steps=2,
        rlhf_over={'num_generations': 4}, reward_over={'orm': [_varied_content_reward]},
        # temperature>0 so the group's completions differ (greedy would make them identical -> std=0 ->
        # no gradient); a short completion budget keeps the rollout cheap on one card.
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(
        history, configs, 'grpo', max_loss=20.0, require_keys=('train/approx_kl', 'train/clip_ratio'))


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
        rl_type='rft',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.rft_math(n=4),
        out_dir=out_dir,
        max_steps=2,
        reward_over={'orm': ['accuracy']}, rlhf_over={'rft_num_samples': 2, 'rft_select': 'best_of_n', 'rft_iterations': 1},
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
        rl_type='gkd',
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
        rl_type='opsd',
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
        rl_type='mopd',
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
    carries PPO's ``train/total_reward`` and ``value_loss`` (the critic's own objective -- absent unless the
    value head ran), and ``checkpoint-final`` holds the policy adapter AND a ``value_model`` subdir written
    together (the atomic policy+value save a step-aligned resume depends on).
    """
    out_dir = str(tmp_path / 'out_ppo')
    configs = rl_configs(
        rl_type='ppo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=4),
        out_dir=out_dir,
        max_steps=2,
        reward_over={'orm': [tiny_qwen2_5], 'orm_model_type': ['qwen2'], 'orm_task_type': ['seq_cls']},
        rlhf_over={'num_generations': 2},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(
        history, configs, 'ppo', max_loss=50.0, require_keys=('train/total_reward', 'value_loss'))
    # Atomic policy+value: the critic is saved under checkpoint-final/value_model alongside the policy
    # adapter, so a resume can step-align them (run_ppo.resume rejects mismatched step counts).
    value_dir = os.path.join(configs['checkpoint_config'].output_dir, 'checkpoint-final', 'value_model')
    assert os.path.isdir(value_dir), f'ppo: no value_model subdir under checkpoint-final ({value_dir})'
    assert any(f.endswith('.safetensors') for f in os.listdir(value_dir)), \
        f'ppo: value_model checkpoint has no weights: {sorted(os.listdir(value_dir))}'


@pytest.mark.slow
@pytest.mark.accel(1)
def test_grpo_reference_kl_in_reward_engages_reference(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """S7-47: GRPO's reference-KL path (``beta>0`` + ``calculate_KL`` + ``kl_in_reward``) really engages.

    The KL penalty is computed in the REMOTE reward worker (``RewardWorker.grpo_advantages`` ->
    ``sequence_kl``), not driver-side, so no in-process spy can observe it -- the oracle is instead the
    worker's own FAIL-LOUDLY guard: with ``kl_in_reward=True`` and ``beta>0`` it raises
    ``'kl_in_reward=True requires an active reference model'`` unless the reference lane produced real
    ``ref_logps`` (``worker.py``: ``ref_logps`` is None iff ``scored['ref']`` is absent, which happens iff
    ``grpo.assembly`` never built the reference). Under LoRA the anchor is the adapter-disabled policy base
    (``validate._check_rlhf_ref_model`` refuses a separate ``ref_model`` here, and dev does not train
    full-parameter). So a GREEN run is POSITIVE PROOF the whole chain is wired: ``beta>0`` +
    ``calculate_KL`` built the frozen reference, the loop forwarded it with ``disable_lora=True``, the worker
    read per-token ``ref_logps``, and ``sequence_kl`` folded ``beta*kl`` into the per-sample reward that drives
    the advantage. A KL path that failed to build the reference or dropped ``ref_logps`` cannot reach green --
    it dies on the guard. Reverse-verify: neuter the reference build in ``grpo.assembly`` (so ``scored['ref']``
    is absent) -> the worker guard raises -> red; restore -> green.
    """
    prompts = rl_data.prompt_only(n=4)
    configs = rl_configs(
        rl_type='grpo', model=tiny_qwen2_5, model_type='qwen2', template='qwen2_5', dataset=prompts,
        out_dir=str(tmp_path / 'out_kl'), max_steps=2, train_over={'learning_rate': 5e-3},
        # beta>0 + kl_in_reward routes the reference KL into the per-sample reward (worker.sequence_kl),
        # which is the path guarded by the fail-loudly 'requires an active reference model' check.
        rlhf_over={'num_generations': 4, 'beta': 5.0, 'calculate_KL': True, 'kl_in_reward': True},
        reward_over={'orm': [_varied_content_reward]},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16})
    history = run_rl(configs)
    # Green here is the oracle: the run cannot complete unless the reference-KL lane fed real ref_logps to
    # the worker (else the guard raises). The GRPO surrogate telemetry additionally proves a real on-policy
    # rollout fed the policy gradient, and lora_B movement proves the KL-shaped advantage back-propagated.
    assert_rl_trained(history, configs, 'grpo', max_loss=50.0,
                      require_keys=('train/approx_kl', 'train/clip_ratio'))


@pytest.mark.slow
# accel(2): the scalar reward model is a frozen seq_cls Ray actor on its OWN disjoint DeviceGroup (reward_0),
# planned after the trainer+sampler ranks -- so a mixed rule+RM channel needs the policy card plus one
# reward-model card. The rule scorer is a plain callable and plans no group.
@pytest.mark.accel(2)
def test_grpo_multi_reward_channel_rule_plus_scalar_rm(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained,
                                                       capfd):
    """S7-44: a MIXED ``--orm`` channel (a rule + a scalar reward model) resolves, places, scores and folds.

    This is the multi-reward-channel path a single-rule GRPO never touches: ``resolve_reward_scorers`` splits
    the channel by ``is_registered_reward`` into rules-first then model ids, aligns the per-slot companions
    (``orm_model_type`` / ``orm_task_type``) to the MODEL-ID subsequence (NOT the rules), gives the scalar RM
    its own ``reward_0`` DeviceGroup, and ``weighted_reward`` folds each sample's per-scorer row by
    ``orm_weights`` (aligned to the MERGED channel order). ORACLE (driver-observable -- the scoring itself
    runs in a remote worker, so no in-process spy can see it): the rule half is a CONSTANT (``_constant_reward``,
    no within-group variance), so the ONLY thing that can make a group's reward vary -- and thus give GRPO a
    non-zero advantage and move ``lora_B`` -- is the scalar RM's varied ``seq_cls`` scores. ``assert_rl_trained``
    reads ``lora_B`` back from the checkpoint and requires it non-zero, so a green run POSITIVELY proves the
    model half of the mixed channel really scored with finite, completion-dependent values. The tiny checkpoint
    is a plain causal LM, so ``orm_task_type=['seq_cls']`` forces it to a ``seq_cls num_labels=1`` head; without
    that override it misresolves to a generative judge whose random-weight verdicts parse to ``None``, which
    ``weighted_reward``'s nansum DROPS -- the folded reward collapses to the constant, every group's std is 0,
    the advantage is 0, and ``lora_B`` never moves. (The ``reward_0`` DeviceGroup is also asserted from the
    driver-side initialize log, but note it is planned for ANY model id in the channel -- scalar or judge -- so
    it proves placement, not the scalar head; the ``lora_B`` check is what proves the seq_cls RM scored.)
    Reverse-verify: drop ``orm_task_type`` -> the RM becomes a None-emitting judge -> the constant rule is the
    only signal -> zero advantage -> ``lora_B`` stays 0 -> ``assert_rl_trained`` goes red.
    """
    prompts = rl_data.prompt_only(n=4)
    configs = rl_configs(
        rl_type='grpo', model=tiny_qwen2_5, model_type='qwen2', template='qwen2_5', dataset=prompts,
        out_dir=str(tmp_path / 'out_multi_reward'), max_steps=2,
        rlhf_over={'num_generations': 4},
        # A CONSTANT rule (callable) + a scalar RM (model id): the rule contributes no within-group variance,
        # so any lora_B movement must come from the RM's varied seq_cls scores (the teeth -- see the docstring).
        # Companions align to the model-id subsequence (one entry); orm_weights aligns to the merged [rule, RM]
        # channel order (two entries).
        reward_over={
            'orm': [_constant_reward, tiny_qwen2_5],
            'orm_weights': [1.0, 1.0],
            'orm_model_type': ['qwen2'],
            'orm_task_type': ['seq_cls'],
        },
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16})
    history = run_rl(configs)
    assert_rl_trained(history, configs, 'grpo', max_loss=50.0,
                      require_keys=('train/approx_kl', 'train/clip_ratio'))
    # The initialize log lists every planned DeviceGroup; fd-level capture is robust to the logger's
    # propagation and to Ray forwarding it from the initialize call.
    captured = capfd.readouterr()
    device_groups = captured.err + captured.out
    assert 'reward_0' in device_groups, (
        'a scalar reward model in the --orm channel must be placed on its OWN reward_0 DeviceGroup, but no '
        'such group was planned -- the RM was dropped, misresolved to a sampler-reusing generative judge '
        f'(which plans no group), or the multi-model channel split failed. captured: {device_groups[-2000:]}')


@pytest.mark.slow
@pytest.mark.accel(1)
def test_grpo_dynamic_sample_regenerates_zero_variance_groups(tiny_qwen2_5, rl_data, tmp_path,
                                                               assert_rl_trained):
    """S7-23: ``dynamic_sample`` regenerates a zero-variance group on the same prompt, bounded by the cap.

    ``run_grpo``'s fixed-batch branch regenerates each group whose weighted-reward variance is <= 0 (its GRPO
    advantage is degenerate) up to ``max_resample_times`` times, then KEEPS the last try -- the cap-accept is
    intentional behaviour, not a failure. ORACLE (driver-observable via a tally file): the ORM rule is a
    CONSTANT, so EVERY group stays at variance 0 no matter what the sampler redraws -- each group therefore
    regenerates exactly ``max_resample_times`` times before the cap accepts. The rule closure appends one
    line to a file per scoring call (it runs in the remote score worker, but the closure is cloudpickled
    there once and the instance persists, and worker and driver share the filesystem): with 4 prompts and
    ``max_resample_times=2`` the tally is 1 initial whole-batch call + 4 groups x 2 regen calls = 9 calls,
    asserted from below (>= 5) so multi-worker dispatch fan-out cannot make it vacuously pass -- while a
    silently-dead regen loop leaves the tally at ~1 and trips it. ``lora=False`` on the post-condition:
    the cap-accepted constant reward makes every advantage 0, so ``lora_B`` legitimately stays zero here.
    Reverse-verify: disable the regen branch in ``run_grpo`` -> the tally collapses to the initial call ->
    red; restore -> green.
    """
    tally = tmp_path / 'score_calls.txt'

    def _tallying_constant_reward(trajectories, **kwargs):
        """A constant 0.5 ORM that records each scoring call (one line per batch) into ``tally``."""
        with open(tally, 'a', encoding='utf-8') as f:
            f.write(f'{len(trajectories)}\n')
        return [0.5 for _ in trajectories]

    prompts = rl_data.prompt_only(n=4)
    configs = rl_configs(
        rl_type='grpo', model=tiny_qwen2_5, model_type='qwen2', template='qwen2_5', dataset=prompts,
        out_dir=str(tmp_path / 'out_dyn'), max_steps=2,
        rlhf_over={'num_generations': 4, 'dynamic_sample': True, 'max_resample_times': 2},
        reward_over={'orm': [_tallying_constant_reward]},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16})
    history = run_rl(configs)
    # lora=False is the CONTRACT here: the cap-accepted constant reward zeroes every advantage, so lora_B
    # staying zero is the documented cap-accept behaviour, not a broken run.
    assert_rl_trained(history, configs, 'grpo', max_loss=50.0, lora=False,
                      require_keys=('train/approx_kl', 'train/clip_ratio'))
    calls = tally.read_text(encoding='utf-8').splitlines()
    assert len(calls) >= 5, (
        f'dynamic_sample must REGENERATE each zero-variance group up to max_resample_times=2 (expected ~9 '
        f'scoring calls: 1 initial batch + 4 groups x 2 regens), but the rule was only called {len(calls)} '
        f'times (batch sizes {calls}) -- the regeneration loop never ran, so degenerate groups were trained '
        'on instead of being resampled')


@pytest.mark.slow
@pytest.mark.accel(1)
def test_grpo_prm_channel_fuses_process_rewards(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """S7-45: the PRM channel (``--prm`` + ``prm_scorer`` 'delimiter') segments, scores and FUSES process rewards.

    The full chain this drives: ``resolve_reward_scorers('prm')`` accepts a RULE callable; the colocate loop
    routes scoring through ``worker.score_group_prm`` (one pass, both channels); the ``DelimiterPRMScorer``
    segments each response into steps, the rule scores each step prefix, and the per-step scores broadcast to
    tokens land on ``sample.prm_process``; ``grpo_advantages`` then FUSES ``advantage = prm_orm_mix * process
    + (1 - prm_orm_mix) * orm`` per token (``worker.fuse_prm``). ORACLE (driver-observable): the ORM half is
    a CONSTANT (zero within-group variance), so the ONLY source of a non-zero GRPO advantage -- hence any
    ``lora_B`` movement -- is the PRM half: a content-keyed step rule whose score varies per completion.
    ``assert_rl_trained`` reads ``lora_B`` back and requires it non-zero, so green POSITIVELY proves the
    process rewards were segmented, scored, delivered to ``prm_process`` and fused into the advantage. If the
    PRM lane died anywhere -- no ``prm_scorer`` resolved, ``score_group_prm`` never called, ``prm_process``
    never landed, or the fuse skipped -- the folded reward collapses to the constant, every group's std is 0,
    and ``lora_B`` stays 0 (red). Reverse-verify: neuter the ``sample.prm_process = ref`` landing in
    ``run_grpo._score_group`` -> the fuse is skipped (``prm_process is None``) -> constant-only advantage ->
    ``lora_B`` zero -> red; restore -> green. (The 'turn' scorer form needs a multi-turn rollout and is
    carried by the multi-turn tools e2e S7-46, not here.)
    """
    prompts = rl_data.prompt_only(n=4)
    configs = rl_configs(
        rl_type='grpo', model=tiny_qwen2_5, model_type='qwen2', template='qwen2_5', dataset=prompts,
        out_dir=str(tmp_path / 'out_prm'), max_steps=2,
        rlhf_over={'num_generations': 4},
        reward_over={
            # Constant ORM (no within-group variance) + a content-keyed PRM rule: any lora_B movement must
            # come from the fused PROCESS rewards, which is exactly what this test wants to prove.
            'orm': [_constant_reward],
            'prm': [_varied_content_reward],
        },
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16})
    history = run_rl(configs)
    assert_rl_trained(history, configs, 'grpo', max_loss=50.0,
                      require_keys=('train/approx_kl', 'train/clip_ratio'))


@pytest.mark.slow
@pytest.mark.accel(1)
def test_grpo_teacher_disable_adapter_base_resolves_shared(tiny_qwen2_5, rl_data, tmp_path,
                                                           assert_rl_trained, capfd):
    """S7-12 form A: teacher_model == the policy checkpoint resolves to the adapter-disabled BASE (shared).

    ``process._derive_rlhf_teacher`` folds ``teacher_model == [policy]`` + LoRA + no teacher_adapters into
    ``_teacher_use_disable_adapter=True; teacher_model=None``: the teacher IS the policy with its adapter
    off, sharing the trainer actor, so NO teacher DeviceGroup is planned and the teacher lane forwards with
    ``disable_lora=True``. ORACLE (driver-observable, on the RESOLUTION the plan targets -- 'teacher 解析错'):
    (1) the DERIVED config read back after ``run_rl`` (process mutates configs in place): teacher_model
    folded to None and the disable_adapter flag set; (2) no ``('teacher', ...`` DeviceGroup in the initialize
    log -- a resolution that wrongly built a SEPARATE teacher actor would plan one and trip this. Green
    training (lora_B moves off the varied rule) completes the run-level proof. Reverse-verify: neuter the
    fold in ``process._derive_rlhf_teacher`` -> teacher_model survives as a list and (on one card) the
    separate-teacher placement blows up or the derived-config assert fires -> red; restore -> green.
    """
    configs = rl_configs(
        rl_type='grpo', model=tiny_qwen2_5, model_type='qwen2', template='qwen2_5',
        dataset=rl_data.prompt_only(n=4),
        out_dir=str(tmp_path / 'out_teacher_a'), max_steps=2,
        rlhf_over={'num_generations': 4, 'teacher_model': [tiny_qwen2_5]},
        reward_over={'orm': [_varied_content_reward]},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16})
    history = run_rl(configs)
    rl_cfg = configs['rl_config']
    assert rl_cfg.teacher_model is None and rl_cfg._teacher_use_disable_adapter, (
        f'teacher_model == [policy] + LoRA must fold to the adapter-disabled base (teacher_model=None, '
        f'_teacher_use_disable_adapter=True), but resolved teacher_model={rl_cfg.teacher_model!r}, '
        f'flag={rl_cfg._teacher_use_disable_adapter} -- the self-teacher was NOT resolved to the shared '
        'adapter-disabled base')
    assert_rl_trained(history, configs, 'grpo', max_loss=50.0,
                      require_keys=('train/approx_kl', 'train/clip_ratio'))
    captured = capfd.readouterr()
    device_groups = captured.err + captured.out
    assert "('teacher'," not in device_groups, (
        'the adapter-disabled self-teacher shares the policy actor, so no teacher DeviceGroup may be '
        f'planned -- a separate teacher actor was built instead. captured: {device_groups[-2000:]}')


@pytest.mark.slow
# accel(2): a DISTINCT frozen teacher is a Ray actor on its OWN disjoint DeviceGroup (TEACHER_GROUP),
# planned after the trainer+sampler ranks -- the policy card plus one teacher card.
@pytest.mark.accel(2)
def test_grpo_teacher_independent_frozen_places_own_group(tiny_qwen2_5, tiny_qwen2_5_teacher, rl_data,
                                                          tmp_path, assert_rl_trained, capfd):
    """S7-12 form B: a DISTINCT teacher checkpoint stays a separate frozen actor on its own group.

    ``teacher_model=[<a different checkpoint>]`` must NOT fold (the derive only folds teacher == policy):
    the teacher is built by ``build_ref_model`` on the ``teacher`` DeviceGroup and the teacher lane forwards
    it WITHOUT ``disable_lora`` (it is a genuinely different model, inheriting the policy's model_type so the
    ambiguous tiny checkpoint resolves). ORACLE: (1) the derived config preserves ``teacher_model`` with the
    disable_adapter flag UNSET; (2) the ``('teacher', ...`` DeviceGroup appears in the initialize log --
    positive proof the separate frozen teacher was PLACED, not dropped; (3) green training. Reverse-verify:
    neuter the TEACHER_GROUP request in ``grpo.assembly`` -> no teacher group is planned -> the placement
    assert goes red; restore -> green.
    """
    configs = rl_configs(
        rl_type='grpo', model=tiny_qwen2_5, model_type='qwen2', template='qwen2_5',
        dataset=rl_data.prompt_only(n=4),
        out_dir=str(tmp_path / 'out_teacher_b'), max_steps=2,
        rlhf_over={'num_generations': 4, 'teacher_model': [tiny_qwen2_5_teacher]},
        reward_over={'orm': [_varied_content_reward]},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16})
    history = run_rl(configs)
    rl_cfg = configs['rl_config']
    assert rl_cfg.teacher_model == [tiny_qwen2_5_teacher] and not rl_cfg._teacher_use_disable_adapter, (
        'a DISTINCT teacher checkpoint must stay a separate teacher (not folded to the adapter-disabled '
        f'base): teacher_model={rl_cfg.teacher_model!r}, flag={rl_cfg._teacher_use_disable_adapter}')
    assert_rl_trained(history, configs, 'grpo', max_loss=50.0,
                      require_keys=('train/approx_kl', 'train/clip_ratio'))
    captured = capfd.readouterr()
    device_groups = captured.err + captured.out
    assert "('teacher'," in device_groups, (
        'a distinct frozen teacher must be placed on its OWN teacher DeviceGroup, but no such group was '
        f'planned -- the teacher was dropped or misresolved. captured: {device_groups[-2000:]}')


@pytest.mark.slow
@pytest.mark.accel(1)
def test_grpo_teacher_privileged_prompt_policy_self_grades(tiny_qwen2_5, rl_data, tmp_path,
                                                           assert_rl_trained, capfd):
    """S7-12 form C: a ``teacher_prompt`` column with NO dedicated teacher -> the policy self-grades.

    The third resolution: with no ``teacher_model`` but a per-row ``teacher_prompt`` extra (the OPSD shape),
    ``run_grpo``'s teacher lane uses the POLICY itself (``t_model, t_disable = cluster.model, False``) over
    ``privileged_features`` -- the SAME response re-encoded from the privileged view (the first user turn
    carrying the hint), so the teacher logps grade the student's answer hint-augmented. ORACLE: the ORM rule
    runs inside the remote score worker on ROW dicts that are ``messages`` PLUS the sample's extras, so it
    records whether every row in the batch carries the ``teacher_prompt`` column -- proving the
    prompt_extras -> rollout -> sample.extra -> scoring-rows plumbing end to end (a dropped extra anywhere
    on that chain leaves the tally without it and trips the assert). Plus: no teacher DeviceGroup (the
    policy grades itself, nothing is placed) and green training. Reverse-verify: force ``extras = None`` at
    the rollout call in ``run_grpo`` -> the rows lose the column -> the tally goes red; restore -> green.
    """
    tally = tmp_path / 'teacher_prompt_rows.txt'

    def _tallying_varied_reward(trajectories, **kwargs):
        """A content-varied ORM that records whether every row in the batch carries teacher_prompt."""
        with open(tally, 'a', encoding='utf-8') as f:
            f.write(f"{int(all(isinstance(t, dict) and t.get('teacher_prompt') for t in trajectories))}\n")
        return [float(sum(ord(ch) for ch in last_assistant_text(completion)) % 11) / 11.0
                for completion in trajectories]

    configs = rl_configs(
        rl_type='grpo', model=tiny_qwen2_5, model_type='qwen2', template='qwen2_5',
        dataset=rl_data.opsd(n=4),  # rows carry the teacher_prompt column
        out_dir=str(tmp_path / 'out_teacher_c'), max_steps=2,
        rlhf_over={'num_generations': 4},
        reward_over={'orm': [_tallying_varied_reward]},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16})
    history = run_rl(configs)
    assert_rl_trained(history, configs, 'grpo', max_loss=50.0,
                      require_keys=('train/approx_kl', 'train/clip_ratio'))
    rows = tally.read_text(encoding='utf-8').splitlines()
    assert rows and all(flag == '1' for flag in rows), (
        f'every scoring row must carry the teacher_prompt extra (the privileged-view plumbing '
        f'prompt_extras -> rollout -> sample.extra -> scoring rows), but the rule saw: {rows}')
    captured = capfd.readouterr()
    device_groups = captured.err + captured.out
    assert "('teacher'," not in device_groups, (
        'no dedicated teacher was configured, so the policy self-grades and NO teacher DeviceGroup may be '
        f'planned. captured: {device_groups[-2000:]}')


@pytest.mark.slow
@pytest.mark.parametrize('mode_over', [
    {'advantage_reweight': 'rlsd', 'rlsd_lambda': 0.5},
    {'sdar_loss_coef': 0.1},
])
@pytest.mark.accel(1)
def test_grpo_rlsd_sdar_teacher_signal_reweighting(mode_over, tiny_qwen2_5, rl_data, tmp_path,
                                                   assert_rl_trained):
    """S7-13: the RLSD / SDAR teacher-signal reweighting path runs end to end over ``teacher_prompt`` rows.

    Both advanced estimators consume the PRIVILEGED teacher lane: RLSD (``advantage_reweight='rlsd'``)
    reweights the scalar GRPO advantage per token by ``apply_rlsd_reweight(old, teacher, lam)`` inside the
    remote worker; SDAR (``sdar_loss_coef>0``) keeps the scalar advantage and lets ``GRPOLoss._add_sdar``
    consume ``teacher_logps`` directly (the gated distill term). Both ride the SAME teacher lane this drives:
    the opsd dataset's ``teacher_prompt`` column with no dedicated teacher -> the policy self-grades its own
    completions hint-augmented (``privileged_features`` re-encodes the first user turn with the hint).

    ORACLE (fail-loudly guard, driver-observable as a run-level failure): ``RewardWorker.grpo_advantages``
    REFUSES an RLSD/SDAR run whose teacher lane produced no ``teacher_logps`` ('requires a teacher signal') --
    without the guard a missing teacher would silently degrade the run to plain GRPO (a dead-parameter trap).
    So a GREEN run proves the teacher lane ran and fed per-token teacher logps into the reweighting; the
    REVERSE leg feeds the same flags a prompt-only dataset (no teacher_prompt, no --teacher_model) and asserts
    the guard raises, pinning both the positive path and the refusal. ``lora=True``: the varied ORM keeps the
    base advantage non-zero, so the reweighted policy gradient still moves ``lora_B``.
    """
    configs = rl_configs(
        rl_type='grpo', model=tiny_qwen2_5, model_type='qwen2', template='qwen2_5',
        dataset=rl_data.opsd(n=4),  # rows carry the teacher_prompt column -> the privileged teacher lane
        out_dir=str(tmp_path / 'out_rlsd'), max_steps=2,
        rlhf_over={'num_generations': 4, **mode_over},
        reward_over={'orm': [_varied_content_reward]},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16})
    history = run_rl(configs)
    assert_rl_trained(history, configs, 'grpo', max_loss=50.0,
                      require_keys=('train/approx_kl', 'train/clip_ratio'))


@pytest.mark.slow
@pytest.mark.accel(1)
def test_grpo_rlsd_without_teacher_signal_refused(tiny_qwen2_5, rl_data, tmp_path):
    """S7-13 refusal leg: RLSD/SDAR with NO teacher lane fails loudly instead of degrading silently.

    A prompt-only dataset plus no ``--teacher_model`` leaves the teacher lane off entirely, so
    ``teacher_logps`` is None -- the worker guard must raise ('requires a teacher signal') rather than run
    plain GRPO with the flag silently ignored (the dead-parameter trap the guard exists for). This is the
    reverse-verification of the positive test above, pinned as its own assertion.
    """
    import pytest as _pytest

    configs = rl_configs(
        rl_type='grpo', model=tiny_qwen2_5, model_type='qwen2', template='qwen2_5',
        dataset=rl_data.prompt_only(n=4),  # no teacher_prompt column, no teacher_model -> no teacher lane
        out_dir=str(tmp_path / 'out_rlsd_refused'), max_steps=2,
        rlhf_over={'num_generations': 4, 'advantage_reweight': 'rlsd', 'rlsd_lambda': 0.5},
        reward_over={'orm': [_varied_content_reward]},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16})
    with _pytest.raises(ValueError, match='requires a teacher signal'):
        run_rl(configs)


@pytest.mark.slow
@pytest.mark.accel(1)
def test_grpo_chord_sft_mixed_training(tiny_qwen2_5, rl_data, write_jsonl, tmp_path, assert_rl_trained):
    """S7-43: CHORD mixes a supervised SFT RPC into the GRPO step, weighted by the mu schedule.

    ``run_grpo`` builds ``chord_fn`` (cycling defensive copies of the encoded ``chord_sft_dataset`` rows) and
    each optimizer step becomes a DUAL RPC on one gradient: ``forward_backward(loss_key='grpo',
    loss_weight=1-mu, finalize_gradients=False)`` then ``forward_backward(inputs=chord_fn(), kind='sft',
    loss_key='sft', loss_weight=mu, finalize_gradients=True)`` -- weights sum to 1 so only the second RPC
    finalizes. ORACLE (an EXTREME-VALUE hand computation on the mu schedule, driver-observable in the history
    losses): with ``chord_mu_warmup_steps=1, chord_mu_peak=1.0`` the schedule gives ``mu = peak*step/warmup``
    -- step 0: ``mu=0`` (the recorded loss is the PURE GRPO surrogate, which on this varied-rule run is
    near-zero); step 1: ``mu=1.0`` (the recorded loss is the PURE SFT cross-entropy on the chord rows, which
    for a random tiny policy starts at ~ln(vocab) ~= 11.9). A chord path that silently dropped the SFT RPC,
    never built ``chord_fn``, or mis-scheduled mu leaves BOTH steps at the small GRPO scale and trips the
    ~10-magnitude jump assertion. Reverse-verify: neuter the dual-RPC branch (fall through to the single
    plain-GRPO RPC) -> both losses stay small -> red; restore -> green.
    """
    import math as _math

    # The chord SFT side: expert rows (a messages dataset, same shape the prompts dataset uses).
    sft_rows = [{
        'messages': [
            {'role': 'user', 'content': 'Name a colour.'},
            {'role': 'assistant', 'content': 'Blue is a colour.'},
        ]
    } for _ in range(4)]
    chord_dataset = write_jsonl(tmp_path / 'chord_sft.jsonl', sft_rows)
    ln_vocab = _math.log(151936)  # tiny Qwen2.5 tokenizer size -- the random-init SFT CE anchor

    configs = rl_configs(
        rl_type='grpo', model=tiny_qwen2_5, model_type='qwen2', template='qwen2_5',
        dataset=rl_data.prompt_only(n=4),
        out_dir=str(tmp_path / 'out_chord'), max_steps=2,
        rlhf_over={
            'num_generations': 4,
            'chord_sft_dataset': [chord_dataset],
            'chord_sft_per_device_train_batch_size': 1,
            'chord_mu_warmup_steps': 1, 'chord_mu_peak': 1.0,
            'chord_mu_decay_steps': 0, 'chord_mu_valley': 1.0,  # decay=0 -> mu=valley=peak=1.0 past warmup
        },
        reward_over={'orm': [_varied_content_reward]},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16})
    history = run_rl(configs)
    assert_rl_trained(history, configs, 'grpo', max_loss=50.0,
                      require_keys=('train/approx_kl', 'train/clip_ratio'))
    losses = [r['loss'] for r in history]
    # Step 0: mu=0 -> pure GRPO surrogate on a varied-rule tiny run (order << 1).
    assert losses[0] < 1.0, (
        f'mu=0 (warmup step 0) must record the PURE GRPO loss (small surrogate), got {losses[0]:.4f} -- the '
        'mu schedule may be off or the SFT RPC is contaminating step 0')
    # Step 1: mu=1.0 -> pure SFT cross-entropy ~ ln(vocab); allow a generous band (2.0, 3*ln_vocab).
    assert abs(losses[1] - ln_vocab) < 0.5 * ln_vocab, (
        f'mu=1.0 (step 1) must record the PURE SFT cross-entropy ~ln(vocab)={ln_vocab:.2f} on the chord '
        f'rows, got {losses[1]:.4f} (losses={losses}) -- the SFT half of the dual RPC never ran, chord_fn '
        'was not built, or the mu schedule never reached peak')


@pytest.mark.slow
@pytest.mark.accel(1)
def test_grpo_tools_multiturn_rl_trains_agent_trajectory(tiny_qwen2_5, rl_data, tmp_path,
                                                         assert_rl_trained):
    """S7-46: a tools-enabled multi-turn GRPO run -- rollout, tool dispatch, trajectory, reward, training.

    The full agent-RL chain in one run: ``max_turns=2`` + ``tools=['sandbox']`` makes the rollout build the
    real ``MultiTurnRollout`` (``tools_factory`` -> a ``LocalEnv`` pool + the sandbox ``run_command`` plugin
    via the per-episode ``ToolManager``); the tiny policy's completions are real multi-turn trajectories; the
    ORACLE (driver-observable via a tally file): the rule closure runs in the remote score worker on the
    trajectory ROWS, whose ``messages`` carry the whole conversation -- it records the ROLE SET the rows
    carry. The expected set here is exactly ``{user, assistant}``: a random tiny policy under the plain
    qwen2_5 template never emits a valid hermes tool-call markup, so the multi-turn engine's no-more-calls
    branch ends each episode after the assistant turn (the GENUINE tool dispatch -- env executes, a tool
    observation role appears -- needs a tool-capable policy and is proven with a LIVE calculator episode in
    infer's test_tools_multiturn_e2e; this test pins the RL half of the chain). Green proves the
    tools-configured MultiTurnRollout really drove generation (env pool built via ``tools_factory``), the
    trajectory conversations reached the scoring worker intact (an exact role set -- a dropped/garbled row
    trips it), and GRPO took real optimizer steps on those trajectory tokens (lora_B moves). Reverse-verify:
    sabotage the row plumbing (drop ``messages`` from the scoring rows) -> the role-set assert goes red.
    """
    tally = tmp_path / 'tool_turns.txt'

    def _tallying_varied_reward(trajectories, **kwargs):
        """A content-varied ORM that records whether any row's conversation contains a tool observation."""
        def _has_tool_turn(traj):
            messages = traj.get('messages') if isinstance(traj, dict) else None
            return bool(messages) and any(m.get('role') in ('tool', 'function') for m in messages
                                          if isinstance(m, dict))
        with open(tally, 'a', encoding='utf-8') as f:
            import json as _json
            f.write(_json.dumps(sorted({m.get('role') for t in trajectories if isinstance(t, dict)
                                         for m in (t.get('messages') or []) if isinstance(m, dict)})) + '\n')
        return [float(sum(ord(ch) for ch in last_assistant_text(completion)) % 11) / 11.0
                for completion in trajectories]

    configs = rl_configs(
        rl_type='grpo', model=tiny_qwen2_5, model_type='qwen2', template='qwen2_5',
        dataset=rl_data.prompt_only(n=4),
        out_dir=str(tmp_path / 'out_tools'), max_steps=2,
        rlhf_over={'num_generations': 2},  # 2 candidates per prompt keeps the multi-turn rollout cheap
        rollout_over={
            'max_turns': 2, 'tools': ['sandbox'],
            'sandbox_num_envs': 1, 'sandbox_workspace_root': str(tmp_path / 'sandbox'),
        },
        reward_over={'orm': [_tallying_varied_reward]},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16})
    history = run_rl(configs)
    assert_rl_trained(history, configs, 'grpo', max_loss=50.0,
                      require_keys=('train/approx_kl', 'train/clip_ratio'))
    # The trajectory-structure oracle: every scoring row's conversation is a well-formed dialogue whose
    # roles are EXACTLY user + assistant. On a random tiny policy under the plain qwen2_5 template the
    # model never emits a valid hermes tool_call markup, so the multi-turn engine's no-more-calls branch
    # ends each episode after the assistant turn -- which still proves the rollout, the trajectory ROW
    # plumbing (messages reach the scoring worker) and the training on those trajectory tokens. The
    # REAL tool dispatch itself (env executes, observation appended) is proven with a live calculator
    # episode in infer's test_tools_multiturn_e2e; here the RL leg pins the training half of the chain.
    import json as _json

    flags = tally.read_text(encoding='utf-8').splitlines()
    roles = set()
    for flag in flags:
        roles |= set(_json.loads(flag))
    assert roles == {'user', 'assistant'}, (
        f'the scoring rows must carry the rollout trajectories\' conversations with exactly the user and '
        f'assistant roles, but saw {sorted(roles)} -- the trajectory-to-scoring-row plumbing is broken '
        '(or a foreign role leaked in)')
    assert_rl_trained(history, configs, 'grpo', max_loss=50.0, require_keys=('train/approx_kl',))



@pytest.mark.slow
# accel(2): a full-parameter policy trains on its own group (sampler colocated), and the reference is a
# DISTINCT frozen full-weight copy that process.py derives automatically for tuner='full' -- it lands on
# its own REF_GROUP card, so the paired-EMA run needs the policy card plus one reference card.
@pytest.mark.accel(2)
def test_grpo_sync_ref_model_ema_full_param(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained, monkeypatch):
    """S7-48: ``sync_ref_model`` drives a paired EMA sync of the frozen full-param reference.

    The chain: validate requires beta>0 with sync_ref_model; process folds ``ref_model`` to the policy
    checkpoint for a full-param run; assembly builds the reference as a distinct frozen model on its own
    REF_GROUP; ``run_grpo`` then constructs ``CheckpointEngineManager(mode='standalone')`` and calls
    ``sync_ema(ref_model_mixup_alpha)`` every ``ref_model_sync_steps`` from the DRIVER loop. ORACLE: the
    manager is a driver-side local object, so a class-level monkeypatch wrapping (not replacing) sync_ema
    both COUNTS the calls (proving the manager was built and the schedule fired) and delegates to the
    real method -- so the genuine paired 2-rank NCCL EMA machinery (prepare/init/send/receive+blend/
    finalize) is exercised end to end, not stubbed. The recorded alpha must equal ref_model_mixup_alpha,
    and the run must train green (full-param: ``lora=False``, GRPO surrogate telemetry present).
    Reverse-verify: neuter the ema_manager construction branch in run_grpo -> spy count 0 -> red.
    """
    from twinkle.checkpoint_engine import CheckpointEngineManager

    calls = []
    real_sync_ema = CheckpointEngineManager.sync_ema

    def _spy_sync_ema(self, alpha):
        calls.append(alpha)
        return real_sync_ema(self, alpha)

    monkeypatch.setattr(CheckpointEngineManager, 'sync_ema', _spy_sync_ema)

    prompts = rl_data.prompt_only(n=4)
    configs = rl_configs(
        rl_type='grpo', model=tiny_qwen2_5, model_type='qwen2', template='qwen2_5', dataset=prompts,
        out_dir=str(tmp_path / 'out_ema'), max_steps=2,
        tuner=None,  # full-param: the reference is a distinct frozen full-weight copy, EMA across all weights
        train_over={'learning_rate': 5e-3},
        rlhf_over={'num_generations': 4, 'beta': 5.0, 'calculate_KL': True,
                   'sync_ref_model': True, 'ref_model_sync_steps': 1, 'ref_model_mixup_alpha': 0.9},
        reward_over={'orm': [_varied_content_reward]},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16})
    history = run_rl(configs)
    assert calls, (
        'sync_ref_model=True with beta>0 must construct the EMA manager (reference built as a distinct '
        'frozen model) and call sync_ema every ref_model_sync_steps from the driver loop, but sync_ema '
        'was never invoked -- the manager was not built (reference lane missing) or the schedule never fired')
    assert all(alpha == 0.9 for alpha in calls), (
        f'sync_ema must receive ref_model_mixup_alpha=0.9 on every call, got {calls}')
    assert_rl_trained(history, configs, 'grpo', max_loss=50.0, lora=False,
                      expect_tuner_type=None,
                      require_keys=('train/approx_kl', 'train/clip_ratio'))
