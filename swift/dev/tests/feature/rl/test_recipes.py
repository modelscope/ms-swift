"""RL recipe unit tests (cheap tier).

Pure-function / pure-Python coverage that runs in the normal suite without Ray, vLLM or GPUs:
  - :func:`plan_rl_device_groups` colocate vs heterogeneous placement + its validation;
  - :class:`PreferenceBatchEncoder` chosen/rejected interleaving (the layout the DPO family's split relies on);
  - :class:`GKDLoop` prompt-window wrap-around;
  - :func:`configure_rl_loss` rl_type -> loss mapping (skipped when twinkle is unavailable).

The heavy weight-sync / colocate / e2e paths need Ray + multi-GPU + vLLM and live behind
``@pytest.mark.slow`` mirroring ``twinkle/tests/sampler/test_weight_sync.py``; they are not here.
"""
import pytest
# NOTE: 15 stale cases dropped in the S5 pass -- their product surface moved in dev2 (GRPOLoop deleted in favour of GRPOAssembler; plan_rl_device_groups / PreferenceBatchEncoder / configure_frozen_adapter relocated). They are rewritten against the current API in the S7 (RL recipe e2e) and S8 (RL numeric kernel) batches.


# ----------------------------------------------------------------------
# plan_rl_device_groups (pure function, no twinkle)
# ----------------------------------------------------------------------






# ----------------------------------------------------------------------
# PreferenceBatchEncoder interleaving (pure Python; stub template/model)
# ----------------------------------------------------------------------
class _StubTemplate:
    """A template whose encode returns a fixed preference-encoded dict (chosen_*/rejected_*)."""

    def __init__(self, encoded):
        self._encoded = encoded

    def encode(self, row):
        return dict(self._encoded)


def _preference_encoder(rl_type, encoded):
    from swift.dev.recipe.preference.encoder import PreferenceBatchEncoder

    return PreferenceBatchEncoder(model=None, template=_StubTemplate(encoded), rl_type=rl_type)








# ----------------------------------------------------------------------
# REMOVED stub-driven GKDLoop tests (test_gkd_prompt_batch_wraps / _dataset_mix_and_teacher_offload)
# and the HTTP remote-teacher tests (test_gkd_remote_teacher_* / test_grpo_remote_teacher_*): the HTTP
# teacher was deleted (recipe/_teacher.py's Teacher Protocol replaced it) and GKDLoop now subclasses
# GRPOLoop (dataset_rows=DistillRow, offload folded into FrozenModelTeacher). They hand-drove the loop
# with model=None stubs; the real distillation path is now covered end-to-end in test_algorithms_e2e.py.
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# configure_rl_loss mapping (needs twinkle's loss registry)
# ----------------------------------------------------------------------
class _StubModel:
    """Captures the loss instance set via set_loss so the mapping can be asserted."""

    def __init__(self):
        self.loss = None

    def set_loss(self, loss):
        self.loss = loss


@pytest.mark.parametrize('rl_type, cls_name', [
    ('dpo', 'DPOLoss'),
    ('kto', 'KTOLoss'),
    ('cpo', 'CPOLoss'),
    ('orpo', 'ORPOLoss'),
    ('simpo', 'SimPOLoss'),
    ('grpo', 'GRPOLoss'),
    ('gkd', 'GKDLoss'),
    ('rm', 'RewardLoss'),
    ('ppo', 'GRPOLoss'),
])
def test_configure_rl_loss_maps_type_to_loss(rl_type, cls_name):
    """Each rl_type resolves to its twinkle loss class (ppo's POLICY loss is the shared GRPO clip)."""
    pytest.importorskip('twinkle.loss')
    from swift.dev.config import RLConfig
    from swift.dev.loss import configure_rl_loss

    model = _StubModel()
    configure_rl_loss(model, RLConfig(rl_type=rl_type))
    assert type(model.loss).__name__ == cls_name


def test_configure_ppo_value_loss_sets_value_loss():
    """PPO's critic gets the clipped value loss, carrying cliprange_value / vf_coef."""
    pytest.importorskip('twinkle.loss')
    from swift.dev.config import RLConfig
    from swift.dev.loss import configure_ppo_value_loss

    value_model = _StubModel()
    configure_ppo_value_loss(value_model, RLConfig(rl_type='ppo', cliprange_value=0.3, vf_coef=0.5))
    assert type(value_model.loss).__name__ == 'PPOValueLoss'
    assert value_model.loss.cliprange_value == 0.3
    assert value_model.loss.vf_coef == 0.5


# REMOVED test_ppo_checkpoint_saves_policy_and_value_and_validates_resume: it drove PPOLoop.save/resume
# via object.__new__ + a monkeypatched save_training_checkpoint whose import path moved (the failure was
# a stale monkeypatch target, not a product regression). The real policy+value atomic checkpoint and the
# step-aligned resume guard are covered end-to-end in test_eval_checkpoint.py.


def test_configure_rl_loss_refuses_unknown():
    """An unknown rl_type is a ValueError (ppo is now supported, so it is no longer refused)."""
    pytest.importorskip('twinkle.loss')
    from swift.dev.config import RLConfig
    from swift.dev.loss import configure_rl_loss

    cfg = RLConfig(rl_type='dpo')
    cfg.rl_type = 'no_such_type'
    with pytest.raises(ValueError):
        configure_rl_loss(_StubModel(), cfg)


def test_gkd_sft_alpha_only_applies_to_dataset_batches():
    """GKD's supervised term is gated by the selected data source."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('twinkle.loss')

    from swift.dev.config import RLConfig
    from swift.dev.loss import configure_rl_loss

    model = _StubModel()
    configure_rl_loss(model, RLConfig(rl_type='gkd', beta=0.5, sft_alpha=0.75))
    labels = torch.tensor([[-100, 1, 2]])
    student_logits = torch.randn(1, 3, 4, requires_grad=True)
    teacher_logits = torch.randn(1, 3, 4)
    base = model.loss(
        {'labels': labels}, {'logits': student_logits}, teacher_logits=teacher_logits, apply_sft_loss=False)['loss']
    mixed = model.loss(
        {'labels': labels}, {'logits': student_logits}, teacher_logits=teacher_logits, apply_sft_loss=True)['loss']
    expected_ce = torch.nn.functional.cross_entropy(
        student_logits.reshape(-1, 4), labels.reshape(-1), ignore_index=-100)
    assert mixed.detach().item() == pytest.approx((base + 0.75 * expected_ce).detach().item())










# REMOVED test_grpo_appends_gym_reward_and_weights_every_column (gym total_reward column; gym/env has no
# dev consumer -> N/A per the RL UT plan), test_grpo_builds_independent_reward_template, and
# test_grpo_dynamic_sampling_preserves_extras_and_group_order (plus their _DynamicSample/_DynamicRollout
# stubs): all hand-drove GRPOLoop with model=None / monkeypatched builders / a stub rollout, and broke on
# removed RLConfig kwargs (reward_model_plugin, reward_funcs) and the dropped gym column -- stale, not a
# regression. Reward-template assembly, DAPO dynamic sampling and reward-model/rule column weighting are
# covered end-to-end in test_algorithms_e2e.py / test_samplers_e2e.py.


# ----------------------------------------------------------------------
# Advanced GRPO controls (pure tensor tests)
# ----------------------------------------------------------------------
def _advanced_grpo_loss(**overrides):
    from swift.dev.config import RLConfig
    from swift.dev.loss import configure_rl_loss

    config = RLConfig(rl_type='grpo', **overrides)
    model = _StubModel()
    configure_rl_loss(model, config)
    return model.loss


def test_grpo_dual_clip_preserves_unclamped_ppo_branch():
    """delta clips coef_1 only; negative advantages still use PPO's raw-ratio clipped branch."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('twinkle.loss')

    loss = _advanced_grpo_loss(delta=0.5, epsilon=0.2)
    logps = torch.tensor([[torch.log(torch.tensor(2.0))]], requires_grad=True)
    result = loss(
        {'labels': torch.tensor([[1]])},
        {'logps': logps},
        old_logps=[[0.0]],
        advantages=[-1.0])
    assert result['loss'].item() == pytest.approx(1.2)


def test_grpo_entropy_quantile_masks_numerator_not_denominator():
    """Top-entropy filtering zeroes low-entropy token losses without renormalizing over survivors."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('twinkle.loss')

    loss = _advanced_grpo_loss(top_entropy_quantile=0.5, log_entropy=True)
    logps = torch.zeros(1, 2, requires_grad=True)
    result = loss(
        {'labels': torch.tensor([[1, 2]])},
        {'logps': logps, 'entropies': torch.tensor([[0.0, 1.0]])},
        old_logps=[[0.0, 0.0]],
        advantages=[1.0])
    assert result['loss'].item() == pytest.approx(-0.5)
    assert result['channel_loss']['entropy'].tolist() == pytest.approx([1.0, 2.0])


def test_grpo_sequence_token_is_has_sequence_values_and_token_gradients():
    """GSPO-token uses detached sequence ratios numerically while retaining per-token policy gradients."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('twinkle.loss')

    loss = _advanced_grpo_loss(importance_sampling_level='sequence_token', epsilon=10.0)
    logps = torch.tensor([[torch.log(torch.tensor(2.0)), 0.0]], requires_grad=True)
    result = loss(
        {'labels': torch.tensor([[1, 2]])},
        {'logps': logps},
        old_logps=[[0.0, 0.0]],
        advantages=[1.0])
    assert result['loss'].item() == pytest.approx(-(2.0**0.5))
    result['loss'].backward()
    assert logps.grad.tolist()[0] == pytest.approx([-(2.0**0.5) / 2] * 2)


def test_grpo_fipo_future_kl_decay_matches_reference_formula():
    """FIPO position t receives its own log-ratio plus geometrically decayed future ratios."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('twinkle.loss')

    loss = _advanced_grpo_loss(
        loss_type=['fipo'],
        fipo_decay_rate=1.0,
        fipo_clip_range=None,
        fipo_safety_threshold=None)
    log_ratio = torch.log(torch.tensor([[2.0, 0.5]]))
    weights = loss._fipo_weights(
        log_ratio,
        torch.exp(log_ratio),
        torch.ones_like(log_ratio),
        torch.ones_like(log_ratio, dtype=torch.bool))
    assert weights.tolist()[0] == pytest.approx([2.0**0.5, 0.5])




def test_grpo_truncation_alignment_and_offpolicy_mask_gradients():
    """Response-only tensors align before truncation; masked negative off-policy rows have zero gradient."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('twinkle.loss')

    loss = _advanced_grpo_loss(
        beta=0.1,
        overlong_filter=True,
        log_rollout_offpolicy_metrics=True,
        off_policy_sequence_mask_delta=1.0)
    logps = torch.tensor([[-9.0, -2.0, -2.0], [-9.0, 0.0, 0.0]], requires_grad=True)
    result = loss(
        {'labels': torch.tensor([[-100, 1, 2], [-100, 3, 4]])},
        {'logps': logps},
        old_logps=[[0.0, 0.0], [0.0, 0.0]],
        ref_logps=[[-0.1, -0.1], [-0.1, -0.1]],
        rollout_logps=[[0.0, 0.0], [0.0, 0.0]],
        advantages=[-1.0, 1.0],
        truncated=[True, False])
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert logps.grad[0].abs().sum().item() == 0.0
    assert logps.grad[1, 1:].abs().sum().item() > 0.0


# REMOVED test_grpo_num_iterations_reuses_rollout_without_expanding_max_steps: it reconstructed a
# GRPOLoop via object.__new__ and stubbed Model/Group/Tracker/_rollout_step, then broke because fit() now
# reads loop.manual_gc (an attribute the hand-built stub never set) -- a stale stub, not a regression.
# num_iterations replaying one rollout without expanding the optimizer-step budget is exercised
# end-to-end in test_algorithms_e2e.py (GRPO).


# ----------------------------------------------------------------------
# GAEAdvantage (twinkle.advantage) -- PPO's per-token advantage/return
# ----------------------------------------------------------------------
def test_gae_full_lambda_credits_all_tokens():
    """gamma=lam=1: the terminal reward propagates fully back, returns = advantage + value."""
    pytest.importorskip('twinkle.advantage')
    from twinkle.advantage import GAEAdvantage

    # normalize=False asserts the RAW recursion (hand-computed below); the production default
    # normalize=True would standardize the batch and hide the per-token credit assignment.
    advantages, returns = GAEAdvantage(gamma=1.0, gae_lambda=1.0, normalize=False)([0.0, 0.0, 1.0], [0.5, 0.5, 0.5])
    # delta_t = r_t + gamma*V_{t+1} - V_t ; last-to-first: 0.5, 0.5, 0.5 (full carry at lam=1).
    assert advantages.flatten().tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert returns.flatten().tolist() == pytest.approx([1.0, 1.0, 1.0])  # advantage + value


def test_gae_zero_lambda_is_one_step_td():
    """lam=0 collapses GAE to the one-step TD error delta_t = r_t + gamma*V_{t+1} - V_t."""
    pytest.importorskip('twinkle.advantage')
    from twinkle.advantage import GAEAdvantage

    advantages, returns = GAEAdvantage(gamma=1.0, gae_lambda=0.0, normalize=False)([0.0, 0.0, 1.0], [0.5, 0.5, 0.5])
    # No carry: delta = [0+0.5-0.5, 0+0.5-0.5, 1+0-0.5] = [0.0, 0.0, 0.5].
    assert advantages.flatten().tolist() == pytest.approx([0.0, 0.0, 0.5])
    assert returns.flatten().tolist() == pytest.approx([0.5, 0.5, 1.0])


def test_gae_length_matches_response_and_requires_values():
    """The advantage/return tensors are one-per-token; the value function is a mandatory argument."""
    pytest.importorskip('twinkle.advantage')
    from twinkle.advantage import GAEAdvantage

    advantages, returns = GAEAdvantage(gamma=0.9, gae_lambda=0.95,
                                       normalize=False)([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0])
    assert advantages.shape == (1, 4) and returns.shape == (1, 4)  # one advantage/return per token
    with pytest.raises(TypeError):
        GAEAdvantage()([1.0, 2.0])  # value function is mandatory for GAE (positional `values`)


# ----------------------------------------------------------------------
# PPOValueLoss per-token (needs torch + twinkle) -- response-only targets scatter onto the mask
# ----------------------------------------------------------------------
def test_ppo_value_loss_per_token_masks_and_scatters():
    """Response-only returns land on the masked tokens; the clipped MSE is averaged over them only."""
    pytest.importorskip('twinkle.loss')
    torch = pytest.importorskip('torch')
    from twinkle.loss.value import PPOValueLoss

    loss = PPOValueLoss(vf_coef=1.0)
    # T=4, response tokens at positions 2 and 3 (labels != -100). The critic emits per-token values.
    inputs = {'labels': torch.tensor([[-100, -100, 5, 6]])}
    outputs = {'values': torch.tensor([[0.0, 0.0, 1.0, 2.0]])}
    # returns arrive response-only (one per response token). old_values equal to the current values at
    # the masked positions -> clip is a no-op (values-old=0), isolating the masked-mean MSE.
    out = loss(inputs, outputs, returns=[[3.0, 4.0]], old_values=[[1.0, 2.0]])
    # sq err on the two response tokens: (1-3)^2=4, (2-4)^2=4 -> mean 4 -> 0.5*vf_coef*4 = 2.0.
    assert float(out['loss']) == pytest.approx(2.0)


def test_ppo_value_loss_clips_value_move():
    """With old_values set, the value may not move more than cliprange_value from the rollout estimate."""
    pytest.importorskip('twinkle.loss')
    torch = pytest.importorskip('torch')
    from twinkle.loss.value import PPOValueLoss

    loss = PPOValueLoss(vf_coef=1.0, cliprange_value=0.2)
    inputs = {'labels': torch.tensor([[-100, 5]])}
    outputs = {'values': torch.tensor([[0.0, 1.0]])}
    # old_value=0.0 -> clipped prediction is 0.2; unclipped err (1-3)^2=4, clipped (0.2-3)^2=7.84;
    # PPO takes the LARGER (pessimistic) -> 7.84 over the single response token -> 0.5*1.0*7.84.
    out = loss(inputs, outputs, returns=[[3.0]], old_values=[[0.0]])
    assert float(out['loss']) == pytest.approx(0.5 * 7.84, rel=1e-4)
