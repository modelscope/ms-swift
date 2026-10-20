"""S8a — RL numeric-kernel hand-computed oracles (L1, cheap tier).

Pure-function / pure-tensor coverage that runs without Ray, vLLM or GPUs. Each test pins a
hand-computed oracle against the CURRENT dev2 API (the v5 snapshot's GRPOLoop / plan_rl_device_groups /
recipe.preference surface moved: GRPOLoop -> recipe.grpo.assembly module functions + GRPOAssembler,
plan_rl_device_groups -> recipe.utils.plan_device_groups, recipe.preference -> recipe.rlhf,
RLHFConfig -> RLConfig). Maps to ut_plan S8-01..24.

The heavy weight-sync / colocate / e2e paths live in the S7 RL e2e batch; the reward-protocol kernels
(get_reward_funcs / compute-rewards-per-func / rollout encode) live in test_rollout.py (S8b).
"""
from types import SimpleNamespace

import pytest


# ----------------------------------------------------------------------
# S8-01/02/03 — plan_device_groups (pure function, no twinkle)
# ----------------------------------------------------------------------
def test_plan_device_groups_colocate_overlays_sampler_on_trainer_gpus():
    """S8-01: colocate overlays trainer + sampler in ONE rank space over the trainer's GPUs.

    The sampler group reuses ranks ``[0, sampler_ws)`` (its own worker-rank space, NOT the trainer's
    group), so the two groups share physical devices -- that is what colocate means. colocate=True.
    """
    from swift.dev.recipe.utils import plan_device_groups

    groups, colocate = plan_device_groups(4, 'colocate', 2, [])
    assert groups == [('model', [0, 1, 2, 3]), ('sampler', [0, 1])]
    assert colocate is True


def test_plan_device_groups_disaggregated_disjoint_ranks():
    """S8-02: any non-colocate mode appends a DISJOINT sampler group after the trainer's ranks.

    NCCL weight-sync requires distinct devices, so the model and sampler ranks must not overlap.
    Auxiliary frozen-model groups (ref/teacher/reward) are appended last, disjoint, in order.
    """
    from swift.dev.recipe.utils import plan_device_groups

    groups, colocate = plan_device_groups(4, 'disaggregated', 2, [])
    assert groups == [('model', [0, 1, 2, 3]), ('sampler', [4, 5])]
    assert colocate is False
    assert set(groups[0][1]).isdisjoint(groups[1][1])

    # An auxiliary frozen model gets its own disjoint group after model+sampler.
    groups_aux, _ = plan_device_groups(4, 'disaggregated', 2, [('ref', 2)])
    assert groups_aux == [('model', [0, 1, 2, 3]), ('sampler', [4, 5]), ('ref', [6, 7])]


def test_plan_device_groups_validation():
    """S8-03: a colocate sampler that cannot fit, and non-positive counts, are rejected."""
    from swift.dev.recipe.utils import plan_device_groups

    with pytest.raises(ValueError):
        plan_device_groups(2, 'colocate', 4, [])  # sampler wider than the shared trainer GPUs
    with pytest.raises(ValueError):
        plan_device_groups(0, 'disaggregated', 1, [])  # no trainer GPUs
    with pytest.raises(ValueError):
        plan_device_groups(2, 'disaggregated', 0, [])  # no sampler GPUs


# ----------------------------------------------------------------------
# S8-04/05/06 — PreferenceBatchEncoder interleaving (pure Python; stub template)
# ----------------------------------------------------------------------
class _StubTemplate:
    """A template whose encode returns a fixed preference-encoded dict (chosen_*/rejected_*)."""

    def __init__(self, encoded):
        self._encoded = encoded

    def encode(self, row):
        return dict(self._encoded)


def _preference_encoder(rl_type, encoded):
    from swift.dev.recipe.rlhf.encoder import PreferenceBatchEncoder

    return PreferenceBatchEncoder(model=None, template=_StubTemplate(encoded), rl_type=rl_type)


def test_preference_interleave_order_and_prefix_strip():
    """S8-04: two rows -> [chosen_1, rejected_1, chosen_2, rejected_2], prefixes stripped, length dropped."""
    encoder = _preference_encoder(
        'dpo', {
            'chosen_input_ids': [1, 2],
            'chosen_labels': [-100, 2],
            'chosen_length': 2,
            'rejected_input_ids': [3, 4],
            'rejected_labels': [-100, 4],
            'rejected_length': 2,
        })
    features = encoder._interleave([{'x': 0}, {'x': 1}])
    assert len(features) == 4  # 2 pairs -> 4 interleaved features (even, as the DPO split needs)
    assert features[0] == {'input_ids': [1, 2], 'labels': [-100, 2]}  # chosen_1
    assert features[1] == {'input_ids': [3, 4], 'labels': [-100, 4]}  # rejected_1
    assert features[2] == features[0] and features[3] == features[1]
    assert 'length' not in features[0]  # per-side bookkeeping stripped


def test_preference_reward_pair_has_no_labels():
    """S8-05: RM (seq_cls) encodes without labels -> the pair carries only input_ids (RewardLoss reads logits)."""
    encoder = _preference_encoder('rm', {'chosen_input_ids': [1, 2], 'rejected_input_ids': [3, 4]})
    features = encoder._interleave([{'x': 0}])
    assert features == [{'input_ids': [1, 2]}, {'input_ids': [3, 4]}]
    assert encoder._is_reward is True


def test_preference_missing_side_is_fatal():
    """S8-06: a row that did not encode a rejected sequence must fail loudly (paired data is required)."""
    encoder = _preference_encoder('dpo', {'chosen_input_ids': [1, 2], 'chosen_labels': [-100, 2]})
    with pytest.raises(ValueError):
        encoder._interleave([{'x': 0}])


# ----------------------------------------------------------------------
# S8-07/08 — configure_rl_loss / configure_ppo_value_loss mapping (needs twinkle's loss registry)
# ----------------------------------------------------------------------
class _StubModel:
    """Captures the loss(es) set via set_loss so the mapping can be asserted.

    configure_rl_loss calls ``set_loss(loss)`` for a single objective, but the CHORD grpo variant keys
    two losses ('grpo' policy + 'sft' cross-entropy); ``.loss`` resolves the primary either way.
    """

    def __init__(self):
        self.losses = {}

    def set_loss(self, loss, loss_key=None):
        self.losses[loss_key] = loss

    @property
    def loss(self):
        return self.losses.get(None, self.losses.get('grpo'))


@pytest.mark.parametrize('rl_type, cls_name', [
    ('dpo', 'DPOLoss'),
    ('kto', 'KTOLoss'),
    ('cpo', 'CPOLoss'),
    ('orpo', 'ORPOLoss'),
    ('simpo', 'SimPOLoss'),
    ('grpo', 'GRPOLoss'),
    ('gkd', 'GKDLoss'),
    ('rm', 'RewardLoss'),
    ('ppo', 'GRPOLoss'),  # PPO's POLICY loss is the shared GRPO clip surrogate
    ('opsd', 'OPSDLoss'),
    ('mopd', 'MOPDLoss'),
])
def test_configure_rl_loss_maps_type_to_loss(rl_type, cls_name):
    """S8-07: every rl_type resolves to its twinkle loss class via the _RL_LOSS_NAME table."""
    pytest.importorskip('twinkle.loss')
    from swift.dev.config import RLConfig
    from swift.dev.loss import configure_rl_loss

    model = _StubModel()
    configure_rl_loss(model, RLConfig(rl_type=rl_type))
    assert type(model.loss).__name__ == cls_name


def test_configure_ppo_value_loss_sets_value_loss():
    """S8-08 (critic half): PPO's critic gets the clipped value loss carrying cliprange_value / vf_coef."""
    pytest.importorskip('twinkle.loss')
    from swift.dev.config import RLConfig
    from swift.dev.loss import configure_ppo_value_loss

    value_model = _StubModel()
    configure_ppo_value_loss(value_model, RLConfig(rl_type='ppo', cliprange_value=0.3, vf_coef=0.5))
    assert type(value_model.loss).__name__ == 'PPOValueLoss'
    assert value_model.loss.cliprange_value == 0.3
    assert value_model.loss.vf_coef == 0.5


def test_configure_rl_loss_refuses_unknown():
    """S8-07 guard: an unknown rl_type is a ValueError."""
    pytest.importorskip('twinkle.loss')
    from swift.dev.config import RLConfig
    from swift.dev.loss import configure_rl_loss

    cfg = RLConfig(rl_type='dpo')
    cfg.rl_type = 'no_such_type'
    with pytest.raises(ValueError):
        configure_rl_loss(_StubModel(), cfg)


def test_gkd_sft_alpha_only_applies_to_dataset_batches():
    """GKD's supervised CE term is gated by apply_sft_loss (the selected data source), weighted by sft_alpha."""
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


def test_advanced_grpo_loss_is_serializable_and_combines_objectives():
    """S8-08: the CHORD/SDAR grpo loss is a real module-level Loss, picklable, and runs a mixed RL+SFT batch."""
    import pickle

    torch = pytest.importorskip('torch')
    pytest.importorskip('twinkle.loss')
    from twinkle.loss import Loss

    from swift.dev.config import RLConfig
    from swift.dev.loss import configure_rl_loss

    model = _StubModel()
    configure_rl_loss(
        model,
        RLConfig(rl_type='grpo', chord_sft_dataset=['dummy'], sdar_loss_coef=0.25, sdar_gate_beta=2.0))
    grpo_loss = model.losses['grpo']  # chord keys the policy loss under 'grpo' (+ a 'sft' CE loss)
    assert isinstance(grpo_loss, Loss)
    assert 'sft' in model.losses  # the auxiliary supervised objective is registered alongside
    restored = pickle.loads(pickle.dumps(grpo_loss))
    assert type(restored) is type(grpo_loss)

    labels = torch.tensor([[-100, 1, 2], [-100, 2, 3]])
    logps = torch.tensor([[0.0, -0.7, -0.8], [0.0, -0.4, -0.5]], requires_grad=True)
    logits = torch.randn(2, 3, 4, requires_grad=True)
    # A consistent 2-sequence batch: 2 label rows -> 2 advantages, and old/ref/teacher_logps carry one
    # response-only row (2 response tokens) per sequence.
    result = restored(
        {'labels': labels},
        {'logps': logps, 'logits': logits},
        old_logps=[[-0.9, -1.0], [-0.85, -0.95]],
        ref_logps=[[-0.8, -0.9], [-0.75, -0.85]],
        teacher_logps=[[-0.6, -0.7], [-0.55, -0.65]],
        advantages=[1.0, 0.5],
        chord_count=1,
        chord_mu=0.2,
        chord_phi=True)
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    # The GRPO policy surrogate consumes the per-token ``logps`` (SDAR's separate logit path is covered
    # by the distill-signal e2e in S7), so the policy gradient must reach logps.
    assert logps.grad is not None


# ----------------------------------------------------------------------
# S8-09 — response positions + reward-weight device following
# ----------------------------------------------------------------------
def test_grpo_tensor_alignment_and_reward_weights_keep_device():
    """S8-09: tensor labels never enter Python truth testing; weighted rewards follow the reward placement."""
    torch = pytest.importorskip('torch')
    from swift.dev.utils.scoring import response_positions, weighted_rewards

    labels = torch.tensor([-100, 7, 8, -100])
    assert response_positions({'labels': labels}) == [1, 2]

    rewards = torch.tensor([[4.0, 3.0], [8.0, float('nan')]])
    weights = torch.tensor([0.25, 2.0])
    weighted = weighted_rewards(rewards, weights)
    assert weighted.device == rewards.device
    # row0: 4*0.25 + 3*2.0 = 7.0 ; row1: 8*0.25 + nan*2.0 -> nansum = 2.0
    assert weighted.tolist() == pytest.approx([7.0, 2.0])


# ----------------------------------------------------------------------
# S8-10..15 — advanced GRPO controls (pure tensor tests against twinkle GRPOLoss)
# ----------------------------------------------------------------------
def _advanced_grpo_loss(**overrides):
    from swift.dev.config import RLConfig
    from swift.dev.loss import configure_rl_loss

    config = RLConfig(rl_type='grpo', **overrides)
    model = _StubModel()
    configure_rl_loss(model, config)
    return model.loss


def test_grpo_dual_clip_preserves_unclamped_ppo_branch():
    """S8-10: delta clips coef_1 only; negative advantages still use PPO's raw-ratio clipped branch."""
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
    """S8-11: top-entropy filtering zeroes low-entropy token losses without renormalizing over survivors."""
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
    """S8-12: GSPO-token uses detached sequence ratios numerically while retaining per-token policy gradients."""
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
    """S8-13: FIPO position t receives its own log-ratio plus geometrically decayed future ratios."""
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


@pytest.mark.parametrize('mode, expected', [
    ('token_truncate', [2.0, 0.5]),
    ('token_mask', [0.0, 0.5]),
    ('sequence_truncate', [2.0**0.5, 2.0**0.5]),
    ('sequence_mask', [4.0, 0.5]),
])
def test_grpo_rollout_importance_sampling_modes(mode, expected):
    """S8-14: all four rollout IS modes produce the documented token/sequence correction weights."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('twinkle.loss')

    loss = _advanced_grpo_loss(
        rollout_importance_sampling_mode=mode,
        rollout_importance_sampling_threshold=2.0)
    ratios = torch.tensor([[4.0, 0.5]])
    weights = loss._rollout_is_weights(torch.log(ratios), torch.ones_like(ratios, dtype=torch.bool))
    assert weights.tolist()[0] == pytest.approx(expected)


def test_grpo_truncation_alignment_and_offpolicy_mask_gradients():
    """S8-15: response-only tensors align before truncation; masked negative off-policy rows have zero gradient."""
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


# ----------------------------------------------------------------------
# S8-16/17/18 — GAEAdvantage (twinkle.advantage): PPO's per-token advantage/return
# ----------------------------------------------------------------------
def test_gae_full_lambda_credits_all_tokens():
    """S8-16: gamma=lam=1 -> the terminal reward propagates fully back; returns = advantage + value."""
    pytest.importorskip('twinkle.advantage')
    from twinkle.advantage import GAEAdvantage

    # normalize=False asserts the RAW recursion (hand-computed); the production default normalize=True
    # would standardize the batch and hide the per-token credit assignment.
    advantages, returns = GAEAdvantage(gamma=1.0, gae_lambda=1.0, normalize=False)([0.0, 0.0, 1.0], [0.5, 0.5, 0.5])
    assert advantages.flatten().tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert returns.flatten().tolist() == pytest.approx([1.0, 1.0, 1.0])  # advantage + value


def test_gae_zero_lambda_is_one_step_td():
    """S8-17: lam=0 collapses GAE to the one-step TD error delta_t = r_t + gamma*V_{t+1} - V_t."""
    pytest.importorskip('twinkle.advantage')
    from twinkle.advantage import GAEAdvantage

    advantages, returns = GAEAdvantage(gamma=1.0, gae_lambda=0.0, normalize=False)([0.0, 0.0, 1.0], [0.5, 0.5, 0.5])
    # No carry: delta = [0+0.5-0.5, 0+0.5-0.5, 1+0-0.5] = [0.0, 0.0, 0.5].
    assert advantages.flatten().tolist() == pytest.approx([0.0, 0.0, 0.5])
    assert returns.flatten().tolist() == pytest.approx([0.5, 0.5, 1.0])


def test_gae_length_matches_response_and_requires_values():
    """S8-18: advantage/return tensors are one-per-token; the value function is a mandatory argument."""
    pytest.importorskip('twinkle.advantage')
    from twinkle.advantage import GAEAdvantage

    advantages, returns = GAEAdvantage(gamma=0.9, gae_lambda=0.95,
                                       normalize=False)([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0])
    assert advantages.shape == (1, 4) and returns.shape == (1, 4)  # one advantage/return per token
    with pytest.raises(TypeError):
        GAEAdvantage()([1.0, 2.0])  # value function is mandatory for GAE (positional `values`)


# ----------------------------------------------------------------------
# S8-19/20 — PPOValueLoss per-token (twinkle.loss.value): response-only targets scatter onto the mask
# ----------------------------------------------------------------------
def test_ppo_value_loss_per_token_masks_and_scatters():
    """S8-19: response-only returns land on the masked tokens; the clipped MSE is averaged over them only."""
    pytest.importorskip('twinkle.loss')
    torch = pytest.importorskip('torch')
    from twinkle.loss.value import PPOValueLoss

    loss = PPOValueLoss(vf_coef=1.0)
    # T=4, response tokens at positions 2 and 3 (labels != -100). old_values == current values at the
    # masked positions -> clip is a no-op, isolating the masked-mean MSE.
    inputs = {'labels': torch.tensor([[-100, -100, 5, 6]])}
    outputs = {'values': torch.tensor([[0.0, 0.0, 1.0, 2.0]])}
    out = loss(inputs, outputs, returns=[[3.0, 4.0]], old_values=[[1.0, 2.0]])
    # sq err on the two response tokens: (1-3)^2=4, (2-4)^2=4 -> mean 4 -> 0.5*vf_coef*4 = 2.0.
    assert float(out['loss']) == pytest.approx(2.0)


def test_ppo_value_loss_clips_value_move():
    """S8-20: with old_values set, the value may not move more than cliprange_value from the rollout estimate."""
    pytest.importorskip('twinkle.loss')
    torch = pytest.importorskip('torch')
    from twinkle.loss.value import PPOValueLoss

    loss = PPOValueLoss(vf_coef=1.0, cliprange_value=0.2)
    inputs = {'labels': torch.tensor([[-100, 5]])}
    outputs = {'values': torch.tensor([[0.0, 1.0]])}
    # old_value=0.0 -> clipped prediction 0.2; unclipped err (1-3)^2=4, clipped (0.2-3)^2=7.84; PPO takes
    # the LARGER (pessimistic) -> 7.84 over the single response token -> 0.5*1.0*7.84.
    out = loss(inputs, outputs, returns=[[3.0]], old_values=[[0.0]])
    assert float(out['loss']) == pytest.approx(0.5 * 7.84, rel=1e-4)


# ----------------------------------------------------------------------
# S8-21..24 — reward weighting / metric accumulation (rft + terminal-batch scoring contracts)
# ----------------------------------------------------------------------
def test_weighted_reward_nansum_none_does_not_poison():
    """S8-21: one row's weighted reward sum is nan-aware -- an unparseable (None) verdict never poisons it.

    ``weights`` is the tensor build_reward_weights produces (the real caller's orm_weights), not a list.
    """
    torch = pytest.importorskip('torch')
    from swift.dev.recipe.grpo.assembly import weighted_reward

    weights = torch.tensor([0.25, 2.0])
    assert weighted_reward([4.0, 3.0], weights) == pytest.approx(7.0)  # 4*0.25 + 3*2.0
    assert weighted_reward([8.0, None], weights) == pytest.approx(2.0)  # None->nan dropped: 8*0.25
    assert weighted_reward([1.0, 2.0], None) == pytest.approx(3.0)  # no weights -> plain sum


def test_build_reward_weights_single_source():
    """S8-22: None -> all-ones; a length mismatch against the scorer count is a ValueError."""
    pytest.importorskip('torch')
    from swift.dev.utils.scoring import build_reward_weights

    assert build_reward_weights(None, 3).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert build_reward_weights([0.5, 2.0], 2).tolist() == pytest.approx([0.5, 2.0])
    with pytest.raises(ValueError):
        build_reward_weights([1.0], 2)  # one weight for two scorers


def test_accumulate_reward_metric_resets_then_accumulates():
    """S8-23: the metric is reset FIRST (reflects exactly this batch), then fed per-scorer columns + total."""
    torch = pytest.importorskip('torch')
    from swift.dev.utils.scoring import accumulate_reward_metric

    class Metric:

        def __init__(self):
            self.reset_count = 0
            self.payload = None

        def reset(self):
            self.reset_count += 1

        def accumulate(self, *, rewards, completion_lengths):
            self.payload = (rewards, completion_lengths)

    metric = Metric()
    rewards = torch.tensor([[1.0, 2.0], [3.0, float('nan')]])
    samples = [SimpleNamespace(response_token_ids=[1, 2, 3]), SimpleNamespace(response_token_ids=[4])]
    accumulate_reward_metric(metric, samples, rewards, scorer_names=['rule', 'rm'], reward_weights=[1.0, 2.0])

    assert metric.reset_count == 1  # reset before accumulate
    reward_cols, lengths = metric.payload
    assert reward_cols['rule'] == pytest.approx([1.0, 3.0])
    assert reward_cols['total'] == pytest.approx([5.0, 3.0])  # 1*1+2*2=5 ; 3*1+nansum(nan*2)=3
    assert lengths == [3, 1]


def test_weighted_rewards_columns_and_total():
    """S8-24: the [N, n_scorers] matrix collapses to one [N] weighted-sum reward; None weights = plain sum."""
    torch = pytest.importorskip('torch')
    from swift.dev.utils.scoring import weighted_rewards

    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert weighted_rewards(rewards, None).tolist() == pytest.approx([3.0, 7.0])  # all-ones -> sum
    assert weighted_rewards(rewards, [1.0, 0.5]).tolist() == pytest.approx([2.0, 5.0])  # 1+1 ; 3+2
