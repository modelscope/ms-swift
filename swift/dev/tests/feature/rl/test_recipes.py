"""RL recipe unit tests (cheap tier).

Pure-function / pure-Python coverage that runs in the normal suite without Ray, vLLM or GPUs:
  - :func:`plan_rl_device_groups` colocate vs heterogeneous placement + its validation;
  - :class:`PreferenceLoop` chosen/rejected interleaving (the layout the DPO family's split relies on);
  - :class:`GKDLoop` prompt-window wrap-around;
  - :func:`configure_rlhf_loss` rlhf_type -> loss mapping (skipped when twinkle is unavailable).

The heavy weight-sync / colocate / e2e paths need Ray + multi-GPU + vLLM and live behind
``@pytest.mark.slow`` mirroring ``twinkle/tests/sampler/test_weight_sync.py``; they are not here.
"""
import pytest


# ----------------------------------------------------------------------
# plan_rl_device_groups (pure function, no twinkle)
# ----------------------------------------------------------------------
def test_plan_device_groups_colocate_shares_one_group():
    """colocate: trainer + sampler share ONE 'model' group over the trainer's GPUs; colocate=True."""
    from swift.dev.recipe.run_grpo import plan_rl_device_groups

    groups, sampler_group, colocate = plan_rl_device_groups(4, 'colocate', 2)
    assert groups == [('model', [0, 1, 2, 3])]
    assert sampler_group == 'model'  # sampler placed in the shared group
    assert colocate is True


def test_plan_device_groups_heterogeneous_disjoint_ranks():
    """server/None: a 'model' group then a DISJOINT 'sampler' group after it; colocate=False."""
    from swift.dev.recipe.run_grpo import plan_rl_device_groups

    groups, sampler_group, colocate = plan_rl_device_groups(4, 'server', 2)
    assert groups == [('model', [0, 1, 2, 3]), ('sampler', [4, 5])]
    assert sampler_group == 'sampler'
    assert colocate is False
    # ranks must not overlap (NCCL weight-sync requires distinct devices).
    model_ranks, sampler_ranks = groups[0][1], groups[1][1]
    assert set(model_ranks).isdisjoint(sampler_ranks)


def test_plan_device_groups_validation():
    """Colocate that cannot fit the sampler, and non-positive counts, are rejected."""
    from swift.dev.recipe.run_grpo import plan_rl_device_groups

    with pytest.raises(ValueError):
        plan_rl_device_groups(2, 'colocate', 4)  # sampler bigger than the shared trainer GPUs
    with pytest.raises(ValueError):
        plan_rl_device_groups(0, 'server', 1)  # no trainer GPUs
    with pytest.raises(ValueError):
        plan_rl_device_groups(2, 'server', 0)  # no sampler GPUs


# ----------------------------------------------------------------------
# PreferenceLoop interleaving (pure Python; stub template/model)
# ----------------------------------------------------------------------
class _StubTemplate:
    """A template whose encode returns a fixed preference-encoded dict (chosen_*/rejected_*)."""

    def __init__(self, encoded):
        self._encoded = encoded

    def encode(self, row):
        return dict(self._encoded)


def _preference_loop(rlhf_type, encoded):
    from swift.dev.recipe.run_dpo import PreferenceLoop

    return PreferenceLoop(model=None, dataloader=None, template=_StubTemplate(encoded), rlhf_type=rlhf_type)


def test_preference_interleave_order_and_prefix_strip():
    """Two rows -> [chosen_1, rejected_1, chosen_2, rejected_2] with prefixes stripped, `length` dropped."""
    loop = _preference_loop(
        'dpo', {
            'chosen_input_ids': [1, 2],
            'chosen_labels': [-100, 2],
            'chosen_length': 2,
            'rejected_input_ids': [3, 4],
            'rejected_labels': [-100, 4],
            'rejected_length': 2,
        })
    features = loop._interleave([{'x': 0}, {'x': 1}])
    assert len(features) == 4  # 2 pairs -> 4 interleaved features (even, as the DPO split needs)
    assert features[0] == {'input_ids': [1, 2], 'labels': [-100, 2]}  # chosen_1
    assert features[1] == {'input_ids': [3, 4], 'labels': [-100, 4]}  # rejected_1
    assert features[2] == features[0] and features[3] == features[1]
    assert 'length' not in features[0]  # bookkeeping stripped


def test_preference_reward_pair_has_no_labels():
    """RM (seq_cls) encodes without labels -> the pair features carry only input_ids (RewardLoss reads
    logits, not logps)."""
    loop = _preference_loop('rm', {'chosen_input_ids': [1, 2], 'rejected_input_ids': [3, 4]})
    features = loop._interleave([{'x': 0}])
    assert features == [{'input_ids': [1, 2]}, {'input_ids': [3, 4]}]
    assert loop._is_reward is True


def test_preference_missing_side_is_fatal():
    """A row that did not encode a rejected sequence must fail loudly (paired data is required)."""
    loop = _preference_loop('dpo', {'chosen_input_ids': [1, 2], 'chosen_labels': [-100, 2]})
    with pytest.raises(ValueError):
        loop._interleave([{'x': 0}])


# ----------------------------------------------------------------------
# GKDLoop prompt window (pure Python)
# ----------------------------------------------------------------------
def test_gkd_prompt_batch_wraps():
    """The per-step prompt window rolls forward and wraps around the prompt list."""
    from swift.dev.recipe.run_gkd import GKDLoop

    prompts = [[{'role': 'user', 'content': str(i)}] for i in range(3)]
    loop = GKDLoop(model=None, teacher=None, template=None, prompts=prompts, batch_size=2)
    assert loop._prompt_batch(0) == [prompts[0], prompts[1]]
    assert loop._prompt_batch(1) == [prompts[2], prompts[0]]  # wraps at 3


# ----------------------------------------------------------------------
# configure_rlhf_loss mapping (needs twinkle's loss registry)
# ----------------------------------------------------------------------
class _StubModel:
    """Captures the loss instance set via set_loss so the mapping can be asserted."""

    def __init__(self):
        self.loss = None

    def set_loss(self, loss):
        self.loss = loss


@pytest.mark.parametrize('rlhf_type, cls_name', [
    ('dpo', 'DPOLoss'),
    ('kto', 'DPOLoss'),
    ('cpo', 'CPOLoss'),
    ('orpo', 'ORPOLoss'),
    ('simpo', 'SimPOLoss'),
    ('grpo', 'GRPOLoss'),
    ('gkd', 'GKDLoss'),
    ('rm', 'RewardLoss'),
    ('ppo', 'GRPOLoss'),
])
def test_configure_rlhf_loss_maps_type_to_loss(rlhf_type, cls_name):
    """Each rlhf_type resolves to its twinkle loss class (ppo's POLICY loss is the shared GRPO clip)."""
    pytest.importorskip('twinkle.loss')
    from swift.dev.config import RLHFConfig
    from swift.dev.loss import configure_rlhf_loss

    model = _StubModel()
    configure_rlhf_loss(model, RLHFConfig(rlhf_type=rlhf_type))
    assert type(model.loss).__name__ == cls_name


def test_configure_ppo_value_loss_sets_value_loss():
    """PPO's critic gets the clipped value loss, carrying cliprange_value / vf_coef."""
    pytest.importorskip('twinkle.loss')
    from swift.dev.config import RLHFConfig
    from swift.dev.loss import configure_ppo_value_loss

    value_model = _StubModel()
    configure_ppo_value_loss(value_model, RLHFConfig(rlhf_type='ppo', cliprange_value=0.3, vf_coef=0.5))
    assert type(value_model.loss).__name__ == 'PPOValueLoss'
    assert value_model.loss.cliprange_value == 0.3
    assert value_model.loss.vf_coef == 0.5


def test_configure_rlhf_loss_refuses_unknown():
    """An unknown rlhf_type is a ValueError (ppo is now supported, so it is no longer refused)."""
    pytest.importorskip('twinkle.loss')
    from swift.dev.config import RLHFConfig
    from swift.dev.loss import configure_rlhf_loss

    cfg = RLHFConfig(rlhf_type='dpo')
    cfg.rlhf_type = 'no_such_type'
    with pytest.raises(ValueError):
        configure_rlhf_loss(_StubModel(), cfg)


# ----------------------------------------------------------------------
# GAEAdvantage (twinkle.advantage) -- PPO's per-token advantage/return
# ----------------------------------------------------------------------
def test_gae_full_lambda_credits_all_tokens():
    """gamma=lam=1: the terminal reward propagates fully back, returns = advantage + value."""
    pytest.importorskip('twinkle.advantage')
    from twinkle.advantage import GAEAdvantage

    advantages, returns = GAEAdvantage()([0.0, 0.0, 1.0], [0.5, 0.5, 0.5], gamma=1.0, lam=1.0)
    assert advantages == pytest.approx([0.5, 0.5, 0.5])
    assert returns == pytest.approx([1.0, 1.0, 1.0])


def test_gae_zero_lambda_is_one_step_td():
    """lam=0 collapses GAE to the one-step TD error delta_t = r_t + gamma*V_{t+1} - V_t."""
    pytest.importorskip('twinkle.advantage')
    from twinkle.advantage import GAEAdvantage

    advantages, returns = GAEAdvantage()([0.0, 0.0, 1.0], [0.5, 0.5, 0.5], gamma=1.0, lam=0.0)
    assert advantages == pytest.approx([0.0, 0.0, 0.5])
    assert returns == pytest.approx([0.5, 0.5, 1.0])


def test_gae_length_matches_response_and_requires_values():
    """The advantage/return lists are one-per-token; missing values is a hard error."""
    pytest.importorskip('twinkle.advantage')
    from twinkle.advantage import GAEAdvantage

    advantages, returns = GAEAdvantage()([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], gamma=0.9, lam=0.95)
    assert len(advantages) == 4 and len(returns) == 4
    with pytest.raises(ValueError):
        GAEAdvantage()([1.0, 2.0])  # value function is mandatory for GAE


# ----------------------------------------------------------------------
# PPOValueLoss per-token (needs torch + twinkle) -- response-only targets scatter onto the mask
# ----------------------------------------------------------------------
def test_ppo_value_loss_per_token_masks_and_scatters():
    """Response-only returns land on the masked tokens; the clipped MSE is averaged over them only."""
    pytest.importorskip('twinkle.loss')
    torch = pytest.importorskip('torch')
    from twinkle.loss.value import PPOValueLoss

    loss = PPOValueLoss(vf_coef=1.0)
    # T=4, response tokens at positions 2 and 3 (labels != -100). The critic emits per-token values in
    # logits (task='value' skips pooling).
    inputs = {'labels': torch.tensor([[-100, -100, 5, 6]])}
    outputs = {'logits': torch.tensor([[0.0, 0.0, 1.0, 2.0]])}
    # returns arrive response-only (one per response token); old_values=None -> no clipping.
    out = loss(inputs, outputs, returns=[[3.0, 4.0]], old_values=None)
    # sq err on the two response tokens: (1-3)^2=4, (2-4)^2=4 -> mean 4 -> 0.5*vf_coef*4 = 2.0.
    assert float(out.loss) == pytest.approx(2.0)


def test_ppo_value_loss_clips_value_move():
    """With old_values set, the value may not move more than cliprange_value from the rollout estimate."""
    pytest.importorskip('twinkle.loss')
    torch = pytest.importorskip('torch')
    from twinkle.loss.value import PPOValueLoss

    loss = PPOValueLoss(vf_coef=1.0, cliprange_value=0.2)
    inputs = {'labels': torch.tensor([[-100, 5]])}
    outputs = {'logits': torch.tensor([[0.0, 1.0]])}
    # old_value=0.0 -> clipped prediction is 0.2; unclipped err (1-3)^2=4, clipped (0.2-3)^2=7.84;
    # PPO takes the LARGER (pessimistic) -> 7.84 over the single response token -> 0.5*1.0*7.84.
    out = loss(inputs, outputs, returns=[[3.0]], old_values=[[0.0]])
    assert float(out.loss) == pytest.approx(0.5 * 7.84, rel=1e-4)
