"""S8b — dev2 reward / advantage / channel protocol kernels (L1, cheap tier).

The v5 ``test_rollout.py`` targeted a ``swift.dev.rollout`` package and a dev-side
``compute_rewards_per_func`` that dev2 abandoned: the rollout layer (RolloutSample / MultiTurnRollout /
gym) moved to ``twinkle_agentic.rollout`` -- which has its OWN suite under ``twinkle/tests/twinkle_agentic``
-- and ``swift.rl_core`` is the LEGACY core (consumed by swift/rlhf_trainers, not by dev2). So the
rollout-INTERNAL cases (ut_plan S8-27/32/33/34/35/36) are covered by twinkle_agentic's suite, not
duplicated here; this file pins dev2's OWN seams -- reward resolution, the advantage delegation to
twinkle, the async-reward shell, and the merged ``--orm``/``--prm`` channel split/alignment.

Maps to ut_plan S8-28/30/40/45/46/57.
"""
import asyncio

import pytest


# ----------------------------------------------------------------------
# S8-57 — is_registered_reward: rule vs reward-model id (the channel-split predicate)
# ----------------------------------------------------------------------
@pytest.mark.parametrize('spec, expected', [
    ('format', True),  # a registered reward name
    ('accuracy', True),
    ('Qwen/Qwen2.5-0.5B-Instruct', False),  # a model id, not a registered rule
    ('some/localjudge', False),
])
def test_is_registered_reward_four_states(spec, expected):
    """S8-57 (string states): a registered name is a rule; any other string is a reward-model id."""
    from swift.dev.rewards.reward import is_registered_reward

    assert is_registered_reward(spec) is expected


def test_is_registered_reward_class_and_callable_are_rules():
    """S8-57 (object states): a plugin class or a bare callable is a rule (registered-name-first)."""
    from swift.dev.rewards.reward import is_registered_reward

    def my_reward(completions, **kw):
        return [1.0] * len(completions)

    assert is_registered_reward(my_reward) is True  # callable -> rule
    assert is_registered_reward(type(my_reward)) is True  # a class -> rule


# ----------------------------------------------------------------------
# S8-28 — get_reward_funcs: registered names -> instances, callables pass through, unknown fails fast
# ----------------------------------------------------------------------
def test_get_reward_funcs_resolves_names_and_callables():
    """S8-28: a registered name resolves to its ORM instance (args=config), a callable passes through;
    both are wrapped in the async shell, and the display names suit per-reward metric keys."""
    from swift.dev.rewards.reward import _AwaitableToSync, get_reward_funcs
    from twinkle.reward.format_reward import FormatReward

    def my_reward(completions, **kw):
        return [1.0] * len(completions)

    funcs, names = get_reward_funcs(['format', my_reward])
    assert names == ['FormatReward', 'my_reward']
    # Each resolved reward rides in the async shell (S8-40); the wrapped target is the instance/callable.
    assert isinstance(funcs[0], _AwaitableToSync) and isinstance(funcs[0]._func, FormatReward)
    assert isinstance(funcs[1], _AwaitableToSync) and funcs[1]._func is my_reward


def test_get_reward_funcs_unknown_name_fails_fast():
    """S8-28 guard: an unregistered reward name is a ValueError naming the available plugins."""
    from swift.dev.rewards.reward import get_reward_funcs

    with pytest.raises(ValueError, match='not registered'):
        get_reward_funcs(['no_such_reward_xyz'])


# ----------------------------------------------------------------------
# S8-40 — _AwaitableToSync: sync passthrough / async asyncio.run / label preserved
# ----------------------------------------------------------------------
def test_awaitable_to_sync_three_states():
    """S8-40: a sync reward's list passes through; an async reward's coroutine is run to completion;
    the wrapper keeps the reward's __name__ so its metric label survives."""
    from swift.dev.rewards.reward import _AwaitableToSync

    def sync_reward(completions, **kw):
        return [1.0, 2.0]

    async def async_reward(completions, **kw):
        await asyncio.sleep(0)
        return [float(len(c)) for c in completions]

    assert _AwaitableToSync(sync_reward)(['a', 'b']) == [1.0, 2.0]  # sync passthrough
    assert _AwaitableToSync(async_reward)(['a', 'bb']) == [1.0, 2.0]  # async resolved via asyncio.run
    assert _AwaitableToSync(sync_reward).__name__ == 'sync_reward'  # label preserved
    assert _AwaitableToSync(async_reward).__name__ == 'async_reward'


# ----------------------------------------------------------------------
# S8-30 — compute_advantages: group scale zero-centers; flat/matrix input; validation
# ----------------------------------------------------------------------
def test_advantage_api_delegates_to_twinkle():
    """S8-30: the dev advantage entry point delegates to twinkle.advantage; group scaling zero-centers
    each generation group; accepts a flat [N] list or an [N, n_funcs] matrix; validates its inputs."""
    torch = pytest.importorskip('torch')
    from swift.dev.rewards.advantage import compute_advantages

    # flat [N] input, one group of 4 -> grpo standardizes to mean 0.
    adv = compute_advantages([1.0, 2.0, 3.0, 4.0], num_generations=4, scale_rewards='group')
    assert isinstance(adv, list) and len(adv) == 4
    assert abs(sum(adv)) < 1e-5
    # grpo hand oracle: (r - mean) / std with the UNBIASED (sample) std twinkle uses -- mean=2.5,
    # var = sum((r-2.5)^2)/(n-1) = 5/3. Pinning the unbiased choice: a switch to population std
    # (sqrt(1.25)) would shift every value and redden this.
    std = (5.0 / 3.0)**0.5
    assert adv == pytest.approx([(r - 2.5) / std for r in (1.0, 2.0, 3.0, 4.0)])

    # matrix [N, 1] input behaves identically.
    r = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    assert abs(sum(compute_advantages(r, num_generations=4))) < 1e-5

    # validation
    with pytest.raises(ValueError, match='reward_weights length'):
        compute_advantages(r, num_generations=4, reward_weights=[1.0, 2.0])  # 2 weights, 1 func
    with pytest.raises(ValueError, match='not in'):
        compute_advantages(r, num_generations=4, advantage_estimator='nope')
    with pytest.raises(ValueError, match='not divisible'):
        compute_advantages(r, num_generations=3)
    with pytest.raises(ValueError, match='requires kl_values'):
        compute_advantages(r, num_generations=4, kl_in_reward=True, beta=0.1)


# ----------------------------------------------------------------------
# S8-42 — advantage estimator family: each estimator's hand-computed numerics (scale='none')
# ----------------------------------------------------------------------
@pytest.mark.parametrize('estimator, extra, expected', [
    # rloo: adv_i = r_i - mean(others) = r_i - (sum - r_i)/(K-1); sum=10, K=4.
    ('rloo', {}, [-2.0, -2.0 / 3.0, 2.0 / 3.0, 2.0]),
    # reinforce++: group-centered (scale='none' skips the advantage-std normalization).
    ('reinforce_plus_plus', {}, [-1.5, -0.5, 0.5, 1.5]),
    # opo: length-weighted baseline = sum(len*r)/sum(len) = (1+4+9+16)/10 = 3 -> r - 3.
    ('opo', {
        'response_lengths': [1, 2, 3, 4]
    }, [-2.0, -1.0, 0.0, 1.0]),
    # gpg: alpha = N/count_nonzero = 4/4 = 1, f_norm=1 -> 1*(r - group_mean 2.5)/1.
    ('gpg', {}, [-1.5, -0.5, 0.5, 1.5]),
    # remax: adv = reward - baseline (baselines all 0 here).
    ('remax', {
        'reward_baselines': [0.0, 0.0, 0.0, 0.0]
    }, [1.0, 2.0, 3.0, 4.0]),
])
def test_advantage_estimator_family_numerics(estimator, extra, expected):
    """S8-42: every estimator resolves through compute_advantages and matches its hand-computed formula."""
    pytest.importorskip('torch')
    from swift.dev.rewards.advantage import compute_advantages

    adv = compute_advantages([1.0, 2.0, 3.0, 4.0], num_generations=4, scale_rewards='none',
                             advantage_estimator=estimator, **extra)
    assert adv == pytest.approx(expected)


def test_advantage_estimator_required_inputs():
    """S8-42 guard: remax needs per-sample reward_baselines, opo needs response_lengths -- both fail loud."""
    pytest.importorskip('torch')
    from swift.dev.rewards.advantage import compute_advantages

    with pytest.raises(ValueError, match='reward_baselines'):
        compute_advantages([1.0, 2.0, 3.0, 4.0], num_generations=4, advantage_estimator='remax')
    with pytest.raises(ValueError, match='response_lengths'):
        compute_advantages([1.0, 2.0, 3.0, 4.0], num_generations=4, advantage_estimator='opo')


# ----------------------------------------------------------------------
# S8-45 — split_channel: the merged --orm/--prm list separates rules from model ids
# ----------------------------------------------------------------------
def test_split_channel_separates_rules_from_model_ids():
    """S8-45: a merged channel list splits into (rule_items, model_ids), order preserved within each,
    using is_registered_reward as the predicate (registered names + callables are rules)."""
    from swift.dev.rewards.scorer import split_channel

    def my_rule(completions, **kw):
        return [0.0] * len(completions)

    rules, model_ids = split_channel(['format', 'Qwen/Qwen2.5-0.5B-Instruct', 'accuracy', my_rule])
    assert rules == ['format', 'accuracy', my_rule]  # registered names + callable, typed order
    assert model_ids == ['Qwen/Qwen2.5-0.5B-Instruct']  # the lone model id


# ----------------------------------------------------------------------
# S8-46 — resolve_channel: at most one model id, and companion lists align with the model subsequence
# ----------------------------------------------------------------------
def test_resolve_channel_rejects_two_model_ids():
    """S8-46: a channel carries at most ONE reward model alongside any number of rules."""
    from swift.dev.rewards.scorer import ChannelCompanions, resolve_channel

    with pytest.raises(ValueError, match='at most one reward model'):
        resolve_channel(
            ['format', 'model-a', 'model-b'],
            ChannelCompanions(),
            channel='orm',
            sampler_model=None,
            backend='no')


def test_resolve_channel_rejects_misaligned_companion():
    """S8-46: a companion list (adapters/model_type/...) must carry exactly one entry per model id."""
    from swift.dev.rewards.scorer import ChannelCompanions, resolve_channel

    # One model id but two adapters -> misaligned; raises before any model download/inspection.
    with pytest.raises(ValueError, match='exactly one entry per model id'):
        resolve_channel(
            ['format', 'model-a'],
            ChannelCompanions(adapters=['adapter-1', 'adapter-2']),
            channel='orm',
            sampler_model=None,
            backend='no')


# ----------------------------------------------------------------------
# S8-38/39 — the privileged teacher view (RLSD/SDAR/OPSD): encode-and-align-check tail
# ----------------------------------------------------------------------
class _PrivilegedTemplate:
    """Stub template: train-mode encode masks the prompt turns and exposes the final assistant turn's
    token ids as the trainable response, so response_positions(feature) == len(response_ids)."""

    def __init__(self):
        self.mode = None
        self.rows = []

    def set_mode(self, mode):
        self.mode = mode

    def encode(self, row):
        self.rows.append(row)
        messages = row['messages']
        prompt_len = 0
        for message in messages[:-1]:
            content = message['content']
            prompt_len += len(content) if isinstance(content, list) else 1
        response = messages[-1]['content']
        response = list(response) if isinstance(response, list) else [response]
        return {'input_ids': [0] * prompt_len + response, 'labels': [-100] * prompt_len + response}


def test_encode_privileged_feature_single_turn_construction():
    """S8-39: the first user turn is replaced by the privileged prompt, the assistant turn carries the
    student's response ids verbatim, encoded in train mode with add_eos=False; the input is not mutated."""
    from swift.dev.utils.scoring import encode_privileged_feature, response_positions

    template = _PrivilegedTemplate()
    prompt = [{'role': 'user', 'content': 'original question'}]
    feature = encode_privileged_feature(template, prompt, 'PRIVILEGED HINT', [7, 8, 9])

    # encode_privileged_view set_mode('train')s a shallow COPY of the template, so the mode is not
    # observable on the original; the encoded row IS (copy.copy shares the .rows list).
    row = template.rows[0]
    assert row['add_eos'] is False  # the assistant turn is exactly response_ids (no extra eos)
    assert row['messages'][0] == {'role': 'user', 'content': 'PRIVILEGED HINT'}  # hint replaced the user turn
    assert row['messages'][-1] == {'role': 'assistant', 'content': [7, 8, 9]}  # response ids verbatim
    assert prompt[0]['content'] == 'original question'  # the caller's prompt is copied, not mutated
    assert response_positions(feature) == [1, 2, 3]  # 3 trainable positions == len(response_ids)


def test_encode_privileged_feature_requires_a_user_turn():
    """S8-39 guard: no user turn to carry the privileged context is a ValueError (a trailing assistant
    turn is dropped first, so an assistant-only prompt leaves nothing)."""
    from swift.dev.utils.scoring import encode_privileged_feature

    with pytest.raises(ValueError, match='requires a user turn'):
        encode_privileged_feature(_PrivilegedTemplate(), [{'role': 'assistant', 'content': 'x'}], 'hint', [1])


def test_encode_privileged_view_template_and_alignment_guards():
    """S8-38: no template is a ValueError; a template whose encode changes the trainable-token count is a
    hard RuntimeError (teacher and student must score exactly the same response tokens, never truncated)."""
    from swift.dev.utils.scoring import encode_privileged_view

    with pytest.raises(ValueError, match='requires a template'):
        encode_privileged_view(None, [{'role': 'user', 'content': 'q'}], [1, 2])

    class _Misaligned(_PrivilegedTemplate):

        def encode(self, row):
            return {'input_ids': [0, 0], 'labels': [-100, -100]}  # 0 trainable, but 2 response_ids

    messages = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': [1, 2]}]
    with pytest.raises(RuntimeError, match='changed the response-token count'):
        encode_privileged_view(_Misaligned(), messages, [1, 2])
