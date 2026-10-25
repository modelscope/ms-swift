"""S8b — dev reward protocol seams: resolution, delegation, privileged views, channels (L1).

dev2's reward layer keeps only its own seams here: ``get_reward_funcs`` / ``is_registered_reward``
(the rule-vs-model-id dispatch used by every channel), ``compute_advantages`` (the thin delegate to
``twinkle.advantage`` -- the estimator math lives in twinkle's suite, what is pinned is the dev
entry contract: unbiased group std and the validation surface), the privileged teacher views in
``swift.dev.utils.scoring`` (RLSD/SDAR/OPSD share the encode-and-align-check tail), and the channel
split/resolve in ``swift.dev.rewards.scorer``. Rollout-internal cases (ut_plan S8-27/32-36) are
twinkle_agentic-layer and covered by its own suite.

Maps to ut_plan S8-26/28/30/38/39/40/42/45/46/57.
"""
import asyncio

import pytest


# ----------------------------------------------------------------------
# S8-30 (+26 merged) — compute_advantages: the dev entry delegating to twinkle.advantage
# ----------------------------------------------------------------------
class TestComputeAdvantages:
    """S8-26+S8-30: the group advantage reuses the twinkle estimator (no dev-side fork); the group
    scale zero-centers and divides by the UNBIASED (sample) std, and the entry validates its knobs."""

    def test_group_scale_uses_unbiased_std(self):
        """rewards [1,2,3,4], one group of 4: mean 2.5, sample std sqrt(5/3) -- hand-computed oracle
        (a population-std fork would give sqrt(1.25) and shift every value)."""
        from swift.dev.rewards.advantage import compute_advantages

        std = (5.0 / 3.0) ** 0.5
        oracle = [(r - 2.5) / std for r in (1.0, 2.0, 3.0, 4.0)]
        assert list(compute_advantages([1.0, 2.0, 3.0, 4.0], num_generations=4,
                                       scale_rewards='group')) == pytest.approx(oracle)

    def test_groups_are_independent(self):
        """Two groups of 2 zero-center within their OWN group, not across the batch (scale='none'
        isolates the centering; group [10, 30] centers to +-10, a batch-wide center would not)."""
        from swift.dev.rewards.advantage import compute_advantages

        out = list(compute_advantages([0.0, 2.0, 10.0, 30.0], num_generations=2, scale_rewards='none'))
        assert out == pytest.approx([-1.0, 1.0, -10.0, 10.0])

    @pytest.mark.parametrize('kwargs', [
        dict(reward_weights=[1.0, 2.0]),  # 2 weights for 1 function
        dict(advantage_estimator='nope'),  # unknown estimator
        dict(num_generations=3),  # 4 rewards do not split into groups of 3
        dict(kl_in_reward=True, beta=0.1),  # KL-in-reward without kl_values
    ])
    def test_validation_rejects_bad_knobs(self, kwargs):
        from swift.dev.rewards.advantage import compute_advantages

        with pytest.raises((ValueError, TypeError)):
            compute_advantages([1.0, 2.0, 3.0, 4.0], num_generations=4, **kwargs)


# ----------------------------------------------------------------------
# S8-42 — the estimator family, each resolved through compute_advantages, hand-computed
# ----------------------------------------------------------------------
@pytest.mark.parametrize('estimator, kwargs, oracle', [
    # rloo: r_i - mean(other three) = [1-3, 2-7/3, 3-5/3, 4-1]
    ('rloo', {}, [-2.0, -2.0 / 3.0, 2.0 / 3.0, 2.0]),
    # reinforce++: plain group centering, no length term
    ('reinforce_plus_plus', {}, [-1.5, -0.5, 0.5, 1.5]),
    # opo: length-weighted baseline = sum(r*l)/sum(l) = 30/10 = 3 -> r - 3
    ('opo', dict(response_lengths=[1, 2, 3, 4]), [-2.0, -1.0, 0.0, 1.0]),
    # gpg: group-centered (f_norm=1 default)
    ('gpg', {}, [-1.5, -0.5, 0.5, 1.5]),
    # remax: reward - baseline, baselines all 0 -> reward itself
    ('remax', dict(reward_baselines=[0.0, 0.0, 0.0, 0.0]), [1.0, 2.0, 3.0, 4.0]),
])
def test_advantage_estimator_family(estimator, kwargs, oracle):
    """S8-42: every estimator resolves through the same entry and matches its hand-computed formula
    (scale='none' isolates the estimator from the std normalization)."""
    from swift.dev.rewards.advantage import compute_advantages

    out = compute_advantages([1.0, 2.0, 3.0, 4.0], num_generations=4, scale_rewards='none',
                             advantage_estimator=estimator, **kwargs)
    assert list(out) == pytest.approx(oracle)


# ----------------------------------------------------------------------
# S8-28 — get_reward_funcs: resolve registered names and raw callables
# ----------------------------------------------------------------------
class TestGetRewardFuncs:

    def test_registered_name_resolves_to_wrapped_callable(self):
        """'accuracy' resolves through the registry to a MathAccuracy instance wrapped for sync
        calling, and the metric label keeps the registry display name."""
        from swift.dev.rewards.reward import _AwaitableToSync, get_reward_funcs

        funcs, names = get_reward_funcs(['accuracy'])
        assert names == ['MathAccuracy']
        assert isinstance(funcs[0], _AwaitableToSync)
        assert callable(funcs[0])

    def test_plain_callable_is_accepted_and_wrapped(self):
        """A raw callable is accepted without registration, wrapped for sync calling (it may be
        async), and keeps its own metric label."""
        from swift.dev.rewards.reward import _AwaitableToSync, get_reward_funcs

        def my_reward(completions, **kwargs):
            return [0.0 for _ in completions]

        funcs, names = get_reward_funcs([my_reward])
        assert isinstance(funcs[0], _AwaitableToSync)
        assert funcs[0](['a', 'b']) == [0.0, 0.0]  # delegates to the original callable
        assert names == ['my_reward']


# ----------------------------------------------------------------------
# S8-57 — is_registered_reward: the four input states every channel dispatches on
# ----------------------------------------------------------------------
@pytest.mark.parametrize('spec, expected', [
    ('accuracy', True),  # a registered rule name
    ('not_a_registered_reward', False),  # an unknown string is a MODEL id, never a rule
    (None, False),
])
def test_is_registered_reward_string_states(spec, expected):
    from swift.dev.rewards.reward import is_registered_reward

    assert is_registered_reward(spec) is expected


def test_is_registered_reward_class_and_callable_states():
    from swift.dev.rewards.orm import MathAccuracy
    from swift.dev.rewards.reward import ORM, is_registered_reward

    assert is_registered_reward(ORM) is True  # a Reward subclass is a rule
    assert is_registered_reward(MathAccuracy) is True
    assert is_registered_reward(lambda completions, **kw: [0.0]) is True  # any callable is a rule
    assert is_registered_reward(object) is True  # every class is callable -> a rule
    assert is_registered_reward(42) is False  # a plain value is neither


# ----------------------------------------------------------------------
# S8-40 — _AwaitableToSync: the async-reward seam
# ----------------------------------------------------------------------
class TestAwaitableToSync:

    def test_sync_passthrough(self):
        """A sync reward's list passes through untouched (no event loop is spun up)."""
        from swift.dev.rewards.reward import _AwaitableToSync

        def sync_reward(completions, **kwargs):
            return [float(len(c)) for c in completions]

        wrapped = _AwaitableToSync(sync_reward)
        assert wrapped(['ab', 'c']) == [2.0, 1.0]

    def test_async_resolved_via_asyncio_run(self):
        """An ``async def`` reward is resolved to completion: the coroutine's awaited value, not the
        coroutine object, comes back."""
        from swift.dev.rewards.reward import _AwaitableToSync

        async def async_reward(completions, **kwargs):
            await asyncio.sleep(0)
            return [float(len(c)) for c in completions]

        wrapped = _AwaitableToSync(async_reward)
        assert wrapped(['ab', 'c']) == [2.0, 1.0]

    def test_display_name_preserved(self):
        """The wrapper's __name__ keeps the reward's metric label (display_name reads it first)."""
        from swift.dev.rewards.reward import _AwaitableToSync

        def my_custom_reward(completions, **kwargs):
            return [0.0]

        wrapped = _AwaitableToSync(my_custom_reward)
        assert wrapped.__name__ == 'my_custom_reward'


# ----------------------------------------------------------------------
# S8-38/39 — the privileged teacher views (RLSD / SDAR / OPSD shared tail)
# ----------------------------------------------------------------------
class _StubPrivilegedTemplate:
    """Stands in for the dev Template: records the encoded row and the mode, exposes exactly
    ``len(response_ids)`` trainable label positions so the alignment check can pass or fail on cue."""

    def __init__(self, trainable_positions=3):
        self.modes = []
        self.rows = []
        self._trainable = trainable_positions

    def set_mode(self, mode):
        self.modes.append(mode)

    def encode(self, row):
        self.rows.append(row)
        labels = [-100] * 2 + list(range(100, 100 + self._trainable))
        return {'input_ids': [1, 2, 3, 4, 5], 'labels': labels}


class TestEncodePrivilegedView:

    def test_encodes_in_train_mode_and_checks_alignment(self):
        """S8-38: the shared response ids encode under a train-mode COPY of the template; the encoded
        feature must expose exactly one trainable position per response id (teacher and student score
        identical tokens)."""
        from swift.dev.utils.scoring import encode_privileged_view

        template = _StubPrivilegedTemplate(trainable_positions=3)
        feature = encode_privileged_view(template, [{'role': 'user', 'content': 'q'}], [11, 12, 13])
        assert feature['labels'] == [-100, -100, 100, 101, 102]
        assert template.modes == ['train']  # the copy is switched to train; row seen once
        assert template.rows[0]['add_eos'] is False  # the assistant turn is exactly response_ids

    def test_response_count_mismatch_is_a_hard_error(self):
        """S8-38 guard: a tokenizer/template difference changing the response-token count silently
        mis-scores teacher vs student -- raise instead of truncating."""
        from swift.dev.utils.scoring import encode_privileged_view

        template = _StubPrivilegedTemplate(trainable_positions=2)  # 2 trainable vs 3 response ids
        with pytest.raises(RuntimeError, match='response-token count'):
            encode_privileged_view(template, [{'role': 'user', 'content': 'q'}], [11, 12, 13])

    def test_missing_template_rejected(self):
        from swift.dev.utils.scoring import encode_privileged_view

        with pytest.raises(ValueError, match='template'):
            encode_privileged_view(None, [{'role': 'user', 'content': 'q'}], [11])


class TestEncodePrivilegedFeature:

    def test_single_turn_privileged_construction(self):
        """S8-39: the FIRST user turn's content becomes the teacher prompt, the student's response
        ids ride the appended assistant turn verbatim, and the shared align-check tail runs."""
        from swift.dev.utils.scoring import encode_privileged_feature

        template = _StubPrivilegedTemplate(trainable_positions=2)
        feature = encode_privileged_feature(
            template,
            [{'role': 'user', 'content': 'original question'}],
            teacher_prompt='privileged teacher hint',
            response_ids=[21, 22])
        assert feature['labels'] == [-100, -100, 100, 101]
        row = template.rows[0]
        assert row['messages'] == [
            {'role': 'user', 'content': 'privileged teacher hint'},
            {'role': 'assistant', 'content': [21, 22]},  # ids verbatim, not text
        ]

    def test_requires_a_user_turn(self):
        from swift.dev.utils.scoring import encode_privileged_feature

        with pytest.raises(ValueError, match='user turn'):
            encode_privileged_feature(_StubPrivilegedTemplate(), [{'role': 'system', 'content': 's'}],
                                      'teacher', [1])


# ----------------------------------------------------------------------
# S8-45/46 — channel split and companion alignment
# ----------------------------------------------------------------------
class TestSplitChannel:

    def test_mixed_channel_splits_rules_and_model_ids(self):
        """S8-45: is_registered_reward decides the split -- registered names stay rules, unknown
        strings are model ids; order is preserved within each half."""
        from swift.dev.rewards.scorer import split_channel

        rules, model_ids = split_channel(['accuracy', 'format', 'openai/gpt-oss-20b'])
        assert rules == ['accuracy', 'format']
        assert model_ids == ['openai/gpt-oss-20b']

    def test_all_rules_channel_has_no_model_ids(self):
        from swift.dev.rewards.scorer import split_channel

        assert split_channel(['accuracy', 'format']) == (['accuracy', 'format'], [])


class TestResolveChannelCompanions:

    def test_companion_must_align_with_model_ids(self):
        """S8-46: companion lists carry exactly one entry per model id in the channel; a mismatched
        length is rejected (a silently misaligned adapter/model_type would target the wrong model)."""
        from swift.dev.rewards.scorer import ChannelCompanions, resolve_channel

        companions = ChannelCompanions(model_types=['a', 'b'])  # 2 entries, 0 model ids
        with pytest.raises(ValueError, match='model_type'):
            resolve_channel(['accuracy'], companions, channel='orm', sampler_model=None,
                            backend='transformers')

    def test_more_than_one_model_id_rejected(self):
        from swift.dev.rewards.scorer import ChannelCompanions, resolve_channel

        with pytest.raises(ValueError, match='at most one reward model'):
            resolve_channel(['accuracy', 'model/a', 'model/b'], ChannelCompanions(), channel='orm',
                            sampler_model=None, backend='transformers')

    def test_rules_only_plan_preserves_typed_order(self):
        """The plan keeps the user's typed order (that is what *_weights align against)."""
        from swift.dev.rewards.scorer import ChannelCompanions, resolve_channel

        plan = resolve_channel(['format', 'accuracy'], ChannelCompanions(), channel='orm',
                               sampler_model=None, backend='transformers')
        assert plan.entries == ['format', 'accuracy']  # typed names preserved, in order
        assert plan.rules == plan.entries and plan.specs == []  # all rules, no model specs
