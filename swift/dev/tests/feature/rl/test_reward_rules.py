"""S8c — reward-rule ground truth: format/cosine/repetition/overlong/accuracy/code (L1).

Every rule is pinned with a hand-computed or hand-labelled oracle on minimal trajectory rows
(``messages`` with the trailing assistant turn as the candidate, ``response_token_ids`` /
``solution`` as the length/gold columns the rules actually read). The length-aware rules read their
knobs off a real ``RewardConfig`` (``cosine_max_len`` etc. default to None -- the recipe derives
them at runtime -- so the tests state them explicitly).

Maps to ut_plan S8-47..53.
"""
import json
import math

import pytest


def _row(completion, **extra):
    return {
        'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': completion}],
        **extra,
    }


THINK_OPEN, THINK_CLOSE = chr(60) + 'think' + chr(62), chr(60) + '/think' + chr(62)
#: The ReAct format the rule demands: a reasoning block, then ``Action:``, then ``Action Input:``.
REACT_GOOD = THINK_OPEN + 'reason here' + THINK_CLOSE + '\nAction: add\nAction Input: {"a": 1, "b": 2}'


class TestReActFormat:
    """S8-47: the react-format rule rewards exactly the think-block + Action + Action Input shape."""

    def test_ground_truth_rows(self):
        from swift.dev.rewards.orm import ReActFormat

        scores = ReActFormat()([
            _row(REACT_GOOD),
            _row('no tags, plain answer'),  # missing the whole structure
            _row(THINK_OPEN + 'only think' + THINK_CLOSE),  # think block but no Action
            _row('Action: add\nAction Input: {}'),  # Action without the think block
        ])
        assert scores == [1.0, 0.0, 0.0, 0.0]


class TestCosineReward:
    """S8-48: length-interpolated reward -- wrong answers rise from -0.5 (t=0) to 0 (t=T), correct
    answers from 0 to +0.5; the interpolation is the cosine curve, not linear."""

    def _args(self):
        from swift.dev.config import RewardConfig

        return RewardConfig(cosine_max_len=10)  # value knobs keep their defaults

    def test_wrong_answer_curve(self):
        from swift.dev.rewards.orm import CosineReward

        rule = CosineReward(self._args())
        rows = [_row('x', response_token_ids=[1] * t) for t in (1, 5, 10)]  # no gold -> wrong
        oracle = [-0.5 + 0.5 * (1 - math.cos(t * math.pi / 10)) / 2 for t in (1, 5, 10)]
        assert rule(rows) == pytest.approx(oracle)  # [-0.4878, -0.25, 0.0]
        assert oracle[0] < -0.455  # the curve at t=1 sits BELOW linear (-0.45): cosine, not linear

    def test_correct_answer_curve_decays_from_one(self):
        """A correct answer starts at min_len_value_correct=1.0 (t=0) and decays to
        max_len_value_correct=0.5 at t=T -- short correct answers are worth more."""
        from swift.dev.rewards.orm import CosineReward

        rule = CosineReward(self._args())
        rows = [_row('the answer is \\boxed{4}', solution='4', response_token_ids=[1] * t)
                for t in (1, 10)]
        oracle = [1.0 - 0.5 * (1 - math.cos(t * math.pi / 10)) / 2 for t in (1, 10)]
        assert rule(rows) == pytest.approx(oracle)  # [~0.9878, 0.5]


class TestRepetitionPenalty:
    """S8-49: penalty = (1 - unique/total ngrams) * max_penalty, on lowercased words."""

    def test_ground_truth(self):
        from swift.dev.config import RewardConfig
        from swift.dev.rewards.orm import RepetitionPenalty

        rule = RepetitionPenalty(RewardConfig())  # n=3, max_penalty=-1.0
        scores = rule([_row('a a a a'), _row('a b c d a b c d'), _row('a b c d e f g h i')])
        # 'a a a a': ngrams {aaa}, total 2 -> (1-1/2)*-1 = -0.5
        # 'a b c d a b c d': 6 ngrams, 4 unique -> (1-4/6)*-1 = -1/3
        # all-unique: 0.0
        assert scores == pytest.approx([-0.5, -(1.0 / 3.0), 0.0])

    def test_short_completion_is_exempt(self):
        """Fewer words than the ngram size cannot repeat -- 0.0, not a division error."""
        from swift.dev.config import RewardConfig
        from swift.dev.rewards.orm import RepetitionPenalty

        assert RepetitionPenalty(RewardConfig())([_row('a b')]) == [0.0]


class TestSoftOverlong:
    """S8-50: -(len - (max - cache)) / cache, floored at 0 -- a soft ramp instead of a hard cut."""

    def test_ground_truth(self):
        from swift.dev.config import RewardConfig
        from swift.dev.rewards.orm import SoftOverlong

        rule = SoftOverlong(RewardConfig(soft_max_length=20, soft_cache_length=5))
        scores = rule([_row('x', response_token_ids=[1] * t) for t in (30, 25, 15, 5)])
        # over-budget: -(30-15)/5 = -3, -(25-15)/5 = -2; within budget: 0, 0
        assert scores == pytest.approx([-3.0, -2.0, 0.0, 0.0])


class TestDirectTwinkleRules:
    """S8-51: 'format' and 'math' resolve DIRECTLY to twinkle's FormatReward/MathReward classes in the
    registry -- a no-arg construct must work and the native input contract must hold."""

    def test_format_rule_is_the_twinkle_class(self):
        from swift.dev.rewards.reward import get_reward_funcs

        funcs, names = get_reward_funcs(['format'])
        assert names == ['FormatReward']
        assert type(funcs[0]._func).__module__.startswith('twinkle'), 'must be the twinkle class'

    def test_math_rule_reads_the_solution_column(self):
        from swift.dev.rewards.reward import get_reward_funcs

        funcs, names = get_reward_funcs(['math'])
        assert names == ['MathReward']
        assert type(funcs[0]._func).__module__.startswith('twinkle')
        scores = funcs[0]([_row('the answer is \\boxed{4}', solution='4'),
                           _row('the answer is \\boxed{5}', solution='4')])
        assert scores == pytest.approx([1.0, 0.0])


class TestMathAccuracy:
    """S8-52: the 'accuracy' ORM reads the dataset's ``solution`` gold column."""

    def test_ground_truth(self):
        from swift.dev.rewards.orm import MathAccuracy

        rule = MathAccuracy()
        assert rule.ground_truth_key == 'solution'
        scores = rule([_row('The answer is 4', solution='4'), _row('The answer is 5', solution='4')])
        assert scores == pytest.approx([1.0, 0.0])


class TestCodeExecutionReward:
    """S8-53: the reward EXECUTES the generated program in a throwaway LocalEnv (workspace=None -> a
    temp dir per run, no docker) and scores the fraction of test cases it passes."""

    def _cases(self):
        return json.dumps({'inputs': ['hello', 'world'], 'outputs': ['hello', 'world']})

    def test_real_execution_pass_fraction(self):
        import asyncio

        from swift.dev.rewards.code_reward import CodeExecutionReward

        reward = CodeExecutionReward()  # args=None -> python + LocalEnv
        correct = _row('```python\nimport sys\nprint(sys.stdin.read().strip())\n```', test_cases=self._cases())
        half = _row('```python\nprint("hello")\n```', test_cases=self._cases())  # passes 1 of 2
        scores = asyncio.run(reward([correct, half]))
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.5)

    def test_missing_test_cases_rejected(self):
        """Guard: no test_cases column is a loud error -- the reward is meaningless without cases."""
        import asyncio

        from swift.dev.rewards.code_reward import CodeExecutionReward

        with pytest.raises(ValueError, match='test_cases'):
            asyncio.run(CodeExecutionReward()([_row('```python\nprint(1)\n```')]))
