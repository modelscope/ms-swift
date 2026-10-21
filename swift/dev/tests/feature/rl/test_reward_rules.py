"""S8c — reward-rule truth values (L1, hand-annotated rows).

Each ORM rule is scored against a hand-computed oracle on a labelled trajectory row, so a wrong
formula (not a wrong wiring) reddens. The rules read their hyperparameters off ``args`` (the run
Config) via the ``cls(args=config)`` plugin contract; a SimpleNamespace stands in for the Config here.
``format``/``math`` are twinkle's own classes used directly (no swift re-wrapper).

Maps to ut_plan S8-47..52.
"""
import math
from types import SimpleNamespace

import pytest


def _row(completion, response_token_ids=None, **columns):
    """A trajectory row: the completion is the last assistant turn; extras ride flat (the ORM protocol)."""
    row = {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': completion}]}
    if response_token_ids is not None:
        row['response_token_ids'] = response_token_ids
    row.update(columns)
    return row


# ----------------------------------------------------------------------
# S8-47 — ReActFormat: the <think>..</think> Action: .. Action Input: .. shape
# ----------------------------------------------------------------------
def test_react_format_truth_rows():
    """S8-47: a well-formed ReAct completion scores 1.0, anything else 0.0 (regex over the completion)."""
    from swift.dev.rewards.orm import ReActFormat

    rule = ReActFormat()
    good = _row('<think>reasoning</think>\nAction: search\nAction Input: {"q": 1}')
    missing_action_input = _row('<think>r</think>\nAction: search')
    plain = _row('just an answer')
    assert rule([good, missing_action_input, plain]) == [1.0, 0.0, 0.0]


# ----------------------------------------------------------------------
# S8-49 — RepetitionPenalty: (1 - unique_ngrams/total_ngrams) * max_penalty
# ----------------------------------------------------------------------
def test_repetition_penalty_truth_rows():
    """S8-49: the repeated-ngram fraction scales max_penalty; a too-short completion is unpenalized (0)."""
    from swift.dev.rewards.orm import RepetitionPenalty

    args = SimpleNamespace(repetition_n_grams=2, repetition_max_penalty=-1.0)
    rule = RepetitionPenalty(args)
    # "x x x x": 2-grams (x,x)x3 -> total 3, unique 1 -> (1 - 1/3) * -1 = -2/3
    all_same = _row('x x x x')
    # "a b c d": 2-grams (a,b)(b,c)(c,d) -> total 3, unique 3 -> (1 - 1) * -1 = 0
    all_unique = _row('a b c d')
    # fewer words than ngram_size -> 0 (no penalty)
    too_short = _row('hello')
    rewards = rule([all_same, all_unique, too_short])
    assert rewards[0] == pytest.approx(-2.0 / 3.0)
    assert rewards[1] == pytest.approx(0.0)
    assert rewards[2] == 0.0


# ----------------------------------------------------------------------
# S8-50 — SoftOverlong: min(-(len - (max-cache))/cache, 0)
# ----------------------------------------------------------------------
def test_soft_overlong_truth_rows():
    """S8-50: no penalty under the soft window, a linearly growing negative penalty past it."""
    from swift.dev.rewards.orm import SoftOverlong

    args = SimpleNamespace(soft_max_length=10, soft_cache_length=5)  # expected_len = 10 - 5 = 5
    rule = SoftOverlong(args)
    under = _row('x', response_token_ids=[0] * 3)  # exceed = -2 -> min(0.4, 0) = 0
    at_edge = _row('x', response_token_ids=[0] * 5)  # exceed = 0 -> 0
    over = _row('x', response_token_ids=[0] * 8)  # exceed = 3 -> -3/5 = -0.6
    far_over = _row('x', response_token_ids=[0] * 15)  # exceed = 10 -> -10/5 = -2.0
    assert rule([under, at_edge, over, far_over]) == pytest.approx([0.0, 0.0, -0.6, -2.0])


# ----------------------------------------------------------------------
# S8-48 — CosineReward: length-conditioned cosine, sign flipped by correctness
# ----------------------------------------------------------------------
def test_cosine_reward_truth_rows():
    """S8-48: cosfn(len, max_len, min, max) with the min/max swapped for correct vs wrong answers.

    The accuracy oracle is stubbed so this isolates the length curve (the real accuracy rule is S8-52).
    """
    from swift.dev.rewards.orm import CosineReward

    args = SimpleNamespace(
        cosine_min_len_value_wrong=-1.0,
        cosine_max_len_value_wrong=0.0,
        cosine_min_len_value_correct=0.0,
        cosine_max_len_value_correct=1.0,
        cosine_max_len=10,
    )

    def cosfn(t, T, min_value, max_value):
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    # A WRONG answer (accuracy 0): min=max_len_value_wrong=0, max=min_len_value_wrong=-1.
    wrong_rule = CosineReward(args, accuracy_orm=lambda trajs: [0.0] * len(trajs))
    wrong_lens = [0, 5, 10]
    wrong_rows = [_row('x', response_token_ids=[0] * n) for n in wrong_lens]
    expected_wrong = [cosfn(n, 10, args.cosine_max_len_value_wrong, args.cosine_min_len_value_wrong) for n in wrong_lens]
    assert wrong_rule(wrong_rows) == pytest.approx(expected_wrong)
    assert wrong_rule(wrong_rows) == pytest.approx([-1.0, -0.5, 0.0])

    # A CORRECT answer (accuracy 1): min=max_len_value_correct=1, max=min_len_value_correct=0.
    correct_rule = CosineReward(args, accuracy_orm=lambda trajs: [1.0] * len(trajs))
    correct_rows = [_row('x', response_token_ids=[0] * n) for n in wrong_lens]
    assert correct_rule(correct_rows) == pytest.approx([0.0, 0.5, 1.0])


# ----------------------------------------------------------------------
# S8-51 — format/math rules are twinkle's classes used directly (no swift re-wrapper)
# ----------------------------------------------------------------------
def test_format_and_math_rules_use_twinkle_classes_directly():
    """S8-51: identical scorers are twinkle's own classes; resolve() no-arg constructs them (their native
    row protocol is already the ORM input, so no swift subclass is interposed)."""
    from swift.dev.rewards.orm import REWARD, orms
    from swift.dev.utils.plugin import PluginRegistry
    from twinkle.reward import FormatReward, MathReward

    assert orms['format'] is FormatReward
    assert orms['math'] is MathReward
    assert isinstance(PluginRegistry.resolve(REWARD, 'format'), FormatReward)
    assert isinstance(PluginRegistry.resolve(REWARD, 'math'), MathReward)


# ----------------------------------------------------------------------
# S8-52 — MathAccuracy: swift's gold column default is 'solution'
# ----------------------------------------------------------------------
def test_math_accuracy_uses_solution_gold_key():
    """S8-52: MathAccuracy is twinkle's math-verify scorer with swift's 'solution' gold key as default."""
    pytest.importorskip('math_verify')
    from swift.dev.rewards.orm import MathAccuracy

    rule = MathAccuracy()
    assert rule.ground_truth_key == 'solution'
    # A correct row scores 1.0 and a wrong row 0.0 against the 'solution' gold column.
    correct = _row('The answer is 4', solution='4')
    wrong = _row('The answer is 5', solution='4')
    scores = rule([correct, wrong])
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# S8-53 — CodeExecutionReward: REAL execution in a throwaway LocalEnv, pass-fraction score
# ----------------------------------------------------------------------
def _code_traj(code, cases):
    return {
        'messages': [{'role': 'user', 'content': 'echo stdin'}, {'role': 'assistant', 'content': code}],
        'test_cases': cases,
    }


def test_code_execution_reward_runs_real_code_in_local_env():
    """S8-53: the reward EXECUTES the generated program in a throwaway LocalEnv (workspace=None -> a temp
    dir per run, no docker) and scores the fraction of test cases it passes (stdin -> stdout)."""
    import asyncio
    import json

    from swift.dev.rewards.code_reward import CodeExecutionReward

    cases = json.dumps({'inputs': ['hello', 'world'], 'outputs': ['hello', 'world']})
    reward = CodeExecutionReward()  # args=None -> python + LocalEnv
    correct = _code_traj('```python\nimport sys\nprint(sys.stdin.read().strip())\n```', cases)
    half = _code_traj('```python\nprint("hello")\n```', cases)  # passes only the 'hello' case
    scores = asyncio.run(reward([correct, half]))
    assert scores[0] == pytest.approx(1.0)  # both cases pass
    assert scores[1] == pytest.approx(0.5)  # one of two -> the pass fraction


def test_code_execution_reward_requires_test_cases():
    """S8-53 guard: no test_cases column is a loud error -- the reward is meaningless without cases."""
    import asyncio

    from swift.dev.rewards.code_reward import CodeExecutionReward

    traj = {
        'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': '```python\nprint(1)\n```'}]
    }
    with pytest.raises(ValueError, match='test_cases'):
        asyncio.run(CodeExecutionReward()([traj]))
