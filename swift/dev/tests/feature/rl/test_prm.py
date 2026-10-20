"""S8c — PRM (process-reward) scorer truth values (L1, hand-computed).

The PRMScorer template is segment -> score-each-step-by-its-prefix (one batch) -> broadcast onto tokens.
These pin the two scorers' segmentation and the per-token broadcast against hand-computed spans, plus the
MultiTurnContext turn alignment and the guards (a span must tile the response and never cross a turn).
``decode`` / ``score_steps`` are injected (the scorer owns no model), so this is pure Python.

Maps to ut_plan S8-54/55.
"""
import pytest

# token id -> text; token 3 carries the newline delimiter so the response cuts after it.
_TEXT = {1: 'p', 2: 'q', 3: 'r\n', 4: 's', 5: 't', 6: 'u'}


def _decode(ids):
    return ''.join(_TEXT[t] for t in ids)


# ----------------------------------------------------------------------
# S8-54 — DelimiterPRMScorer: one step per delimiter-separated segment, broadcast per token
# ----------------------------------------------------------------------
def test_delimiter_prm_segments_and_broadcasts():
    """S8-54: the delimiter scan cuts the response into spans; each step is scored by its cumulative
    prefix; broadcast maps a step's score onto every token in its span."""
    from swift.dev.rewards.prm import DelimiterPRMScorer

    scorer = DelimiterPRMScorer(args=None)  # delimiter defaults to '\n'
    assert scorer.delimiter == '\n'
    response = [1, 2, 3, 4, 5, 6]  # decoded 'pqr\nstu' -> the '\n' sits in token 3
    captured = []

    def score_steps(rows):
        captured.extend(rows)
        return [10.0, 20.0]  # one score per step (2 spans)

    per_token = scorer(
        prompt_messages=[{'role': 'user', 'content': 'Q'}],
        response_token_ids=response,
        decoded=_decode(response),
        score_steps=score_steps,
        decode=_decode)

    # spans [(0,3),(3,6)] -> broadcast: first 3 tokens get 10, last 3 get 20.
    assert per_token == [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
    # each step row is the prompt + an assistant turn holding the response PREFIX up to the span end.
    assert len(captured) == 2
    assert captured[0]['messages'][-1] == {'role': 'assistant', 'content': 'pqr\n'}  # decode([1,2,3])
    assert captured[1]['messages'][-1] == {'role': 'assistant', 'content': 'pqr\nstu'}  # decode([1..6])


def test_delimiter_prm_custom_delimiter_and_step_count_mismatch():
    """S8-54: the delimiter is a --prm_step_delimiter knob; score_steps returning the wrong count is fatal."""
    from types import SimpleNamespace

    from swift.dev.rewards.prm import DelimiterPRMScorer

    scorer = DelimiterPRMScorer(args=SimpleNamespace(prm_step_delimiter='|'))
    assert scorer.delimiter == '|'
    response = [1, 2, 3]
    with pytest.raises(ValueError, match='returned 1 scores for 2 steps'):
        # 'pqr\n' has no '|', so one span over the whole response... force a 2-span segmentation via a
        # scorer subclass to exercise the count guard deterministically.
        class _TwoSpan(DelimiterPRMScorer):

            def segment(self, response_token_ids, decoded, decode, turn_context=None):
                return [(0, 1), (1, 3)]

        _TwoSpan(args=None)(
            prompt_messages=[],
            response_token_ids=response,
            decoded=_decode(response),
            score_steps=lambda rows: [1.0],  # one score for two steps
            decode=_decode)


def test_turn_prm_uses_turn_spans_and_rejects_single_turn():
    """S8-54: TurnPRMScorer's steps ARE the assistant-turn spans; a single-turn response has no turn
    structure, so it fails loudly (delimiter is the single-turn scorer)."""
    from swift.dev.rewards.prm import MultiTurnContext, TurnPRMScorer

    scorer = TurnPRMScorer()
    with pytest.raises(ValueError, match='requires a multi-turn rollout'):
        scorer.segment([1, 2, 3], 'pqr', _decode, turn_context=None)

    ctx = MultiTurnContext(
        messages=[{
            'role': 'user',
            'content': 'q'
        }, {
            'role': 'assistant',
            'content': 'a0'
        }, {
            'role': 'user',
            'content': 'obs'
        }, {
            'role': 'assistant',
            'content': 'a1'
        }],
        turn_spans=[(0, 3), (3, 6)],
        assistant_message_indices=[1, 3],
    )
    assert scorer.segment([1, 2, 3, 4, 5, 6], 'pqr\nstu', _decode, turn_context=ctx) == [(0, 3), (3, 6)]


# ----------------------------------------------------------------------
# S8-55 — MultiTurnContext alignment + the tiling / turn-crossing guards
# ----------------------------------------------------------------------
def test_multi_turn_context_turn_index_and_history():
    """S8-55: turn_index_for maps a response-token position to its turn (-1 outside); history_before is a
    deep copy of the real conversation preceding that turn's assistant message (prior turns + observations)."""
    from swift.dev.rewards.prm import MultiTurnContext

    messages = [
        {'role': 'user', 'content': 'q0'},
        {'role': 'assistant', 'content': 'a0'},  # index 1 -> turn 0
        {'role': 'user', 'content': 'obs0'},  # index 2: the tool observation between turns
        {'role': 'assistant', 'content': 'a1'},  # index 3 -> turn 1
    ]
    ctx = MultiTurnContext(messages=messages, turn_spans=[(0, 3), (3, 5)], assistant_message_indices=[1, 3])

    assert ctx.turn_index_for(0) == 0 and ctx.turn_index_for(2) == 0
    assert ctx.turn_index_for(3) == 1 and ctx.turn_index_for(4) == 1
    assert ctx.turn_index_for(5) == -1  # past the last span
    # turn 1's history is everything before its assistant message (index 3): q0, a0, obs0.
    hist = ctx.history_before(1)
    assert hist == messages[:3]
    assert hist is not messages and hist[0] is not messages[0]  # a deep copy
    assert ctx.history_before(0) == [messages[0]]  # turn 0 sees only the first user turn


def test_checked_spans_rejects_gap_and_turn_crossing():
    """S8-55: spans must tile [0, length) exactly (a gap/overlap is loud), and under a turn_context no
    span may cross an assistant-turn boundary."""
    from swift.dev.rewards.prm import MultiTurnContext, PRMScorer

    # a gap (cursor 0, span starts at 1) does not tile
    with pytest.raises(ValueError, match='do not tile'):
        PRMScorer._checked_spans([(1, 3)], length=3)
    # spans covering less than the whole response
    with pytest.raises(ValueError, match='cover only 2 of 3'):
        PRMScorer._checked_spans([(0, 2)], length=3)

    ctx = MultiTurnContext(
        messages=[{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}],
        turn_spans=[(0, 2), (2, 4)],
        assistant_message_indices=[1, 1],
    )
    # a span [1, 3) starts in turn 0 and ends in turn 1 -> crosses the boundary
    with pytest.raises(ValueError, match='crosses an assistant-turn boundary'):
        PRMScorer._checked_spans([(0, 1), (1, 3), (3, 4)], length=4, turn_context=ctx)
    # spans aligned to the turn boundaries pass
    assert PRMScorer._checked_spans([(0, 2), (2, 4)], length=4, turn_context=ctx) == [(0, 2), (2, 4)]
