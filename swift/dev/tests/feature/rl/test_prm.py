"""S8c — PRM process-reward scorers: delimiter/turn segmentation + broadcast (L1).

The PRMScorer is a template method owning NO model: ``score_steps`` (the frozen PRM forward) and
``decode`` (the tokenizer) are injected into ``__call__``, which segments the response into step
spans, builds one prefix row per step (prompt + response-so-far, or per-turn history), scores the
rows in one batch, and broadcasts each step's scalar onto its tokens. Fakes stand in for the two
injections; every span/row/value below is hand-computed.

Maps to ut_plan S8-54 (delimiter/turn dual scorer ground truth) and S8-55 (MultiTurnContext).
"""
import pytest

from swift.dev.rewards.prm import (
    DelimiterPRMScorer,
    MultiTurnContext,
    TurnPRMScorer,
    prm_scorers,
)


def _dec_newline(ids):
    """Token id parity -> 'x\\n' / 'y': tokens [0,1,2,3,4] decode to 'x\\nyx\\nyx\\n'."""
    return ''.join('x\n' if i % 2 == 0 else 'y' for i in ids)


def _dec_alpha(ids):
    return ''.join('abcxyz'[i % 6] for i in ids)


class TestDelimiterPRMScorer:

    def test_step_spans_and_broadcast(self):
        """S8-54: one step per line. Tokens [0..4] with newlines at 0/2/4 -> spans (0,1),(1,3),(3,5);
        step scores [1,2,3] broadcast onto each span's tokens -> [1, 2, 2, 3, 3]."""
        ids = [0, 1, 2, 3, 4]
        seen = []

        def score_steps(rows):
            seen.extend(rows)
            return [float(i + 1) for i in range(len(rows))]

        out = DelimiterPRMScorer()([], ids, _dec_newline(ids), score_steps, _dec_newline)
        assert out == [1.0, 2.0, 2.0, 3.0, 3.0]

    def test_step_rows_are_response_so_far(self):
        """S8-54: each step row is the PROMPT plus the response up to that step's end (the PRM judges
        the prefix, never the future) -- here the assistant contents grow step by step."""
        ids = [0, 1, 2, 3, 4]
        seen = []
        DelimiterPRMScorer()([{'role': 'user', 'content': 'q'}], ids, _dec_newline(ids),
                             lambda rows: (seen.extend(rows), [0.0] * len(rows))[1], _dec_newline)
        assert [row['messages'][0]['content'] for row in seen] == ['q'] * 3  # prompt preserved
        assert [row['messages'][-1]['content'] for row in seen] == ['x\n', 'x\nyx\n', 'x\nyx\nyx\n']

    def test_no_delimiter_is_one_step(self):
        ids = [1, 3]  # 'yy' -- no newline anywhere
        out = DelimiterPRMScorer()([], ids, 'yy', lambda rows: [4.0], _dec_newline)
        assert out == [4.0, 4.0]

    def test_empty_response_scores_empty(self):
        assert DelimiterPRMScorer()([], [], '', lambda rows: [], _dec_newline) == []


class TestTurnPRMScorer:

    def _context(self):
        messages = [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 't1'},
            {'role': 'user', 'content': 'q2'},
            {'role': 'assistant', 'content': 't2'},
        ]
        return MultiTurnContext(messages=messages, turn_spans=[(0, 2), (2, 5)],
                                assistant_message_indices=[1, 3])

    def test_turns_are_steps(self):
        """S8-54: one assistant turn is one step; spans (0,2),(2,5) with scores [5,7] broadcast to
        [5,5,7,7,7]."""
        ctx = self._context()
        ids = [0, 1, 2, 3, 4]
        out = TurnPRMScorer()([], ids, _dec_alpha(ids), lambda rows: [5.0, 7.0], _dec_alpha,
                              turn_context=ctx)
        assert out == [5.0, 5.0, 7.0, 7.0, 7.0]

    def test_step_rows_use_turn_history(self):
        """S8-55: each turn's row is the history BEFORE that turn plus the turn's own tokens -- turn 0
        sees [user q]; turn 1 sees [user q, assistant t1, user q2]. The tool observation a turn
        triggers is context for the NEXT turn, never its own."""
        ctx = self._context()
        seen = []
        TurnPRMScorer()([], [0, 1, 2, 3, 4], 'abcxy', lambda rows: (seen.extend(rows), [0.0, 0.0])[1],
                        _dec_alpha, turn_context=ctx)
        assert [m['content'] for m in seen[0]['messages']] == ['q', 'ab']  # turn 0: q + tokens[0:2]
        assert [m['content'] for m in seen[1]['messages']] == ['q', 't1', 'q2', 'cxy']  # turn 1

    def test_single_turn_rejected(self):
        """S8-54 guard: a single-turn response has no turn structure -- point at the delimiter scorer
        instead of silently scoring the whole response as one turn."""
        with pytest.raises(ValueError, match='turn_context'):
            TurnPRMScorer()([], [0, 1], 'ab', lambda rows: [0.0], _dec_alpha)


class TestMultiTurnContext:
    """S8-55: the multi-turn ledger the turn scorer and the worker's privileged views share."""

    def test_turn_index_for_and_span_lookup(self):
        ctx = TestTurnPRMScorer()._context()
        assert ctx.turn_index_for(0) == 0
        assert ctx.turn_index_for(3) == 1
        assert ctx.turn_index_for(9) == -1  # outside every span

    def test_history_before_deep_copies(self):
        """history_before hands out a private copy -- mutating it must not corrupt the ledger."""
        ctx = TestTurnPRMScorer()._context()
        history = ctx.history_before(1)
        assert [m['role'] for m in history] == ['user', 'assistant', 'user']
        history[0]['content'] = 'mutated'
        assert ctx.messages[0]['content'] == 'q'


def test_prm_scorers_registry():
    """The two registered scorers are exactly the delimiter and turn implementations."""
    assert set(prm_scorers) == {'delimiter', 'turn'}
    assert prm_scorers['delimiter'] is DelimiterPRMScorer
    assert prm_scorers['turn'] is TurnPRMScorer
