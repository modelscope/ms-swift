"""S8c — dev2 scalar reward-model row protocol + the LLM-judge text kernels (L1).

``RewardModel.score`` is the dev2 seq_cls reward-model handle: it encodes message rows and forwards them
through the score head, returning one float per row. The contract that matters is ``task='seq_cls'`` --
forwarding the default 'causal_lm' would gather label logps out of a pooled [B, 1] score (a shape error on
transformers, and megatron would miss the seq_cls pooling branch). The judge kernels (render/parse/query)
are the pure-text half of the generative/api judge. Stubs stand in for the model/template, so no GPU.

Maps to ut_plan S8-43 (RM row protocol + batch scoring) and the judge text kernels behind S8-44's channel.
"""
import pytest


class _StubTemplate:

    def __init__(self):
        self.rows = []

    def encode(self, row):
        self.rows.append(row)
        return {'input_ids': [len(row['messages'])]}


class _StubSeqClsModel:
    """A frozen scalar RM: forward_only must be routed as seq_cls and emits a [B, 1] score."""

    def __init__(self, logits=None):
        self._logits = logits
        self.calls = []

    def forward_only(self, *, inputs, return_logits, task):
        self.calls.append((return_logits, task))
        assert return_logits is True
        assert task == 'seq_cls', f'scalar RM forward_only must be routed as seq_cls, got task={task!r}'
        if self._logits is None:
            import torch
            return {'logits': torch.tensor([[float(item['input_ids'][0])] for item in inputs])}
        return self._logits


def test_reward_model_scores_rows_via_seq_cls_head():
    """S8-43: each message row is deep-copied, encoded, and scored through the seq_cls head -> one float
    per row (the [B, 1] score flattened); the metric label is the handle's display_name."""
    from swift.dev.rewards.scorer import RewardModel

    template = _StubTemplate()
    model = _StubSeqClsModel()
    rm = RewardModel(model, template, display_name='my_rm')
    rows = [
        {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]},
        {'messages': [{'role': 'user', 'content': 'q2'}]},
    ]
    scores = rm.score(rows)
    assert scores == [2.0, 1.0]  # len(messages) per row, read back through the head
    assert model.calls == [(True, 'seq_cls')]  # routed as seq_cls, logits requested
    assert template.rows[0] is not rows[0]  # the row is deep-copied before encode (never mutated in place)
    assert rm.display_name == 'my_rm'


def test_reward_model_raises_when_forward_returns_no_logits():
    """S8-43 guard: a forward that yields no logits is a hard error, not a silent zero score."""
    from swift.dev.rewards.scorer import RewardModel

    rm = RewardModel(_StubSeqClsModel(logits={}), _StubTemplate())
    with pytest.raises(RuntimeError, match='returned no logits'):
        rm.score([{'messages': [{'role': 'user', 'content': 'q'}]}])


# ----------------------------------------------------------------------
# The LLM-judge text kernels (the generative/api judge's pure half)
# ----------------------------------------------------------------------
@pytest.mark.parametrize('text, expected', [
    ('verdict Reward: 0.8 done', 0.8),  # the prompted Reward: convention wins
    ('Reward: 5', 5.0),  # out of the [0,1] Reward range -> falls back to the first bare number
    ('the answer scored 42 ok', 42.0),  # no Reward: tag -> first bare number
    ('got -3.5 here', -3.5),  # a signed decimal
    ('no digits at all', None),  # nothing numeric -> None (recorded as nan), never a guessed 0
    ('', None),
    (None, None),
])
def test_parse_judge_score(text, expected):
    """S8-44 support: a judge's free-form verdict is parsed to a float, or None when unparseable (a silent
    0 would rank an unparseable verdict as a genuinely bad candidate)."""
    from swift.dev.rewards.scorer import parse_judge_score

    assert parse_judge_score(text) == expected


def test_render_conversation_flattens_and_skips_incomplete():
    """S8-44 support: messages flatten to ``Role: content`` lines; a contentless or non-dict entry is skipped."""
    from swift.dev.rewards.scorer import render_conversation

    messages = [
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'yo'},
        {'role': 'system'},  # no content -> skipped
        'junk',  # not a dict -> skipped
    ]
    assert render_conversation(messages) == 'User: hi\nAssistant: yo'
    assert render_conversation(None) == ''


def test_judge_query_strips_trailing_assistant_and_formats():
    """S8-44 support: the candidate (trailing assistant turn) is stripped from the rendered prompt and
    supplied verbatim through the template's {completion}, the conversation through {prompt}."""
    from swift.dev.rewards.scorer import judge_query

    messages = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'cand'}]
    assert judge_query('P={prompt} C={completion}', messages, 'cand') == 'P=User: q C=cand'
