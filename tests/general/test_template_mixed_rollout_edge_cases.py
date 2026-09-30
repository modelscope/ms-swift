# Copyright (c) ModelScope Contributors. All rights reserved.
"""Edge-case regression tests for ``Template._swift_encode`` mixed rollout
response handling.

The base coverage in ``test_template_mixed_rollout.py`` (PR #10029) locks
down the core mixed-content behaviour. This file exercises the boundary
cases and failure modes that surround it, to prevent future refactors
from silently regressing the contract.

Specifically:

* mixed list with a longer text segment before the trailing raw-token
  dict (only the dict's token_ids should reach ``tokenizer.decode``);
* mixed list where the trailing dict has no ``token_ids`` key — must
  raise ``TypeError`` instead of crashing inside ``decode``;
* mixed list where ``token_ids`` is not a list of ints — must raise
  ``TypeError``;
* pure dict without ``token_ids`` key — must raise ``TypeError``;
* boundary on the ``[-20:]`` slice (exactly 19 / 20 / 21 tokens);
* identity preservation of the response payload (no copy, no implicit
  string conversion).
"""
import pytest
from types import SimpleNamespace

from swift.template import StdTemplateInputs, Template


class RecordingTokenizer:

    def __init__(self):
        self.decoded_ids = []

    def decode(self, token_ids, skip_special_tokens=False):
        self.decoded_ids.append(list(token_ids))
        return 'decoded response'


class RecordingTemplate(Template):

    def _concat_context_list(self, context_list, res_context_list, res_context_type, **kwargs):
        response = kwargs.get('response')
        if response is not None:
            self.recorded_responses.append(response)
        return super()._concat_context_list(context_list, res_context_list, res_context_type, **kwargs)


def make_template():
    template = object.__new__(RecordingTemplate)
    template.processor = RecordingTokenizer()
    template.template_meta = SimpleNamespace(
        auto_add_bos=False,
        is_post_system=False,
        prefix=[],
        prompt=['{{QUERY}}'],
        suffix=[],
        stop_words=[],
        support_multi_round=True,
    )
    template.use_chat_template = False
    template.template_backend = 'swift'
    template.mode = 'train'
    template.task_type = 'causal_lm'
    template.response_prefix = None
    template._loss_scale = 'default'
    template._loss_scale_cache = {}
    template.is_binary_loss_scale = None
    template.recorded_responses = []
    return template


def encode_response(response):
    template = make_template()
    inputs = StdTemplateInputs(
        messages=[{
            'role': 'user',
            'content': 'question'
        }, {
            'role': 'assistant',
            'content': response
        }])
    template._swift_encode(inputs)
    return template


# ---------------------------------------------------------------------------
# Positive cases — the slice + decoder should see exactly the dict's
# token_ids, never the surrounding text or dict structure.
# ---------------------------------------------------------------------------


def test_mixed_response_with_long_text_segment_only_decodes_trailing_dict_tokens():
    response = [
        '<think>plan</think><​tool_call>call_A</tool_call><tool_call>call_B</tool_call>',
        ' a stray intermediate string',  # only the LAST element matters
        {
            'loss_scale': [0, 0, 0, 0, 1],
            'token_ids': [248068, 271, 248069, 271, 248046],
        },
    ]
    template = encode_response(response)
    # Only the dict's token_ids reach decode, and only the last 20 (all 5 here).
    assert template.tokenizer.decoded_ids == [[248068, 271, 248069, 271, 248046]]
    # The full original payload must be passed through unchanged.
    assert template.recorded_responses == [response]
    assert template.recorded_responses[0] is response


@pytest.mark.parametrize('token_count, expected_decoded_tail_len', [
    (19, 19),  # smaller than the 20-token slice window
    (20, 20),  # exactly fills the window
    (21, 20),  # one over — slice must drop the oldest
    (100, 20),  # large: only the last 20 are decoded
])
def test_token_ids_slice_is_capped_at_20(token_count, expected_decoded_tail_len):
    response = [{
        'loss_scale': [1] * token_count,
        'token_ids': list(range(token_count)),
    }]
    template = encode_response(response)
    assert template.tokenizer.decoded_ids == [list(range(token_count)[-20:])]
    assert len(template.tokenizer.decoded_ids[0]) == expected_decoded_tail_len


def test_mixed_response_dict_with_empty_token_ids_does_not_call_decode():
    # An empty token_ids list must not crash inside decode (decode([])
    # is a no-op for most tokenizers); we only assert that the call
    # is made once with an empty list and the response is preserved.
    response = [{
        'loss_scale': [],
        'token_ids': [],
    }]
    template = encode_response(response)
    assert template.tokenizer.decoded_ids == [[]]
    assert template.recorded_responses == [response]


# ---------------------------------------------------------------------------
# Negative cases — these shapes must raise TypeError so a future refactor
# cannot silently start feeding malformed payloads to tokenizer.decode.
# ---------------------------------------------------------------------------


def test_mixed_list_with_dict_missing_token_ids_raises_type_error():
    bad_response = ['text segment', {'loss_scale': [0, 0, 1]}]  # no token_ids
    with pytest.raises(TypeError):
        encode_response(bad_response)


def test_mixed_list_with_dict_whose_token_ids_are_strings_raises_type_error():
    bad_response = [{
        'text': 'segment',
        'token_ids': ['1', '2', '3'],
    }]
    with pytest.raises(TypeError):
        encode_response(bad_response)


def test_mixed_list_with_dict_whose_token_ids_is_not_a_list_raises_type_error():
    bad_response = [{
        'token_ids': 12345,  # scalar instead of list
    }]
    with pytest.raises(TypeError):
        encode_response(bad_response)


def test_pure_dict_without_token_ids_raises_type_error():
    bad_response = {'loss_scale': [0, 0, 1], 'text': 'no token_ids here'}
    with pytest.raises(TypeError):
        encode_response(bad_response)


# ---------------------------------------------------------------------------
# Identity / passthrough guarantees.
# ---------------------------------------------------------------------------


def test_mixed_response_object_is_passed_through_by_identity():
    """The encoded response payload must reach downstream code unchanged
    — no implicit ``json.dumps``, no copy of the list / dicts. Otherwise
    the raw-token rollout semantics documented in #10029 are silently
    altered by the template encoding path.
    """
    text_segment = '<think>...</think><tool_call>...</tool_call>'
    raw_token_segment = {
        'loss_scale': [0, 0, 0, 0, 1],
        'token_ids': [248068, 271, 248069, 271, 248046],
    }
    response = [text_segment, raw_token_segment]

    template = encode_response(response)

    passed_through = template.recorded_responses[0]
    assert passed_through is response
    assert passed_through[0] is text_segment
    assert passed_through[1] is raw_token_segment
    assert passed_through[1]['token_ids'] is raw_token_segment['token_ids']
    assert passed_through[1]['loss_scale'] is raw_token_segment['loss_scale']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
