from types import SimpleNamespace

import pytest

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
    inputs = StdTemplateInputs(messages=[
        {'role': 'user', 'content': 'question'},
        {'role': 'assistant', 'content': response},
    ])
    template._swift_encode(inputs)
    return template


def test_swift_encode_supports_mixed_text_and_raw_token_response():
    response = [
        'existing rollout content',
        {
            'loss_scale': [0] * 25,
            'token_ids': list(range(25)),
        },
    ]

    template = encode_response(response)

    assert template.tokenizer.decoded_ids == [list(range(5, 25))]
    assert template.recorded_responses == [response]
    assert template.recorded_responses[0] is response


@pytest.mark.parametrize('response, expected_decoded_ids', [
    ('plain text response', []),
    (['previous response', 'plain text response'], []),
    ([1, 2, 3], [[1, 2, 3]]),
    ({'loss_scale': [1, 1, 1], 'token_ids': [1, 2, 3]}, [[1, 2, 3]]),
])
def test_swift_encode_preserves_existing_response_types(response, expected_decoded_ids):
    template = encode_response(response)

    assert template.tokenizer.decoded_ids == expected_decoded_ids
    assert template.recorded_responses == [response]
    assert template.recorded_responses[0] is response
