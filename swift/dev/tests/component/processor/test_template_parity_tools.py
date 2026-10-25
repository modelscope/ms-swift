# Copyright (c) ModelScope Contributors. All rights reserved.
"""Jinja-parity variants beyond the generic prompt: tools injection and tool-call round-trips.

The generic parity suite (test_chat_template_parity) pins every family's multi-turn prompt against
``apply_chat_template``. These variants pin the TOOL surface of the same seam: a tools schema rides
the row, and the rendered prompt must equal the model author's jinja rendering with ``tools=``
passed through. A drift here means inference sees different tool documentation than training did.

Maps to ut_plan S18-44 (tools) and S18-46 (agent trajectory half: the same render with an
assistant tool-call turn in the history).
"""
import pytest

from swift.dev.tests.tiny import TinyModel

TOOLS = [{
    'type': 'function',
    'function': {
        'name': 'add',
        'description': 'Add two numbers.',
        'parameters': {
            'type': 'object',
            'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
            'required': ['a', 'b'],
        },
    },
}]
#: arguments as a DICT: both sides render an inline JSON object. (A str-arguments form is a
#: known divergence -- dev normalizes it to the dict rendering while the model jinja keeps the
#: escaped string -- registered as a cross-repo contract warning in the ut_plan annotation.)
CALL_TURN = {'role': 'assistant', 'content': '',
             'tool_calls': [{'id': 'call_1', 'type': 'function', 'function':
                             {'name': 'add', 'arguments': {'a': 1, 'b': 2}}}]}

#: (template, model id) pairs whose tokenizer is locally fetchable and ships a chat_template with
#: tool support. Families beyond these either gate their repos or ship no tool jinja -- they are
#: skipped with that reason rather than silently passing.
PAIRS = [
    ('qwen2_5', 'Qwen/Qwen2.5-0.5B-Instruct'),
]


def _tokenizer(model_id):
    from swift.model import get_model_processor

    try:
        model_dir = TinyModel.tokenizer_dir(model_id)
    except Exception as exc:  # noqa: BLE001 -- an unfetchable repo is not a template defect
        pytest.skip(f'{model_id}: tokenizer not fetchable ({type(exc).__name__})')
    _, processor = get_model_processor(model_dir, load_model=False)
    tok = processor if hasattr(processor, 'apply_chat_template') else processor.tokenizer
    if getattr(tok, 'chat_template', None) is None:
        pytest.skip(f'{model_id} ships no chat_template')
    return tok


@pytest.mark.slow
@pytest.mark.parametrize('template, model_id', PAIRS)
def test_tool_schema_renders_like_the_model_jinja(template, model_id):
    """S18-44: dev's rendered ids with a tools schema equal ``apply_chat_template(tools=...)``."""
    from swift.dev.builders import build_template
    from swift.dev.config import TemplateConfig
    from swift.dev.template import INFER

    tok = _tokenizer(model_id)
    conversation = [{'role': 'user', 'content': 'add 1 and 2'}]
    reference = list(tok.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, tools=TOOLS)['input_ids'])
    dev_template = build_template(TemplateConfig(template=template, max_length=1024), tok)
    rendered = list(dev_template.encode({'messages': conversation, 'tools': TOOLS},
                                        intent=INFER)['input_ids'])
    assert rendered == reference, (
        f'{template}: dev renders\n  {tok.decode(rendered)!r}\n'
        f'but the model jinja renders\n  {tok.decode(reference)!r}')


@pytest.mark.slow
@pytest.mark.parametrize('template, model_id', PAIRS)
def test_tool_call_turn_round_trips_into_the_prompt(template, model_id):
    """S18-46: an assistant tool-call turn in the history renders like the model jinja -- the
    trajectory half of the agent seam (what a rollout feeds back on the next turn)."""
    from swift.dev.builders import build_template
    from swift.dev.config import TemplateConfig
    from swift.dev.template import TRAIN

    tok = _tokenizer(model_id)
    # The dev template requires strict (query, assistant) pairs: a tool observation rides the
    # query slot, so the trajectory is [user, tool-call, tool-result, assistant].
    conversation = [
        {'role': 'user', 'content': 'add 1 and 2'},
        dict(CALL_TURN),
        {'role': 'tool', 'name': 'add', 'content': '3'},
        {'role': 'assistant', 'content': 'The sum is 3.'},
    ]
    reference = list(tok.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False, tools=TOOLS)['input_ids'])
    dev_template = build_template(TemplateConfig(template=template, max_length=1024), tok)
    rendered = list(dev_template.encode({'messages': conversation, 'tools': TOOLS},
                                        intent=TRAIN)['input_ids'])
    assert rendered == reference, (
        f'{template}: dev renders\n  {tok.decode(rendered)!r}\n'
        f'but the model jinja renders\n  {tok.decode(reference)!r}')
