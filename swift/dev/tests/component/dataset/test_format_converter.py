# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end tests for the ``format_converter`` layer.

Two seams are covered, each driven through the real code path rather than a stub:

- **The factory** (:func:`get_converter`) -- column names in, the right converter out. Its whole job is
  resolution order (explicit pin, then lowest-:attr:`priority` detector, then the fallback), so the
  tests pin that order down, including the ties the priority numbers exist to break.
- **Each converter's row rewrite** -- driven through :class:`Preprocessor`, i.e. a real ``dataset.map``,
  so a row passes through ``detect`` -> ``convert`` -> ``check_messages`` -> ``cast_mm_data`` exactly as
  it does in production. The assertion is on the standard ``messages`` (and lifted ``images``) that come
  out the other side.

The rows a converter produces are then fed through a real template (tokenizer only, ``load_model=False``)
to prove the ``messages`` -> ``input_ids`` link, closing the loop on "the dataset runs end to end". Those
encode tests need the local model and are skipped without it; the factory and convert tests are pure
Python and always run. Template tool-calling behaviour is deliberately out of scope here.
"""
import os
from typing import Any, Dict, List

import pytest
from datasets import Dataset as HfDataset

MODEL = os.environ.get('SWIFT_TEST_MODEL', '/mnt/workspace/yzhao/tastelikefeet/Qwen3.5-4B-CM-v2')
MODEL_TYPE = os.environ.get('SWIFT_TEST_MODEL_TYPE', 'qwen3_5')

needs_model = pytest.mark.skipif(not os.path.isdir(MODEL), reason=f'no local model at {MODEL}')


def convert_rows(rows: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """Run the default :class:`Preprocessor` over ``rows`` through a real ``map``.

    Returns the standard rows with null columns dropped, so an assertion reads as "exactly these
    columns", the way ``test_parity.py`` compares them.
    """
    from swift.dev.dataset import Preprocessor
    processed = Preprocessor(**kwargs)(HfDataset.from_list(rows), load_from_cache_file=False)
    return [{key: value for key, value in row.items() if value is not None} for row in processed]


@pytest.fixture(scope='module')
def processor():
    from swift.model import get_model_processor
    return get_model_processor(MODEL, load_model=False, model_type=MODEL_TYPE)[1]


def build_template(processor, max_length=8192):
    from swift.template import get_template
    template = get_template(processor, template_type='qwen2_5', max_length=max_length)
    template.set_mode('train')
    return template


# ---- the factory: resolution order -----------------------------------------------------------


def test_list_formats_is_in_detection_order():
    from swift.dev.dataset.format_converter import list_formats
    # Lowest priority first -- the order the factory asks detectors in.
    assert list_formats() == ['openai', 'anthropic', 'alpaca', 'response']


def test_dialogue_column_beats_an_incidental_instruction_column():
    """The tie ``priority`` exists to break: a dataset with both is a dialogue dataset."""
    from swift.dev.dataset.format_converter import OpenAIConverter, get_converter
    assert isinstance(get_converter(['conversations', 'instruction', 'input']), OpenAIConverter)


def test_alpaca_wins_over_the_response_fallback():
    from swift.dev.dataset.format_converter import AlpacaConverter, get_converter
    assert isinstance(get_converter(['instruction', 'input', 'output']), AlpacaConverter)


def test_unrecognised_columns_fall_back_to_response():
    """No detector claims these, so the fallback pass routes to ResponseConverter."""
    from swift.dev.dataset.format_converter import ResponseConverter, get_converter
    assert isinstance(get_converter(['some_odd_column']), ResponseConverter)


def test_explicit_format_name_pins_a_converter_whose_detect_would_say_no():
    """Anthropic's ``detect`` always returns False (it shares the messages column); a pin overrides that."""
    from swift.dev.dataset.format_converter import AnthropicConverter, get_converter
    assert isinstance(get_converter(['messages'], format_name='anthropic'), AnthropicConverter)


def test_unknown_format_name_raises_with_the_available_set():
    from swift.dev.dataset.format_converter import get_converter
    with pytest.raises(ValueError, match='not registered'):
        get_converter(['messages'], format_name='no_such_format')


def test_caller_aliases_are_applied_before_detection():
    """A dataset named so unusually no format recognises it, renamed by the caller into one that is."""
    from swift.dev.dataset.format_converter import AlpacaConverter, get_converter
    converter = get_converter(['instr', 'inp'], aliases={'instr': 'instruction', 'inp': 'input'})
    assert isinstance(converter, AlpacaConverter)


# ---- alias resolution: the two contest rules -------------------------------------------------


def test_a_standard_name_present_beats_its_own_alias():
    """``response`` and ``output`` both present: ``response`` is the one meant, ``output`` is left alone."""
    from swift.dev.dataset.format_converter import ResponseConverter
    assert ResponseConverter().resolve_aliases({'response', 'output'}) == {}


def test_when_only_aliases_compete_the_first_declared_wins():
    """``answer`` and ``output`` both map to ``response``; ``answer`` is declared first, so it wins."""
    from swift.dev.dataset.format_converter import ResponseConverter
    assert ResponseConverter().resolve_aliases({'answer', 'output'}) == {'answer': 'response'}


def test_resolution_does_not_depend_on_column_order():
    from swift.dev.dataset.format_converter import ResponseConverter
    converter = ResponseConverter()
    assert converter.resolve_aliases(['output', 'answer']) == converter.resolve_aliases(['answer', 'output'])


# ---- ResponseConverter -----------------------------------------------------------------------


def test_response_plain_query_and_response():
    rows = convert_rows([{'query': 'q', 'response': 'a'}])
    assert rows == [{'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]}]


def test_response_aliases_rename_prompt_and_answer():
    rows = convert_rows([{'prompt': 'q', 'answer': 'a'}])
    assert rows == [{'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]}]


def test_response_prepends_system_and_flattens_history():
    rows = convert_rows([{'system': 's', 'query': 'q', 'response': 'a', 'history': [['h1', 'h2']]}])
    assert rows == [{
        'messages': [
            {'role': 'system', 'content': 's'},
            {'role': 'user', 'content': 'h1'},
            {'role': 'assistant', 'content': 'h2'},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a'},
        ]
    }]


def test_response_parses_a_history_that_arrived_as_a_string_literal():
    rows = convert_rows([{'query': 'q', 'response': 'a', 'history': "[['h1', 'h2']]"}])
    messages = rows[0]['messages']
    assert messages[0] == {'role': 'user', 'content': 'h1'}
    assert messages[-1] == {'role': 'assistant', 'content': 'a'}


def test_response_collapses_a_multi_answer_list_to_the_first():
    rows = convert_rows([{'query': 'q', 'response': ['a1', 'a2']}])
    assert rows[0]['messages'][-1] == {'role': 'assistant', 'content': 'a1'}


def test_response_keeps_rejected_response_flat():
    """The template owns expanding ``rejected_response``; the converter must not do it early."""
    rows = convert_rows([{'query': 'q', 'response': 'a', 'rejected_response': 'bad'}])
    assert rows[0]['rejected_response'] == 'bad'
    assert 'rejected_messages' not in rows[0]


# ---- AlpacaConverter -------------------------------------------------------------------------


def test_alpaca_joins_instruction_and_input_into_one_user_turn():
    rows = convert_rows([{'instruction': 'i', 'input': 'x', 'output': 'o'}])
    assert rows == [{'messages': [{'role': 'user', 'content': 'i\nx'}, {'role': 'assistant', 'content': 'o'}]}]


def test_alpaca_tolerates_an_empty_input():
    rows = convert_rows([{'instruction': 'i', 'input': '', 'output': 'o'}])
    assert rows == [{'messages': [{'role': 'user', 'content': 'i'}, {'role': 'assistant', 'content': 'o'}]}]


def test_instruction_without_input_is_a_response_dataset():
    """No ``input`` column means Alpaca's detect says no; the response alias ``instruction``->query takes over."""
    rows = convert_rows([{'instruction': 'i', 'output': 'o'}])
    assert rows == [{'messages': [{'role': 'user', 'content': 'i'}, {'role': 'assistant', 'content': 'o'}]}]


# ---- OpenAIConverter: the swift dialect ------------------------------------------------------


def test_openai_plain_messages_pass_through():
    raw = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]
    rows = convert_rows([{'messages': raw}])
    assert rows == [{'messages': raw}]


def test_openai_conversations_alias_with_from_value_keys():
    raw = [{'from': 'human', 'value': 'q'}, {'from': 'gpt', 'value': 'a'}]
    rows = convert_rows([{'conversations': raw}])
    assert rows == [{'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]}]


def test_openai_turn_pairs_expand_one_message_per_party():
    rows = convert_rows([{'messages': [{'user': 'q', 'assistant': 'a'}]}])
    assert rows == [{'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]}]


# ---- OpenAIConverter: the openai (tool_calls) dialect ----------------------------------------


def test_openai_tool_calls_expand_into_a_tool_call_message():
    raw = [{
        'role': 'user',
        'content': 'weather?'
    }, {
        'role': 'assistant',
        'content': None,
        'tool_calls': [{'function': {'name': 'get_weather', 'arguments': '{"city": "SF"}'}}]
    }, {
        'role': 'tool',
        'content': 'sunny'
    }]
    rows = convert_rows([{'messages': raw}])
    assert rows == [{
        'messages': [
            {'role': 'user', 'content': 'weather?'},
            # arguments arrives as a JSON string and is parsed to a dict.
            {'role': 'tool_call', 'content': {'name': 'get_weather', 'arguments': {'city': 'SF'}}},
            {'role': 'tool_response', 'content': 'sunny'},
        ]
    }]


# ---- OpenAIConverter: the anthropic dialect (auto-detected) ----------------------------------


def test_anthropic_content_blocks_expand_and_lift_images():
    raw = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'what is this'},
            {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'AAA'}},
        ]
    }, {
        'role': 'assistant',
        'content': [{'type': 'tool_use', 'name': 'search', 'input': {'q': 1}}]
    }, {
        'role': 'user',
        'content': [{'type': 'tool_result', 'content': 'result'}]
    }]
    rows = convert_rows([{'messages': raw}])
    assert rows == [{
        'messages': [
            {'role': 'user', 'content': 'what is this<image>'},
            {'role': 'tool_call', 'content': {'name': 'search', 'arguments': {'q': 1}}},
            {'role': 'tool_response', 'content': 'result'},
        ],
        # The image block is lifted out of the dialogue into the standard top-level images column,
        # as a base64 data URI, then cast to the {bytes, path} layout the encoder expects.
        'images': [{'bytes': None, 'path': 'data:image/png;base64,AAA'}],
    }]


def test_anthropic_url_image_block_yields_a_plain_url():
    raw = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'look'},
        {'type': 'image', 'source': {'type': 'url', 'url': 'http://x/y.png'}},
    ]}]
    rows = convert_rows([{'messages': raw}])
    assert rows[0]['images'] == [{'bytes': None, 'path': 'http://x/y.png'}]
    assert rows[0]['messages'][0]['content'] == 'look<image>'


def test_anthropic_pinned_format_skips_auto_detection():
    raw = [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}]
    rows = convert_rows([{'messages': raw}], format_name='anthropic')
    assert rows == [{'messages': [{'role': 'user', 'content': 'hi'}]}]


# ---- the messages -> input_ids link, end to end through a real template ----------------------


@needs_model
@pytest.mark.parametrize(
    'raw',
    [
        [{'query': 'hello there', 'response': 'hi'}],
        [{'instruction': 'say hi', 'input': 'please', 'output': 'hi'}],
        [{'messages': [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'hi'}]}],
        [{'conversations': [{'from': 'human', 'value': 'hello'}, {'from': 'gpt', 'value': 'hi'}]}],
    ],
    ids=['response', 'alpaca', 'openai-swift', 'from-value'])
def test_converted_rows_encode_to_input_ids(processor, raw):
    """The whole chain: raw format -> standard messages -> tokens. If this runs, the format is usable."""
    from swift.dev.dataset import EncodePreprocessor, Preprocessor
    standard = Preprocessor()(HfDataset.from_list(raw), load_from_cache_file=False)
    encoded = EncodePreprocessor(build_template(processor))(standard, load_from_cache_file=False)
    assert len(encoded) == 1
    row = encoded[0]
    assert row['input_ids'] and all(isinstance(token, int) for token in row['input_ids'])
    assert row['labels'] and len(row['labels']) == len(row['input_ids'])
