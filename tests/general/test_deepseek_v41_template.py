# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import pytest
from types import SimpleNamespace

from swift.agent_template import agent_template_map
from swift.template import TEMPLATE_MAPPING, StdTemplateInputs

BOS = '<｜begin▁of▁sentence｜>'
EOS = '<｜end▁of▁sentence｜>'
SYSTEM = '<｜System｜>'
USER = '<｜User｜>'
ASSISTANT = '<｜Assistant｜>'
TOOLS = [{'name': 'search', 'description': 'Search', 'parameters': {'type': 'object'}}]


@pytest.fixture
def make_template(monkeypatch):
    monkeypatch.delenv('REASONING_EFFORT', raising=False)

    def make(template_type='deepseek_v41', **kwargs):
        meta = TEMPLATE_MAPPING[template_type]
        template = meta.template_cls(None, meta, **kwargs)
        # These tests exercise prompt serialization before tokenization; no model download is needed.
        template.model_meta = SimpleNamespace(is_multimodal=template_type == 'deepseek_v41')
        template.init_env_args()
        return template

    return make


def render(template, messages, **kwargs):
    inputs = StdTemplateInputs.from_dict({'messages': messages, **kwargs})
    template._swift_prepare_inputs(inputs)
    contexts, scales, _ = template._swift_encode(inputs)
    return ''.join(contexts), list(zip(contexts, scales))


@pytest.mark.parametrize('agent_name,tag_prefix,calls_tag', [
    ('deepseek_v4', '｜DSML｜', 'tool_calls'),
    ('deepseek_v41', '｜DSML｜ ', 'calls'),
])
def test_tool_call_protocol(agent_name, tag_prefix, calls_tag):
    agent = agent_template_map[agent_name]()
    arguments = {'query': '北京\nweather', 'days': 3, 'metric': True, 'extra': None, 'tags': ['a', 'b']}
    messages = [
        {
            'role': 'tool_call',
            'content': {
                'name': 'search',
                'arguments': arguments
            }
        },
        {
            'role': 'tool_call',
            'content': {
                'name': 'finish',
                'arguments': '{}'
            }
        },
    ]
    encoded = agent._format_tool_calls(messages)
    assert encoded == (f'<{tag_prefix}{calls_tag}>\n'
                       f'<{tag_prefix}invoke name="search">\n'
                       f'<{tag_prefix}parameter name="query" string="true">北京\nweather</{tag_prefix}parameter>\n'
                       f'<{tag_prefix}parameter name="days" string="false">3</{tag_prefix}parameter>\n'
                       f'<{tag_prefix}parameter name="metric" string="false">true</{tag_prefix}parameter>\n'
                       f'<{tag_prefix}parameter name="extra" string="false">null</{tag_prefix}parameter>\n'
                       f'<{tag_prefix}parameter name="tags" string="false">["a", "b"]</{tag_prefix}parameter>\n'
                       f'</{tag_prefix}invoke>\n'
                       f'<{tag_prefix}invoke name="finish">\n\n</{tag_prefix}invoke>\n'
                       f'</{tag_prefix}{calls_tag}>')
    functions = agent.get_toolcall(encoded)
    assert [function.name for function in functions] == ['search', 'finish']
    assert [json.loads(function.arguments) for function in functions] == [arguments, {}]
    tools_prompt = agent._format_tools(TOOLS)
    assert f'<{tag_prefix}{calls_tag}>' in tools_prompt
    assert f'<{tag_prefix}invoke name="$TOOL_NAME">' in tools_prompt
    assert f'<{tag_prefix}parameter name="$PARAMETER_NAME"' in tools_prompt
    assert tools_prompt.startswith('\n\n## Tools' if agent_name == 'deepseek_v41' else '## Tools')
    assert agent._format_tools(TOOLS, 'Policy').startswith('Policy\n\n## Tools')


@pytest.mark.parametrize('effort,budget', [(None, 75), ('low', 50), ('high', 75), ('max', 100), (1, 1), (62, 62),
                                           (100, 100)])
def test_reasoning_budget(make_template, effort, budget):
    template = make_template(enable_thinking=True)
    text, _ = render(template, [{'role': 'user', 'content': 'Hi'}], chat_template_kwargs={'reasoning_effort': effort})
    assert text == (
        BOS + SYSTEM + f'Reasoning Effort: {budget} '
        '(range 1-100, the higher the value, the more thorough the reasoning)\n\n' + USER + 'Hi' + ASSISTANT
        + '<think>')


@pytest.mark.parametrize('effort', [0, 101, True, 1.5, '75', 'unknown'])
def test_invalid_request_reasoning_budget(make_template, effort):
    template = make_template(enable_thinking=True)
    with pytest.raises(ValueError, match='integer in'):
        render(template, [{'role': 'user', 'content': 'Hi'}], chat_template_kwargs={'reasoning_effort': effort})


@pytest.mark.parametrize('env_effort,request_effort,budget', [('61', None, 61), ('max', None, 100), ('61', 'low', 50)])
def test_reasoning_budget_environment(make_template, monkeypatch, env_effort, request_effort, budget):
    monkeypatch.setenv('REASONING_EFFORT', env_effort)
    template = make_template(enable_thinking=True)
    text, _ = render(
        template, [{
            'role': 'user',
            'content': 'Hi'
        }], chat_template_kwargs={'reasoning_effort': request_effort})
    assert f'Reasoning Effort: {budget} ' in text


@pytest.mark.parametrize('enable_thinking,request_kwargs', [
    (False, {
        'reasoning_effort': 'max'
    }),
    (True, {
        'reasoning_effort': 100,
        'enable_thinking': False
    }),
])
def test_budget_does_not_enable_thinking(make_template, enable_thinking, request_kwargs):
    template = make_template(enable_thinking=enable_thinking)
    text, _ = render(template, [{'role': 'user', 'content': 'Hi'}], chat_template_kwargs=request_kwargs)
    assert text == BOS + USER + 'Hi' + ASSISTANT + '</think>'


@pytest.mark.parametrize('tools', [None, TOOLS])
def test_history_thinking(make_template, tools):
    template = make_template(enable_thinking=True, preserve_thinking=False)
    text, _ = render(
        template, [
            {
                'role': 'user',
                'content': 'First'
            },
            {
                'role': 'assistant',
                'content': '<think>Old reasoning</think> Answer with spaces '
            },
            {
                'role': 'user',
                'content': 'Next'
            },
        ],
        tools=tools)
    history = '<think>Old reasoning</think>' if tools else '</think>'
    assert ASSISTANT + history + ' Answer with spaces ' + EOS + USER + 'Next' in text
    assert text.endswith(ASSISTANT + '<think>')


@pytest.mark.parametrize('enable_thinking,prefix', [(True, '<think></think>'), (False, '</think>')])
def test_tool_history_channel_markers(make_template, enable_thinking, prefix):
    template = make_template(enable_thinking=enable_thinking, preserve_thinking=False)
    text, _ = render(
        template, [
            {
                'role': 'user',
                'content': 'Search'
            },
            {
                'role': 'tool_call',
                'content': {
                    'name': 'search',
                    'arguments': {}
                }
            },
            {
                'role': 'tool_response',
                'content': 'Found'
            },
            {
                'role': 'assistant',
                'content': 'Done'
            },
            {
                'role': 'user',
                'content': 'Next'
            },
        ],
        tools=TOOLS)
    assert ASSISTANT + prefix + '\n\n<｜DSML｜ calls>' in text
    assert EOS + USER + '<tool_result>Found</tool_result>' + ASSISTANT + prefix + 'Done' + EOS in text


@pytest.mark.parametrize('loss_scale', ['default', 'last_round'])
def test_system_queries_in_multiple_rounds(make_template, loss_scale):
    template = make_template(enable_thinking=False, loss_scale=loss_scale)
    template.set_mode('train')
    text, contexts = render(template, [
        {
            'role': 'system',
            'content': 'Policy'
        },
        {
            'role': 'user',
            'content': 'First'
        },
        {
            'role': 'system',
            'content': 'Hint 1'
        },
        {
            'role': 'assistant',
            'content': 'Answer 1'
        },
        {
            'role': 'system',
            'content': 'Hint 2'
        },
        {
            'role': 'user',
            'content': 'Second'
        },
        {
            'role': 'assistant',
            'content': 'Answer 2'
        },
    ])
    assert text == (
        BOS + SYSTEM + 'Policy' + USER + 'First' + SYSTEM + 'Hint 1' + ASSISTANT + '</think>Answer 1' + EOS + SYSTEM
        + 'Hint 2' + USER + 'Second' + ASSISTANT + '</think>Answer 2' + EOS)
    for context, scale in contexts:
        if any(query in context for query in ('Policy', 'First', 'Hint 1', 'Hint 2', 'Second')):
            assert scale == 0
        if 'Answer 1' in context:
            assert scale == (1 if loss_scale == 'default' else 0)
        if 'Answer 2' in context:
            assert scale == 1


def test_system_query_after_tool_response(make_template):
    template = make_template(enable_thinking=False)
    text, _ = render(template, [
        {
            'role': 'user',
            'content': 'Search'
        },
        {
            'role': 'tool_call',
            'content': {
                'name': 'search',
                'arguments': {}
            }
        },
        {
            'role': 'tool_response',
            'content': 'Found'
        },
        {
            'role': 'system',
            'content': 'Use the result'
        },
    ])
    assert text.endswith(EOS + USER + '<tool_result>Found</tool_result>' + SYSTEM + 'Use the result' + ASSISTANT
                         + '</think>')


def test_system_query_preserves_response_loss(make_template):
    template = make_template(enable_thinking=False)
    template.set_mode('train')
    _, contexts = render(template, [
        {
            'role': 'user',
            'content': 'First'
        },
        {
            'role': 'assistant',
            'content': 'Answer 1'
        },
        {
            'role': 'system',
            'content': 'Hint'
        },
        {
            'role': 'assistant',
            'content': 'Answer 2',
            'loss': False
        },
    ])
    assert ('</think>Answer 1', 1) in contexts
    assert ('</think>Answer 2', 0) in contexts


def test_v4_effort_prompt_is_unchanged(make_template, monkeypatch):
    monkeypatch.setenv('REASONING_EFFORT', 'max')
    template = make_template('deepseek_v4', enable_thinking=True)
    text, _ = render(template, [{'role': 'user', 'content': 'Hi'}])
    assert text.startswith(BOS + 'Reasoning Effort: Absolute maximum with no shortcuts permitted.\n')
    assert SYSTEM not in text
    assert text.endswith(USER + 'Hi' + ASSISTANT + '<think>')
