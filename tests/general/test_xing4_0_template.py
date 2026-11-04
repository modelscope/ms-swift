# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import pytest
from types import SimpleNamespace

from swift.agent_template import agent_template_map
from swift.template import TEMPLATE_MAPPING, StdTemplateInputs

# ref: https://modelscope.cn/models/XingChen-AGI/Xing4.0-29B-A4B (chat_template.jinja)
SYSTEM = '<_system>'
USER = '<_user>'
BOT = '<_bot>'
EOS = '<_end>'
OBSERVATION = '<_observation>'
THINK_OPEN = '<think>'
THINK_CLOSE = '</think>'
TOOL_CALL_OPEN = '<tool_call>'
TOOL_CALL_CLOSE = '</tool_call>'
TOOL_RESPONSE_OPEN = '<tool_response>'
TOOL_RESPONSE_CLOSE = '</tool_response>'
TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get weather of a city',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string',
                    'description': 'city name'
                }
            },
            'required': ['city']
        }
    }
}
TOOLS = [TOOL]
TOOL_CALL = (f'{TOOL_CALL_OPEN}get_weather<param_key>city</param_key><param_value>beijing</param_value>'
             f'<param_key>days</param_key><param_value>3</param_value>{TOOL_CALL_CLOSE}')
TOOLS_PROMPT = ('\n# Tools\n\nYou may call one or more functions to assist with the user query.'
                '\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n'
                f'{json.dumps(TOOL, ensure_ascii=False)}\n</tools>\n'
                '\nFor each function call, output the function name and arguments within the following XML format:\n'
                f'{TOOL_CALL_OPEN}{{function-name}}<param_key>{{param-key-1}}</param_key>'
                f'<param_value>{{param-value-1}}</param_value><param_key>{{param-key-2}}</param_key>'
                f'<param_value>{{param-value-2}}</param_value>...{TOOL_CALL_CLOSE}')


@pytest.fixture
def make_template():

    def make(**kwargs):
        meta = TEMPLATE_MAPPING['xing4_0']
        template = meta.template_cls(None, meta, **kwargs)
        # These tests exercise prompt serialization before tokenization; no model download is needed.
        template.model_meta = SimpleNamespace(is_multimodal=False)
        template.init_env_args()
        return template

    return make


def render(template, messages, **kwargs):
    inputs = StdTemplateInputs.from_dict({'messages': messages, **kwargs})
    template._swift_prepare_inputs(inputs)
    contexts, scales, _ = template._swift_encode(inputs)
    return ''.join(contexts), list(zip(contexts, scales))


def test_template_meta(make_template):
    meta = TEMPLATE_MAPPING['xing4_0']
    assert meta.prefix == [SYSTEM]
    assert meta.system_prefix == [f'{SYSTEM}{{{{SYSTEM}}}}']
    assert meta.prompt == [f'{USER}{{{{QUERY}}}}{BOT}']
    assert meta.chat_sep == [f'{EOS}\n']
    assert meta.suffix == [f'{EOS}\n']
    assert meta.agent_template == 'xing4_0'
    assert meta.preserve_thinking is False
    # The jinja template thinks unless `enable_thinking` is explicitly false, so the default is on.
    assert make_template().enable_thinking is True


@pytest.mark.parametrize('enable_thinking,tail', [(None, f'{THINK_OPEN}\n'), (True, f'{THINK_OPEN}\n'),
                                                  (False, THINK_CLOSE)])
def test_generation_prompt(make_template, enable_thinking, tail):
    template = make_template(enable_thinking=enable_thinking)
    text, _ = render(template, [{'role': 'user', 'content': '1+1=?'}])
    assert text == f'{SYSTEM}{USER}1+1=?{BOT}{tail}'


def test_system_prompt(make_template):
    template = make_template()
    text, _ = render(template, [{'role': 'system', 'content': 'SYS'}, {'role': 'user', 'content': '1+1=?'}])
    assert text == f'{SYSTEM}SYS{USER}1+1=?{BOT}{THINK_OPEN}\n'


def test_history_without_reasoning(make_template):
    template = make_template()
    text, _ = render(template, [{
        'role': 'user',
        'content': 'Q1'
    }, {
        'role': 'assistant',
        'content': 'A1'
    }, {
        'role': 'user',
        'content': 'Q2'
    }])
    assert text == f'{SYSTEM}{USER}Q1{BOT}{THINK_CLOSE}A1{EOS}\n{USER}Q2{BOT}{THINK_OPEN}\n'


@pytest.mark.parametrize('preserve_thinking,history', [
    (True, f'{THINK_OPEN}\nR1\n{THINK_CLOSE}A1'),
    (False, f'{THINK_CLOSE}A1'),
])
def test_history_reasoning(make_template, preserve_thinking, history):
    # The default (preserve_thinking=False) drops historical reasoning to match the official jinja;
    # preserve_thinking=True is an explicit opt-in that keeps it.
    template = make_template(preserve_thinking=preserve_thinking)
    text, _ = render(template, [{
        'role': 'user',
        'content': 'Q1'
    }, {
        'role': 'assistant',
        'content': f'{THINK_OPEN}\nR1\n{THINK_CLOSE}A1'
    }, {
        'role': 'user',
        'content': 'Q2'
    }])
    assert text == f'{SYSTEM}{USER}Q1{BOT}{history}{EOS}\n{USER}Q2{BOT}{THINK_OPEN}\n'


def test_train_mode_loss_scale(make_template):
    template = make_template()
    template.set_mode('train')
    text, contexts = render(template, [{'role': 'user', 'content': 'Q1'}, {'role': 'assistant', 'content': 'A1'}])
    assert text == f'{SYSTEM}{USER}Q1{BOT}{THINK_CLOSE}A1{EOS}\n'
    assert contexts == [(SYSTEM, 0.), (f'{USER}Q1{BOT}', 0.), (f'{THINK_CLOSE}A1', 1.), (f'{EOS}\n', 1.)]


def test_tools_prompt(make_template):
    template = make_template()
    text, _ = render(template, [{'role': 'user', 'content': 'beijing weather?'}], tools=TOOLS)
    assert text == f'{SYSTEM}{TOOLS_PROMPT}{USER}beijing weather?{BOT}{THINK_OPEN}\n'


def test_tool_call_turn(make_template):
    template = make_template()
    text, _ = render(
        template, [
            {
                'role': 'user',
                'content': 'Search'
            },
            {
                'role': 'tool_call',
                'content': {
                    'name': 'get_weather',
                    'arguments': {
                        'city': 'beijing',
                        'days': 3
                    }
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
    # The assistant turn is closed by its own EOS, then the observation opens a new `<_bot>` turn.
    assert text == (f'{SYSTEM}{TOOLS_PROMPT}{USER}Search{BOT}{THINK_CLOSE}{TOOL_CALL}{EOS}\n{OBSERVATION}'
                    f'{TOOL_RESPONSE_OPEN}Found{TOOL_RESPONSE_CLOSE}{BOT}{THINK_CLOSE}Done{EOS}\n'
                    f'{USER}Next{BOT}{THINK_OPEN}\n')


def test_multiple_tool_responses(make_template):
    template = make_template()
    text, _ = render(template, [
        {
            'role': 'user',
            'content': 'Q'
        },
        {
            'role': 'tool_call',
            'content': {
                'name': 'f',
                'arguments': {}
            }
        },
        {
            'role': 'tool_response',
            'content': 'R1'
        },
        {
            'role': 'tool_response',
            'content': 'R2'
        },
        {
            'role': 'assistant',
            'content': 'A'
        },
        {
            'role': 'user',
            'content': 'Q2'
        },
    ])
    assert (f'{EOS}\n{OBSERVATION}{TOOL_RESPONSE_OPEN}R1{TOOL_RESPONSE_CLOSE}'
            f'{TOOL_RESPONSE_OPEN}R2{TOOL_RESPONSE_CLOSE}{BOT}') in text


def test_standalone_tool_response(make_template):
    template = make_template()
    text, _ = render(template, [{
        'role': 'user',
        'content': 'Q'
    }, {
        'role': 'tool_response',
        'content': 'R1'
    }, {
        'role': 'assistant',
        'content': 'A'
    }, {
        'role': 'user',
        'content': 'Q2'
    }])
    # Without a preceding tool call the observation is appended to the user query.
    assert text == (f'{SYSTEM}{USER}Q{OBSERVATION}{TOOL_RESPONSE_OPEN}R1{TOOL_RESPONSE_CLOSE}{BOT}'
                    f'{THINK_CLOSE}A{EOS}\n{USER}Q2{BOT}{THINK_OPEN}\n')


def test_agent_template_tool_call_protocol():
    agent = agent_template_map['xing4_0']()
    assert agent._format_tools(TOOLS) == TOOLS_PROMPT
    assert agent._format_tools(TOOLS, 'Policy') == f'Policy{TOOLS_PROMPT}'

    messages = [{
        'role': 'tool_call',
        'content': {
            'name': 'get_weather',
            'arguments': {
                'city': 'beijing',
                'days': 3,
                'metric': True
            }
        }
    }]
    encoded = agent._format_tool_calls(messages)
    assert encoded == (f'{TOOL_CALL_OPEN}get_weather<param_key>city</param_key><param_value>beijing</param_value>'
                       '<param_key>days</param_key><param_value>3</param_value>'
                       f'<param_key>metric</param_key><param_value>true</param_value>{TOOL_CALL_CLOSE}')

    functions = agent.get_toolcall(encoded)
    assert [function.name for function in functions] == ['get_weather']
    # Values rendered by `tojson` are decoded back, so numbers and bools survive the round trip.
    assert [json.loads(function.arguments) for function in functions] == [{
        'city': 'beijing',
        'days': 3,
        'metric': True
    }]


def test_agent_template_react_fallback():
    agent = agent_template_map['xing4_0']()
    response = 'Thought: think\nAction: get_weather\nAction Input: {"city": "beijing"}'
    functions = agent.get_toolcall(response)
    assert [function.name for function in functions] == ['get_weather']
