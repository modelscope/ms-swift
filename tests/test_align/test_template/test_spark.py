import os

os.environ['SWIFT_DEBUG'] = '1'

# Spark-X2.5 alignment tests.
#
# Every expectation below is checked against the rendering of the model's own
# `chat_template.jinja`, either by running swift's `jinja` backend on the same data or, for the
# channels swift expresses differently (inline `<think>` instead of `reasoning_content`,
# `tool_call`/`tool_response` roles instead of `tool_calls`/`tool`), by calling
# `apply_chat_template` directly with the equivalent native-style messages.
#
# Only the processor is loaded (`get_processor` -> `load_model=False`), so the ~8GB of weight
# shards are never downloaded: rendering is pure tokenization.

MODEL_ID = 'XHToken/Spark-X2.5-4B'

BOS = '<｜start▁of▁sentence｜>'
EOS = '<｜end▁of▁sentence｜>'
SYSTEM_BLOCK = f'{BOS}<|System|>\nyou are a helpful assistant.'
USER = f'{BOS}<|User|>'
BOT = f'{BOS}<|Bot|>'
TOOL = f'{BOS}<|Tool|>'

SYSTEM = 'You are a helpful assistant.'

TOOLS = [{
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get the weather for a location',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The city name'
                },
                'days': {
                    'type': 'integer',
                    'description': 'Number of days'
                }
            },
            'required': ['location']
        }
    }
}]

TOOLS_SECTION = ('## Tools\nYou have access to the following functions:\n<tools>\n'
                 '{"name": "get_weather", "description": "Get the weather for a location", '
                 '"parameters": {"type": "object", "properties": {"location": {"type": "string", '
                 '"description": "The city name"}, "days": {"type": "integer", '
                 '"description": "Number of days"}}, "required": ["location"]}}\n</tools>')


def _get_template(enable_thinking=None, preserve_thinking=None):
    from swift.model import get_processor
    from swift.template import get_template
    processor = get_processor(MODEL_ID)
    # Spark ships a `non_thinking_prefix`, so swift defaults to chat mode; thinking has to be
    # requested explicitly, exactly like glm4_5 / deepseek_v4.
    return get_template(processor, enable_thinking=enable_thinking, preserve_thinking=preserve_thinking)


def _render(data, mode='train', backend='swift', enable_thinking=None, preserve_thinking=None):
    template = _get_template(enable_thinking, preserve_thinking)
    template.template_backend = backend
    template.set_mode(mode)
    encoded = template.encode(dict(data))
    return template.safe_decode(encoded['input_ids'])


def _render_native(messages, *, add_generation_prompt=False, **kwargs):
    """Render native-style messages with the model's own jinja, bypassing swift."""
    template = _get_template()
    return template.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs)


def _assert_matches_jinja(data, mode='train', enable_thinking=None, preserve_thinking=None):
    """The swift backend and the jinja backend must agree byte for byte."""
    swift_text = _render(data, mode, 'swift', enable_thinking, preserve_thinking)
    jinja_text = _render(data, mode, 'jinja', enable_thinking, preserve_thinking)
    assert swift_text == jinja_text, f'\n swift: {swift_text!r}\n jinja: {jinja_text!r}'
    return swift_text


CHAT_MESSAGES = [
    {
        'role': 'user',
        'content': 'hello'
    },
    {
        'role': 'assistant',
        'content': 'hi'
    },
    {
        'role': 'user',
        'content': 'bye'
    },
    {
        'role': 'assistant',
        'content': 'see you'
    },
]


def test_spark_chat():
    """The builtin system block is always emitted and every reply opens with a bare `</think>`."""
    expected = (f'{SYSTEM_BLOCK}{EOS}'
                f'{USER}hello{EOS}{BOT}</think>hi{EOS}'
                f'{USER}bye{EOS}{BOT}</think>see you{EOS}')
    assert _assert_matches_jinja({'messages': CHAT_MESSAGES}, enable_thinking=False) == expected


def test_spark_system():
    """A user system does not replace the builtin one, it is appended after a blank line."""
    expected = (f'{SYSTEM_BLOCK}\n\n{SYSTEM}{EOS}'
                f'{USER}hello{EOS}{BOT}</think>hi{EOS}'
                f'{USER}bye{EOS}{BOT}</think>see you{EOS}')
    data = {'messages': [{'role': 'system', 'content': SYSTEM}] + CHAT_MESSAGES}
    assert _assert_matches_jinja(data, enable_thinking=False) == expected


def test_spark_generation_prompt():
    """`<think>` opens the reasoning channel, `</think>` closes it right away."""
    data = {'messages': CHAT_MESSAGES[:1]}
    prompt = f'{SYSTEM_BLOCK}{EOS}{USER}hello{EOS}{BOT}'
    assert _assert_matches_jinja(data, mode='transformers', enable_thinking=True) == f'{prompt}<think>'
    assert _assert_matches_jinja(data, mode='transformers', enable_thinking=False) == f'{prompt}</think>'


THINKING_MESSAGES = [
    {
        'role': 'user',
        'content': 'hello'
    },
    {
        'role': 'assistant',
        'content': '<think>greet back</think>hi'
    },
    {
        'role': 'user',
        'content': 'bye'
    },
    {
        'role': 'assistant',
        'content': '<think>say farewell</think>see you'
    },
]

# The native equivalent: swift keeps the reasoning inline in `content`, the jinja template reads
# it from the `reasoning_content` channel.
NATIVE_THINKING_MESSAGES = [
    {
        'role': 'user',
        'content': 'hello'
    },
    {
        'role': 'assistant',
        'reasoning_content': 'greet back',
        'content': 'hi'
    },
    {
        'role': 'user',
        'content': 'bye'
    },
    {
        'role': 'assistant',
        'reasoning_content': 'say farewell',
        'content': 'see you'
    },
]


def test_spark_thinking_preserved():
    """`preserve_thinking=True` keeps every `<think>` block, like passing `reasoning_content`."""
    rendered = _render({'messages': THINKING_MESSAGES}, enable_thinking=True, preserve_thinking=True)
    assert rendered == _render_native(NATIVE_THINKING_MESSAGES)
    assert rendered == (f'{SYSTEM_BLOCK}{EOS}'
                        f'{USER}hello{EOS}{BOT}<think>greet back</think>hi{EOS}'
                        f'{USER}bye{EOS}{BOT}<think>say farewell</think>see you{EOS}')


def test_spark_thinking_dropped():
    """`preserve_thinking=False` collapses history reasoning to the bare `</think>`.

    Which is exactly what the native template renders for a turn without `reasoning_content`.
    """
    rendered = _render({'messages': THINKING_MESSAGES}, enable_thinking=True, preserve_thinking=False)
    native = _render_native([
        NATIVE_THINKING_MESSAGES[0],
        {
            'role': 'assistant',
            'content': 'hi'
        },
        NATIVE_THINKING_MESSAGES[2],
        NATIVE_THINKING_MESSAGES[3],
    ])
    assert rendered == native
    assert rendered == (f'{SYSTEM_BLOCK}{EOS}'
                        f'{USER}hello{EOS}{BOT}</think>hi{EOS}'
                        f'{USER}bye{EOS}{BOT}<think>say farewell</think>see you{EOS}')


def test_spark_tools():
    """The tools block is glued to the builtin system; a user system still follows it."""
    data = {'messages': CHAT_MESSAGES, 'tools': TOOLS}
    assert _assert_matches_jinja(
        data, enable_thinking=False) == (f'{SYSTEM_BLOCK}{TOOLS_SECTION}{EOS}'
                                         f'{USER}hello{EOS}{BOT}</think>hi{EOS}'
                                         f'{USER}bye{EOS}{BOT}</think>see you{EOS}')

    data = {'messages': [{'role': 'system', 'content': SYSTEM}] + CHAT_MESSAGES, 'tools': TOOLS}
    assert _assert_matches_jinja(
        data, enable_thinking=False) == (f'{SYSTEM_BLOCK}{TOOLS_SECTION}\n\n{SYSTEM}{EOS}'
                                         f'{USER}hello{EOS}{BOT}</think>hi{EOS}'
                                         f'{USER}bye{EOS}{BOT}</think>see you{EOS}')


TOOL_CALL_MESSAGES = [
    {
        'role': 'user',
        'content': "What's the weather in Beijing?"
    },
    {
        'role': 'assistant',
        'content': 'let me check'
    },
    {
        'role': 'tool_call',
        'content': '{"name": "get_weather", "arguments": "{\\"location\\": \\"Beijing\\", \\"days\\": 3}"}'
    },
    {
        'role': 'tool_response',
        'content': '{"temp": 22}'
    },
    {
        'role': 'assistant',
        'content': 'It is 22°C in Beijing.'
    },
]

NATIVE_TOOL_CALL_MESSAGES = [
    {
        'role': 'user',
        'content': "What's the weather in Beijing?"
    },
    {
        'role':
        'assistant',
        'content':
        'let me check',
        'tool_calls': [{
            'type': 'function',
            'function': {
                'name': 'get_weather',
                'arguments': {
                    'location': 'Beijing',
                    'days': 3
                }
            }
        }]
    },
    {
        'role': 'tool',
        'content': '{"temp": 22}'
    },
    {
        'role': 'assistant',
        'content': 'It is 22°C in Beijing.'
    },
]


def test_spark_tool_call():
    """Tool calls stay inside the assistant turn; observations get their own `<|Tool|>` turn.

    Note the non-string argument: the jinja template json-encodes it, so it renders as `3`.
    """
    rendered = _render({'messages': TOOL_CALL_MESSAGES, 'tools': TOOLS}, enable_thinking=False)
    assert rendered == _render_native(NATIVE_TOOL_CALL_MESSAGES, tools=TOOLS, enable_thinking=False)
    assert rendered == (f'{SYSTEM_BLOCK}{TOOLS_SECTION}{EOS}'
                        f"{USER}What's the weather in Beijing?{EOS}"
                        f'{BOT}</think>let me check<tool_call>get_weather'
                        '<arg_key>location</arg_key><arg_value>Beijing</arg_value>'
                        f'<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>{EOS}'
                        f'{TOOL}<tool_response>{{"temp": 22}}</tool_response>{EOS}'
                        f'{BOT}</think>It is 22°C in Beijing.{EOS}')


def test_spark_parallel_tool_calls():
    """Several calls / observations are concatenated without a separator."""
    messages = [
        {
            'role': 'user',
            'content': 'weather?'
        },
        {
            'role': 'assistant',
            'content': ''
        },
        {
            'role': 'tool_call',
            'content': '{"name": "get_weather", "arguments": "{\\"location\\": \\"Beijing\\"}"}'
        },
        {
            'role': 'tool_call',
            'content': '{"name": "get_weather", "arguments": "{\\"location\\": \\"Shanghai\\"}"}'
        },
        {
            'role': 'tool_response',
            'content': '{"temp": 22}'
        },
        {
            'role': 'tool_response',
            'content': '{"temp": 26}'
        },
        {
            'role': 'assistant',
            'content': 'Beijing 22°C, Shanghai 26°C.'
        },
    ]
    native_messages = [
        {
            'role': 'user',
            'content': 'weather?'
        },
        {
            'role':
            'assistant',
            'content':
            '',
            'tool_calls': [{
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'arguments': {
                        'location': 'Beijing'
                    }
                }
            }, {
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'arguments': {
                        'location': 'Shanghai'
                    }
                }
            }]
        },
        {
            'role': 'tool',
            'content': '{"temp": 22}'
        },
        {
            'role': 'tool',
            'content': '{"temp": 26}'
        },
        {
            'role': 'assistant',
            'content': 'Beijing 22°C, Shanghai 26°C.'
        },
    ]
    rendered = _render({'messages': messages, 'tools': TOOLS}, enable_thinking=False)
    assert rendered == _render_native(native_messages, tools=TOOLS, enable_thinking=False)


def test_spark_standalone_tool_response():
    """An observation without a preceding tool call is still its own `<|Tool|>` turn."""
    messages = [
        {
            'role': 'user',
            'content': 'q'
        },
        {
            'role': 'tool',
            'content': 'env info'
        },
        {
            'role': 'assistant',
            'content': 'a'
        },
    ]
    rendered = _render({'messages': messages}, enable_thinking=False)
    assert rendered == _render_native(messages, enable_thinking=False)
    assert rendered == (f'{SYSTEM_BLOCK}{EOS}{USER}q{EOS}'
                        f'{TOOL}<tool_response>env info</tool_response>{EOS}'
                        f'{BOT}</think>a{EOS}')


def test_spark_tool_call_roundtrip():
    """The tool call the template writes is parsed back into the same functions."""
    import json

    from swift.agent_template import agent_template_map
    agent_template = agent_template_map['spark2_5']()
    response = ('<tool_call>get_weather<arg_key>location</arg_key><arg_value>Beijing</arg_value>'
                '</tool_call><tool_call>get_weather<arg_key>location</arg_key>'
                '<arg_value>Shanghai</arg_value></tool_call>')
    functions = agent_template.get_toolcall(response)
    assert [f.name for f in functions] == ['get_weather', 'get_weather']
    assert json.loads(functions[0].arguments) == {'location': 'Beijing'}
    assert json.loads(functions[1].arguments) == {'location': 'Shanghai'}


def test_spark_labels():
    """Only the assistant turns (with their EOS) contribute to the loss."""
    template = _get_template(enable_thinking=False)
    template.set_mode('train')
    encoded = template.encode({'messages': CHAT_MESSAGES})
    labels = template.safe_decode([token for token in encoded['labels'] if token != -100])
    assert labels == f'</think>hi{EOS}</think>see you{EOS}'


def test_spark_model_meta():
    """model_type inference, the text-only assumption, and the eager-attention override."""
    from transformers import AutoConfig

    from swift.model import get_model_info_meta
    model_info, model_meta = get_model_info_meta(MODEL_ID)
    assert model_meta.model_type == 'spark2_5'
    assert model_meta.template == 'spark2_5'
    assert not model_meta.is_multimodal and not model_info.is_multimodal

    config = AutoConfig.from_pretrained(model_info.model_dir, trust_remote_code=True)
    assert config.model_type == 'spark2_5'
    assert config.architectures == ['Spark2_5ForCausalLM']
    # 3 sliding_attention layers per full_attention layer, each with its own rope settings.
    assert config.layer_types[:4] == ['sliding_attention'] * 3 + ['full_attention']
    assert config.get_partial_rotary_factor('full_attention') == 0.25

    # modeling_spark.py implements eager attention only, so the loader pins it regardless of
    # what the user asked for.
    loader = model_meta.loader(model_info, model_meta, attn_impl='flash_attn', model_kwargs={})
    config = loader.get_config(model_info.model_dir)
    loader._postprocess_config(config)
    assert loader.attn_impl == 'eager'
    assert config._attn_implementation == 'eager'


if __name__ == '__main__':
    test_spark_chat()
    test_spark_system()
    test_spark_generation_prompt()
    test_spark_thinking_preserved()
    test_spark_thinking_dropped()
    test_spark_tools()
    test_spark_tool_call()
    test_spark_parallel_tool_calls()
    test_spark_standalone_tool_response()
    test_spark_tool_call_roundtrip()
    test_spark_labels()
    test_spark_model_meta()
    print('all passed')
