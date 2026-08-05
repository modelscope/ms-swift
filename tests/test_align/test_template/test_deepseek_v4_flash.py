import os

os.environ['SWIFT_DEBUG'] = '1'

# DeepSeek-V4-Flash-0731 alignment tests.
#
# The expected strings below are the byte-exact output of the reference implementation
# shipped in the model repo (`encoding/encoding_dsv4.py`, `encoding/tests/test_output_*.txt`).
# Mapping between the two APIs:
#   swift `enable_thinking=True/False`      <-> official `thinking_mode='thinking'/'chat'`
#   swift `preserve_thinking=False`         <-> official `drop_thinking=True` (the default)
#   swift `REASONING_EFFORT` env var        <-> official `reasoning_effort=...`
#   swift `<think>R</think>C` in content    <-> official `reasoning_content=R, content=C`
#   swift `tool_call` / `tool_response`     <-> official assistant `tool_calls` / `tool` role
#
# Only the processor (tokenizer) is loaded, never the weights, so these tests do not
# download the ~163GB of shards.
#
# DeepSeek-V4-Flash is a text-only model: its config carries no vision/audio sub-config,
# `DeepseekV4ForCausalLM` has no multimodal tower, and the official encoder only accepts
# `text` / `tool_result` content blocks (anything else renders as `[Unsupported ...]`).
# There is therefore no multimodal path to align.

MODEL_ID = 'deepseek-ai/DeepSeek-V4-Flash-0731'

BOS = '<｜begin▁of▁sentence｜>'
EOS = '<｜end▁of▁sentence｜>'
USER = '<｜User｜>'
ASSISTANT = '<｜Assistant｜>'

SYSTEM = 'You are a helpful assistant.'

EFFORT_HIGH = (
    'Reasoning Effort: Absolute maximum with no shortcuts permitted.\n'
    'You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve '
    'the root cause, rigorously stress-testing your logic against all potential paths, edge cases, '
    'and adversarial scenarios.\n'
    'Explicitly write out your entire deliberation process, documenting every intermediate step, '
    'considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n')

EFFORT_MAX = ('Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n'
              'You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: '
              'exhaustively decompose the problem into its most fundamental components, trace every causal chain '
              'to its root, and resolve the underlying cause rather than any surface symptom.\n'
              'Do not stop reasoning until you have independently verified the solution from multiple angles and '
              'are certain that no assumption remains unchecked and no error remains undiscovered.\n\n')

TOOLS = [{
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get the weather for a specific location',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The city name'
                },
                'unit': {
                    'type': 'string',
                    'enum': ['celsius', 'fahrenheit'],
                    'description': 'Temperature unit'
                }
            },
            'required': ['location']
        }
    }
}, {
    'type': 'function',
    'function': {
        'name': 'search',
        'description': 'Search the web for information',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Search query'
                },
                'num_results': {
                    'type': 'integer',
                    'description': 'Number of results to return'
                }
            },
            'required': ['query']
        }
    }
}]

TOOL_SCHEMAS = ('{"name": "get_weather", "description": "Get the weather for a specific location", '
                '"parameters": {"type": "object", "properties": {"location": {"type": "string", '
                '"description": "The city name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], '
                '"description": "Temperature unit"}}, "required": ["location"]}}\n'
                '{"name": "search", "description": "Search the web for information", '
                '"parameters": {"type": "object", "properties": {"query": {"type": "string", '
                '"description": "Search query"}, "num_results": {"type": "integer", '
                '"description": "Number of results to return"}}, "required": ["query"]}}')

TOOLS_SECTION = ('## Tools\n\n'
                 'You have access to a set of tools to help answer the user\'s question. '
                 'You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:\n\n'
                 '<｜DSML｜tool_calls>\n'
                 '<｜DSML｜invoke name="$TOOL_NAME">\n'
                 '<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE'
                 '</｜DSML｜parameter>\n'
                 '...\n'
                 '</｜DSML｜invoke>\n'
                 '<｜DSML｜invoke name="$TOOL_NAME2">\n'
                 '...\n'
                 '</｜DSML｜invoke>\n'
                 '</｜DSML｜tool_calls>\n\n'
                 'String parameters should be specified as is and set `string="true"`. '
                 'For all other types (numbers, booleans, arrays, objects), '
                 'pass the value in JSON format and set `string="false"`.\n\n'
                 'If thinking_mode is enabled (triggered by <think>), '
                 'you MUST output your complete reasoning inside <think>...</think> '
                 'BEFORE any tool calls or final response.\n\n'
                 'Otherwise, output directly after </think> with tool calls or final response.\n\n'
                 '### Available Tool Schemas\n\n' + TOOL_SCHEMAS + '\n\n'
                 'You MUST strictly follow the above defined tool name and parameter schemas '
                 'to invoke tool calls.\n')


def _get_template(effort=None, enable_thinking=True):
    from swift.model import get_processor
    from swift.template import get_template
    processor = get_processor(MODEL_ID)
    # `get_env_args` reads the UPPER-CASE environment variable.
    if effort is None:
        os.environ.pop('REASONING_EFFORT', None)
    else:
        os.environ['REASONING_EFFORT'] = effort
    # swift defaults thinking templates that ship a `non_thinking_prefix` to chat mode,
    # so thinking has to be requested explicitly.
    template = get_template(processor, enable_thinking=enable_thinking)
    template.template_backend = 'swift'
    # Matches the official `drop_thinking=True` default.
    template.preserve_thinking = False
    return template


def _render(data, effort=None, enable_thinking=True, mode='train'):
    template = _get_template(effort, enable_thinking)
    template.set_mode(mode)
    encoded = template.encode(dict(data))
    return template.safe_decode(encoded['input_ids'])


def _read_golden(index: int) -> str:
    """The reference output shipped in the model repo, or None when it is not available.

    Only the tokenizer files are needed to run the rest of this module, so a checkout
    without the `encoding/` directory should skip the golden comparison rather than fail.
    """
    from swift.model import get_processor
    processor = get_processor(MODEL_ID)
    path = os.path.join(processor.model_info.model_dir, 'encoding', 'tests', f'test_output_{index}.txt')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        # The golden files end with a trailing newline that the encoder itself does not emit.
        return f.read().rstrip('\n')


THINKING_MESSAGES = [
    {
        'role': 'system',
        'content': SYSTEM
    },
    {
        'role': 'user',
        'content': 'Hello'
    },
    {
        'role': 'assistant',
        'content': '<think>The user said hello, I should greet back.</think>Hi there! How can I help you?'
    },
    {
        'role': 'user',
        'content': 'What is the capital of France?'
    },
    {
        'role':
        'assistant',
        'content':
        '<think>The user asks about the capital of France. It is Paris.</think>'
        'The capital of France is Paris.'
    },
]

# encoding/tests/test_output_2.txt: the reasoning of every turn before the last user
# message is dropped and replaced by a bare `</think>`.
THINKING_EXPECTED = (f'{BOS}{SYSTEM}'
                     f'{USER}Hello{ASSISTANT}</think>Hi there! How can I help you?{EOS}'
                     f'{USER}What is the capital of France?{ASSISTANT}'
                     '<think>The user asks about the capital of France. It is Paris.</think>'
                     f'The capital of France is Paris.{EOS}')


def test_deepseek_v4_flash_thinking():
    """`low` is the default level and contributes no prefix.

    Also pins the rendering against `encoding/tests/test_output_2.txt`.
    """
    assert _render({'messages': THINKING_MESSAGES}) == THINKING_EXPECTED
    assert _render({'messages': THINKING_MESSAGES}, effort='low') == THINKING_EXPECTED

    golden = _read_golden(2)
    if golden is not None:
        assert THINKING_EXPECTED == golden, 'diverged from encoding/tests/test_output_2.txt'


def test_deepseek_v4_flash_reasoning_effort():
    """The three levels, and the one-level-down shift relative to the preview release."""
    body = THINKING_EXPECTED[len(BOS):]
    assert _render({'messages': THINKING_MESSAGES}, effort='high') == BOS + EFFORT_HIGH + body
    assert _render({'messages': THINKING_MESSAGES}, effort='max') == BOS + EFFORT_MAX + body


def test_deepseek_v4_flash_generation_prompt():
    """The last user turn opens the thinking channel with `<think>`."""
    expected = (f'{BOS}{SYSTEM}'
                f'{USER}Hello{ASSISTANT}</think>Hi there! How can I help you?{EOS}'
                f'{USER}What is the capital of France?{ASSISTANT}<think>')
    assert _render({'messages': THINKING_MESSAGES[:-1]}, mode='transformers') == expected


CHAT_MESSAGES = [
    {
        'role': 'system',
        'content': SYSTEM
    },
    {
        'role': 'user',
        'content': 'Hello'
    },
    {
        'role': 'assistant',
        'content': 'Hi there! How can I help you?'
    },
    {
        'role': 'user',
        'content': 'What is the capital of France?'
    },
    {
        'role': 'assistant',
        'content': 'The capital of France is Paris.'
    },
]


def test_deepseek_v4_flash_chat_mode():
    """`thinking_mode='chat'`: `</think>` closes the channel right after `<｜Assistant｜>`."""
    expected = (f'{BOS}{SYSTEM}'
                f'{USER}Hello{ASSISTANT}</think>Hi there! How can I help you?{EOS}'
                f'{USER}What is the capital of France?{ASSISTANT}</think>The capital of France is Paris.{EOS}')
    assert _render({'messages': CHAT_MESSAGES}, enable_thinking=False) == expected

    expected_gen = (f'{BOS}{SYSTEM}'
                    f'{USER}Hello{ASSISTANT}</think>Hi there! How can I help you?{EOS}'
                    f'{USER}What is the capital of France?{ASSISTANT}</think>')
    assert _render({'messages': CHAT_MESSAGES[:-1]}, enable_thinking=False, mode='transformers') == expected_gen


TOOL_MESSAGES = [
    {
        'role': 'system',
        'content': SYSTEM
    },
    {
        'role': 'user',
        'content': "What's the weather in Beijing?"
    },
    {
        'role': 'assistant',
        'content': '<think>The user wants to know the weather in Beijing. I should use the get_weather tool.</think>'
    },
    {
        'role': 'tool_call',
        'content': '{"name": "get_weather", "arguments": "{\\"location\\": \\"Beijing\\", \\"unit\\": \\"celsius\\"}"}'
    },
    {
        'role': 'tool_response',
        'content': '{"temperature": 22, "condition": "sunny", "humidity": 45}'
    },
    {
        'role':
        'assistant',
        'content':
        '<think>Got the weather data. Let me format a nice response.</think>'
        'The weather in Beijing is currently sunny with a temperature of 22°C and 45% humidity.'
    },
]


def test_deepseek_v4_flash_tool_call():
    """encoding/tests/test_output_1.txt.

    The tools section is appended to the system message and the tool_calls block is
    separated from the assistant's textual content by a blank line.
    """
    expected = (
        f'{BOS}{SYSTEM}\n\n{TOOLS_SECTION}'
        f'{USER}What\'s the weather in Beijing?{ASSISTANT}'
        '<think>The user wants to know the weather in Beijing. I should use the get_weather tool.</think>\n\n'
        '<｜DSML｜tool_calls>\n'
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="unit" string="true">celsius</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        f'</｜DSML｜tool_calls>{EOS}'
        f'{USER}<tool_result>{{"temperature": 22, "condition": "sunny", "humidity": 45}}</tool_result>{ASSISTANT}'
        '<think>Got the weather data. Let me format a nice response.</think>'
        f'The weather in Beijing is currently sunny with a temperature of 22°C and 45% humidity.{EOS}')
    assert _render({'messages': TOOL_MESSAGES, 'tools': TOOLS}) == expected

    golden = _read_golden(1)
    if golden is not None:
        assert expected == golden, 'diverged from encoding/tests/test_output_1.txt'


def test_deepseek_v4_flash_tool_call_keeps_history_thinking():
    """Defining tools turns off `drop_thinking`, so *every* turn keeps its reasoning.

    The middle assistant turn is what matters here: it sits before the last user
    message, so without the tools exemption its `<think>...</think>` would collapse
    to a bare `</think>` the way it does in `test_deepseek_v4_flash_thinking`.
    """
    messages = [
        {
            'role': 'system',
            'content': SYSTEM
        },
        {
            'role': 'user',
            'content': "What's the weather in Beijing?"
        },
        {
            'role': 'assistant',
            'content': '<think>Need the get_weather tool.</think>'
        },
        {
            'role': 'tool_call',
            'content': '{"name": "get_weather", "arguments": "{\\"location\\": \\"Beijing\\"}"}'
        },
        {
            'role': 'tool_response',
            'content': '{"temperature": 22}'
        },
        {
            'role': 'assistant',
            'content': '<think>Got it, answer now.</think>It is 22°C in Beijing.'
        },
        {
            'role': 'user',
            'content': 'And Shanghai?'
        },
        {
            'role': 'assistant',
            'content': '<think>Same tool, new city.</think>It is 26°C in Shanghai.'
        },
    ]
    expected = (f'{BOS}{SYSTEM}\n\n{TOOLS_SECTION}'
                f'{USER}What\'s the weather in Beijing?{ASSISTANT}'
                '<think>Need the get_weather tool.</think>\n\n'
                '<｜DSML｜tool_calls>\n'
                '<｜DSML｜invoke name="get_weather">\n'
                '<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>\n'
                '</｜DSML｜invoke>\n'
                f'</｜DSML｜tool_calls>{EOS}'
                f'{USER}<tool_result>{{"temperature": 22}}</tool_result>{ASSISTANT}'
                f'<think>Got it, answer now.</think>It is 22°C in Beijing.{EOS}'
                f'{USER}And Shanghai?{ASSISTANT}'
                f'<think>Same tool, new city.</think>It is 26°C in Shanghai.{EOS}')
    assert _render({'messages': messages, 'tools': TOOLS}) == expected


def test_deepseek_v4_flash_parallel_tool_calls():
    """Several `<tool_result>` blocks in one user turn are joined by a blank line."""
    messages = [
        {
            'role': 'system',
            'content': SYSTEM
        },
        {
            'role': 'user',
            'content': 'weather in Beijing and Shanghai?'
        },
        {
            'role': 'assistant',
            'content': '<think>Two cities, call the tool twice.</think>Let me check both.'
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
            'content': '{"temperature": 22}'
        },
        {
            'role': 'tool_response',
            'content': '{"temperature": 26}'
        },
        {
            'role': 'assistant',
            'content': '<think>Both done.</think>Beijing 22°C, Shanghai 26°C.'
        },
    ]
    expected = (f'{BOS}{SYSTEM}\n\n{TOOLS_SECTION}'
                f'{USER}weather in Beijing and Shanghai?{ASSISTANT}'
                '<think>Two cities, call the tool twice.</think>Let me check both.\n\n'
                '<｜DSML｜tool_calls>\n'
                '<｜DSML｜invoke name="get_weather">\n'
                '<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>\n'
                '</｜DSML｜invoke>\n'
                '<｜DSML｜invoke name="get_weather">\n'
                '<｜DSML｜parameter name="location" string="true">Shanghai</｜DSML｜parameter>\n'
                '</｜DSML｜invoke>\n'
                f'</｜DSML｜tool_calls>{EOS}'
                f'{USER}<tool_result>{{"temperature": 22}}</tool_result>\n\n'
                f'<tool_result>{{"temperature": 26}}</tool_result>{ASSISTANT}'
                f'<think>Both done.</think>Beijing 22°C, Shanghai 26°C.{EOS}')
    assert _render({'messages': messages, 'tools': TOOLS}) == expected


def test_deepseek_v4_flash_tool_call_roundtrip():
    """The DSML tool call the template writes is parsed back into the same function."""
    import json

    from swift.agent_template import agent_template_map
    agent_template = agent_template_map['deepseek_v4']()
    response = ('<think>reason</think>\n\n'
                '<｜DSML｜tool_calls>\n'
                '<｜DSML｜invoke name="get_weather">\n'
                '<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>\n'
                '<｜DSML｜parameter name="num_results" string="false">3</｜DSML｜parameter>\n'
                '</｜DSML｜invoke>\n'
                '</｜DSML｜tool_calls>')
    functions = agent_template.get_toolcall(response)
    assert len(functions) == 1
    assert functions[0].name == 'get_weather'
    assert json.loads(functions[0].arguments) == {'location': 'Beijing', 'num_results': 3}


def test_deepseek_v4_flash_is_text_only():
    """Guards the "no multimodal path" assumption the tests above rely on."""
    from transformers import AutoConfig

    from swift.model import get_model_info_meta
    model_info, model_meta = get_model_info_meta(MODEL_ID)
    # 0731 shares the `deepseek_v4` model_type and only overrides the template via its ModelGroup.
    assert model_meta.model_type == 'deepseek_v4'
    assert model_meta.template == 'deepseek_v4_flash'
    assert not model_meta.is_multimodal
    assert not model_info.is_multimodal

    config = AutoConfig.from_pretrained(model_info.model_dir)
    assert config.model_type == 'deepseek_v4'
    for key in ['vision_config', 'audio_config', 'visual', 'image_token_id']:
        assert getattr(config, key, None) is None, key


if __name__ == '__main__':
    test_deepseek_v4_flash_thinking()
    test_deepseek_v4_flash_reasoning_effort()
    test_deepseek_v4_flash_generation_prompt()
    test_deepseek_v4_flash_chat_mode()
    test_deepseek_v4_flash_tool_call()
    test_deepseek_v4_flash_tool_call_keeps_history_thinking()
    test_deepseek_v4_flash_parallel_tool_calls()
    test_deepseek_v4_flash_tool_call_roundtrip()
    test_deepseek_v4_flash_is_text_only()
    print('all passed')
