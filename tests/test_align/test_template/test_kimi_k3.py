import os

os.environ['SWIFT_DEBUG'] = '1'

# NOTE: All tests here only load the processor (tokenizer + vision processor) and the
# remote-code python files via `get_processor` (i.e. `load_model=False`), so *.safetensors
# / *.bin weight shards are NOT downloaded (see `safe_snapshot_download`, which appends
# `['*.bin', '*.safetensors']` to `ignore_patterns` when `download_model=False`). The
# KimiK3 template `encode`/official processor `__call__` paths are pure tokenization +
# image preprocessing and never run a model forward, so no weights are required.

MODEL_ID = 'moonshotai/Kimi-K3'
MEDIA_PAD = '<|media_pad|>'


def _get_template():
    from swift.model import get_processor
    from swift.template import get_template
    processor = get_processor(MODEL_ID)
    template = get_template(processor)
    return template, processor


def _to_id_list(input_ids):
    if isinstance(input_ids, list):
        return input_ids[0] if input_ids and isinstance(input_ids[0], list) else input_ids
    # torch.Tensor
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    return input_ids.tolist()


def _collapse_media_pad(ids, media_pad_id):
    """Collapse consecutive media_pad runs to a single token.

    swift pre-expands `<|media_pad|>` to the full grid length, whereas the official
    processor keeps one placeholder per image (the model expands it inside forward).
    Collapsing both lets us compare the surrounding XTML structure exactly.
    """
    out = []
    for x in ids:
        if x == media_pad_id and out and out[-1] == media_pad_id:
            continue
        out.append(x)
    return out


def _official_encode(processor, messages, *, add_generation_prompt):
    """The official Kimi-K3 'readme' path: KimiK3Processor.__call__.

    It extracts medias from the (content-parts) messages, preprocesses images, and
    renders the XTML chat via the tokenizer's python encoder.
    """
    batch = processor(
        messages=messages,
        return_tensors='pt',
        add_generation_prompt=add_generation_prompt,
        thinking=True,
        thinking_effort=None,  # suppress the default thinking-effort system message
    )
    return batch


def test_kimi_k3_multimodal_encode_align():
    import torch
    from PIL import Image
    template, processor = _get_template()
    template.set_mode('train')
    tokenizer = template.tokenizer
    media_pad_id = tokenizer.convert_tokens_to_ids(MEDIA_PAD)

    image = Image.new('RGB', (640, 480), (12, 34, 56))

    # swift-style messages (inline <image> tag + inline <think> convention)
    swift_inputs = {
        'messages': [
            {
                'role': 'user',
                'content': '<image>What is in this image?'
            },
            {
                'role': 'assistant',
                'content': '<think>a solid color block</think>A solid color block.'
            },
        ],
        'images': [image],
    }
    encoded = template.encode(swift_inputs)
    swift_ids = _to_id_list(encoded['input_ids'])

    # official-style messages (content parts, reasoning_content channel)
    official_messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'image': image
                },
                {
                    'type': 'text',
                    'text': 'What is in this image?'
                },
            ]
        },
        {
            'role': 'assistant',
            'reasoning_content': 'a solid color block',
            'content': 'A solid color block.'
        },
    ]
    batch = _official_encode(processor, official_messages, add_generation_prompt=False)
    official_ids = _to_id_list(batch['input_ids'])

    # 1. Surrounding XTML structure must match (collapse the media_pad expansion).
    assert _collapse_media_pad(swift_ids, media_pad_id) == _collapse_media_pad(official_ids, media_pad_id), \
        (f'text structure mismatch\n swift   : {tokenizer.decode(swift_ids)}\n'
         f' official: {tokenizer.decode(official_ids)}')

    # 2. swift pre-expands media_pad to exactly the grid length reported by the processor.
    num_pad = sum(1 for x in swift_ids if x == media_pad_id)
    expected = processor.image_processor.media_tokens_calculator({'type': 'image', 'image': image})
    assert num_pad == expected, f'media_pad count {num_pad} != media_tokens_calculator {expected}'

    # 3. Image preprocessing tensors must be identical.
    assert torch.equal(encoded['grid_thws'], batch['grid_thws']), \
        f"grid_thws mismatch: {encoded['grid_thws'].tolist()} vs {batch['grid_thws'].tolist()}"
    assert torch.allclose(encoded['pixel_values'], batch['pixel_values']), 'pixel_values mismatch'
    print(f'[multimodal] pass: {num_pad} media_pad, grid_thws={encoded["grid_thws"].tolist()}, '
          f'pixel_values={tuple(encoded["pixel_values"].shape)}')


def test_kimi_k3_text_infer_align():
    template, processor = _get_template()
    tokenizer = template.tokenizer

    swift_inputs = {'messages': [{'role': 'user', 'content': 'Tell me three random numbers.'}]}
    encoded = template.encode(swift_inputs)
    swift_ids = _to_id_list(encoded['input_ids'])

    official_ids = tokenizer.apply_chat_template([{
        'role': 'user',
        'content': 'Tell me three random numbers.'
    }],
                                                 tokenize=True,
                                                 add_generation_prompt=True,
                                                 thinking=True,
                                                 thinking_effort=None)
    assert swift_ids == official_ids, \
        (f'infer mismatch\n swift   : {tokenizer.decode(swift_ids)}\n'
         f' official: {tokenizer.decode(official_ids)}')
    print(f'[text-infer] pass: {len(swift_ids)} tokens')


def test_kimi_k3_text_multiturn_align():
    template, processor = _get_template()
    tokenizer = template.tokenizer

    swift_inputs = {
        'messages': [
            {
                'role': 'user',
                'content': 'Tell me three random numbers.'
            },
            {
                'role': 'assistant',
                'content': '<think>473, 921, 235, 215, 222.</think>473, 921, 235'
            },
            {
                'role': 'user',
                'content': 'What are the other two?'
            },
        ]
    }
    encoded = template.encode(swift_inputs)
    swift_ids = _to_id_list(encoded['input_ids'])

    official_ids = tokenizer.apply_chat_template([
        {
            'role': 'user',
            'content': 'Tell me three random numbers.'
        },
        {
            'role': 'assistant',
            'reasoning_content': '473, 921, 235, 215, 222.',
            'content': '473, 921, 235'
        },
        {
            'role': 'user',
            'content': 'What are the other two?'
        },
    ],
                                                 tokenize=True,
                                                 add_generation_prompt=True,
                                                 thinking=True,
                                                 thinking_effort=None)
    assert swift_ids == official_ids, \
        (f'multi-turn mismatch\n swift   : {tokenizer.decode(swift_ids)}\n'
         f' official: {tokenizer.decode(official_ids)}')
    print(f'[text-multiturn] pass: {len(swift_ids)} tokens')


def test_kimi_k3_thinking_effort_align():
    template, processor = _get_template()
    tokenizer = template.tokenizer

    swift_inputs = {
        'messages': [{
            'role': 'user',
            'content': 'Prove that sqrt(2) is irrational.'
        }],
        'chat_template_kwargs': {
            'thinking_effort': 'high'
        },
    }
    encoded = template.encode(swift_inputs)
    swift_ids = _to_id_list(encoded['input_ids'])

    official_ids = tokenizer.apply_chat_template([{
        'role': 'user',
        'content': 'Prove that sqrt(2) is irrational.'
    }],
                                                 tokenize=True,
                                                 add_generation_prompt=True,
                                                 thinking=True,
                                                 thinking_effort='high')
    assert swift_ids == official_ids, \
        (f'thinking_effort mismatch\n swift   : {tokenizer.decode(swift_ids)}\n'
         f' official: {tokenizer.decode(official_ids)}')

    # `reasoning_effort` (the K3 API field name) is accepted as an alias.
    swift_inputs2 = {
        'messages': [{
            'role': 'user',
            'content': 'Prove that sqrt(2) is irrational.'
        }],
        'chat_template_kwargs': {
            'reasoning_effort': 'high'
        },
    }
    encoded2 = template.encode(swift_inputs2)
    assert _to_id_list(encoded2['input_ids']) == official_ids
    print(f'[thinking-effort] pass: {len(swift_ids)} tokens')


def test_kimi_k3_tool_call_align():
    template, processor = _get_template()
    tokenizer = template.tokenizer

    tools = [{
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get the weather for a city',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {
                        'type': 'string'
                    }
                },
                'required': ['city'],
            },
        },
    }]
    swift_messages = [
        {
            'role': 'user',
            'content': 'Weather in Beijing and Shanghai?'
        },
        {
            'role': 'assistant',
            'content': '<think>Need two calls.</think>I will check both cities.'
        },
        {
            'role': 'tool_call',
            'content': '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        },
        {
            'role': 'tool_call',
            'content': '{"name": "get_weather", "arguments": {"city": "Shanghai"}}'
        },
        {
            'role': 'tool',
            'content': 'Beijing: sunny'
        },
        {
            'role': 'tool',
            'content': 'Shanghai: rain'
        },
        {
            'role': 'assistant',
            'content': '<think>Summarize.</think>Beijing sunny, Shanghai rainy.'
        },
    ]
    official_messages = [
        {
            'role': 'user',
            'content': 'Weather in Beijing and Shanghai?'
        },
        {
            'role':
            'assistant',
            'reasoning_content':
            'Need two calls.',
            'content':
            'I will check both cities.',
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {
                        'name': 'get_weather',
                        'arguments': '{"city": "Beijing"}'
                    }
                },
                {
                    'id': 'call_2',
                    'type': 'function',
                    'function': {
                        'name': 'get_weather',
                        'arguments': '{"city": "Shanghai"}'
                    }
                },
            ]
        },
        {
            'role': 'tool',
            'tool_call_id': 'call_1',
            'content': 'Beijing: sunny'
        },
        {
            'role': 'tool',
            'tool_call_id': 'call_2',
            'content': 'Shanghai: rain'
        },
        {
            'role': 'assistant',
            'reasoning_content': 'Summarize.',
            'content': 'Beijing sunny, Shanghai rainy.'
        },
    ]

    def _official(messages, add_generation_prompt, **kwargs):
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            thinking=True,
            thinking_effort=None,
            **kwargs)

    # 1. tool declare + generation prompt
    encoded = template.encode({'messages': swift_messages[:1], 'tools': tools})
    swift_ids = _to_id_list(encoded['input_ids'])
    official_ids = _official(official_messages[:1], True)
    assert swift_ids == official_ids, \
        (f'tool declare mismatch\n swift   : {tokenizer.decode(swift_ids)}\n'
         f' official: {tokenizer.decode(official_ids)}')

    # 2. generation prompt right after tool results
    encoded = template.encode({'messages': swift_messages[:6], 'tools': tools})
    swift_ids = _to_id_list(encoded['input_ids'])
    official_ids = _official(official_messages[:4], True)
    assert swift_ids == official_ids, \
        (f'tool result mismatch\n swift   : {tokenizer.decode(swift_ids)}\n'
         f' official: {tokenizer.decode(official_ids)}')

    # 3. full agent loop (training render, labels aligned)
    template.set_mode('train')
    encoded = template.encode({'messages': swift_messages, 'tools': tools})
    template.set_mode('transformers')
    swift_ids = _to_id_list(encoded['input_ids'])
    official_ids = _official(official_messages, False)
    assert swift_ids == official_ids, \
        (f'agent loop mismatch\n swift   : {tokenizer.decode(swift_ids)}\n'
         f' official: {tokenizer.decode(official_ids)}')
    # labels: only assistant output (think/response/tool-calls) is supervised;
    # tool-declare, user prompt and tool results are masked out.
    supervised = tokenizer.decode([t for t, m in zip(swift_ids, encoded['labels']) if m != -100])
    for text in ('Need two calls.', 'I will check both cities.', 'call tool="get_weather"',
                 'Shanghai<|close|>argument<|sep|>', 'Summarize.', 'Beijing sunny, Shanghai rainy.'):
        assert text in supervised, f'missing supervised text: {text!r}\n supervised: {supervised}'
    for text in ('Weather in Beijing and Shanghai?', 'Beijing: sunny', 'Shanghai: rain', 'tool-declare'):
        assert text not in supervised, f'unexpected supervised text: {text!r}\n supervised: {supervised}'

    # 4. thinking_effort is rendered after the tool-declare message (official order)
    encoded = template.encode({
        'messages': swift_messages[:1],
        'tools': tools,
        'chat_template_kwargs': {
            'thinking_effort': 'low'
        },
    })
    swift_ids2 = _to_id_list(encoded['input_ids'])
    official_ids2 = tokenizer.apply_chat_template(
        official_messages[:1],
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        thinking=True,
        thinking_effort='low')
    assert swift_ids2 == official_ids2, \
        (f'tools+thinking_effort mismatch\n swift   : {tokenizer.decode(swift_ids2)}\n'
         f' official: {tokenizer.decode(official_ids2)}')

    # 5. get_toolcall parses the rendered XTML tool section back
    import json
    functions = template.agent_template.get_toolcall(
        '<|open|>think<|sep|>t<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|>'
        '<|open|>tools<|sep|><|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>argument key="city" type="string"<|sep|>Beijing<|close|>argument<|sep|>'
        '<|close|>call<|sep|><|close|>tools<|sep|>')
    assert len(functions) == 1 and functions[0].name == 'get_weather' \
        and json.loads(functions[0].arguments) == {'city': 'Beijing'}, f'functions: {functions}'
    print(f'[tool-call] pass: {len(swift_ids)} tokens (agent loop)')


def test_kimi_k3_decode_thinking():
    # Inference (decode) scenario: `decode_generate_ids` maps the generated XTML
    # think/response channels back to swift's inline `<think>...</think>` convention,
    # and `_thinking_to_xtml` (encode side) inverts it for multi-turn re-feeding.
    template, _ = _get_template()

    xtml = ('<|open|>think<|sep|>REASON<|close|>think<|sep|>'
            '<|open|>response<|sep|>ANSWER<|close|>response<|sep|>')
    inline = template._xtml_to_thinking(xtml)
    assert inline == '<think>REASON</think>ANSWER', f'inline: {inline!r}'
    # round trip: re-encoding the inline form restores the XTML channels
    assert template._thinking_to_xtml(inline) == xtml

    # streaming: an unfinished generation only has the think channel open
    assert template._xtml_to_thinking('<|open|>think<|sep|>partial') == '<think>partial'

    # a tool-calls section is preserved verbatim for get_toolcall
    inline = template._xtml_to_thinking(
        '<|open|>think<|sep|>t<|close|>think<|sep|><|open|>response<|sep|>Call it.<|close|>response<|sep|>'
        '<|open|>tools<|sep|><|open|>call tool="f" index="0"<|sep|><|close|>call<|sep|><|close|>tools<|sep|>')
    assert inline == ('<think>t</think>Call it.'
                      '<|open|>tools<|sep|><|open|>call tool="f" index="0"<|sep|>'
                      '<|close|>call<|sep|><|close|>tools<|sep|>'), f'inline: {inline!r}'
    print('[decode-thinking] pass')


if __name__ == '__main__':
    test_kimi_k3_multimodal_encode_align()
    test_kimi_k3_text_infer_align()
    test_kimi_k3_text_multiturn_align()
    test_kimi_k3_thinking_effort_align()
    test_kimi_k3_tool_call_align()
    test_kimi_k3_decode_thinking()
