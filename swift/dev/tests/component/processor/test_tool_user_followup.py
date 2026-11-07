# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
from types import SimpleNamespace

import json
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from swift.dev.template.agent import agent_template_map
from swift.dev.template.register import get_template as _get_template


def get_template(processor, template_type, agent_template=None, **kwargs):
    return _get_template(
        processor,
        template_type=template_type,
        agent_template=agent_template,
        media_types=frozenset(),
        **kwargs,
    )


def _make_tokenizer(glm=False):
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    special_tokens = ['<|im_start|>', '<think>', '</think>']
    if glm:
        special_tokens += ['[gMASK]', '<sop>', '<|system|>', '<|user|>', '<|assistant|>', '<|observation|>']
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token='<|endoftext|>' if glm else '<|im_end|>',
        pad_token='<|endoftext|>',
        additional_special_tokens=special_tokens,
    )
    tokenizer.model_info = SimpleNamespace(
        config=SimpleNamespace(), task_type='causal_lm', max_model_len=8192, model_dir='', model_type=None)
    tokenizer.model_meta = SimpleNamespace(is_multimodal=False, model_type=None)
    return tokenizer


def _supervised(tokenizer, encoded):
    return tokenizer.decode([token for token in encoded['labels'] if token != -100])


def _qwen_data(n_tools=1, n_users=1):
    call = '<think>\nplan\n</think>\n\n<tool_call>\n<function=weather>\n</function>\n</tool_call>'
    answer = '<think>\nreply\n</think>\n\nfinal_answer'
    messages = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': call}]
    messages += [{'role': 'tool', 'content': f'result_{i}'} for i in range(n_tools)]
    messages += [{'role': 'user', 'content': f'followup_{i}'} for i in range(n_users)]
    messages.append({'role': 'assistant', 'content': answer})
    return {'messages': messages}


@pytest.mark.parametrize('strategy', ['default', 'last_round'])
@pytest.mark.parametrize('n_tools,n_users', [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2)])
def test_qwen_followup_boundaries_and_labels(strategy, n_tools, n_users):
    tokenizer = _make_tokenizer()
    template = get_template(
        tokenizer,
        template_type='qwen3_5',
        loss_scale=strategy,
        preserve_thinking=True,
        add_non_thinking_prefix=False,
    )
    template.set_mode('train')
    data = _qwen_data(n_tools, n_users)
    original = copy.deepcopy(data)
    encoded = template.encode(data)
    call = data['messages'][1]['content']
    answer = data['messages'][-1]['content']
    observations = '\n'.join(f'<tool_response>\nresult_{i}\n</tool_response>' for i in range(n_tools))
    turns = [('user', 'question'), ('assistant', call), ('user', observations)]
    turns += [('user', f'followup_{i}') for i in range(n_users)]
    turns.append(('assistant', answer))
    text = '\n'.join(f'<|im_start|>{role}\n{content}<|im_end|>' for role, content in turns) + '\n'
    assert tokenizer.decode(encoded['input_ids']) == text
    assert data == original

    assert len(encoded['labels']) == len(encoded['input_ids'])
    supervised = _supervised(tokenizer, encoded)
    assert answer in supervised
    assert (call in supervised) == (strategy == 'default')
    assert 'result_' not in supervised and 'followup_' not in supervised


def test_qwen_literal_placeholders_and_structured_calls():
    tokenizer = _make_tokenizer()
    template = get_template(tokenizer, template_type='qwen3_5', is_binary_loss_scale=False)
    template.set_mode('train')
    data = _qwen_data()
    data['messages'][-2]['content'] = '{{QUERY}} {{RESPONSE}} {{ROUND0}}'
    data['messages'][-1]['loss_scale'] = 0.4
    encoded = template.encode(data)
    text = tokenizer.decode(encoded['input_ids'])
    assert '<|im_start|>user\n{{QUERY}} {{RESPONSE}} {{ROUND0}}<|im_end|>' in text
    tokens = tokenizer.encode(data['messages'][-1]['content'], add_special_tokens=False)
    assert encoded['loss_scale'].count(0.4) == len(tokens)
    assert all(weight == 0 for label, weight in zip(encoded['labels'], encoded['loss_scale']) if label == -100)

    structured = _qwen_data()
    call = {'name': 'weather', 'arguments': {'city': 'Beijing'}}
    structured['messages'][1] = {
        'role': 'assistant',
        'content': '',
        'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': call}],
    }
    structured['messages'][2]['tool_call_id'] = 'call_1'
    native = copy.deepcopy(structured)
    native['messages'][1] = {'role': 'tool_call', 'content': call}
    native['messages'][2]['role'] = 'tool_response'
    assert template.encode(structured) == template.encode(native)


def test_qwen_inference_after_user_followup():
    tokenizer = _make_tokenizer()
    template = get_template(tokenizer, template_type='qwen3_5', enable_thinking=True)
    template.set_mode('transformers')
    data = _qwen_data(n_tools=2, n_users=2)
    data['messages'].pop()
    encoded = template.encode(data)
    text = tokenizer.decode(encoded['input_ids'])
    assert text.endswith('followup_1<|im_end|>\n<|im_start|>assistant\n<think>\n')
    assert text.count('<|im_start|>assistant\n') == 2
    assert 'labels' not in encoded


def test_hermes_followup_byte_match():
    tokenizer = _make_tokenizer()
    template = get_template(tokenizer, template_type='qwen2_5', agent_template='hermes')
    template.set_mode('train')
    call = '<tool_call>\n{"name": "weather", "arguments": {}}\n</tool_call>'
    data = {
        'messages': [
            {'role': 'system', 'content': ''},
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': call},
            {'role': 'tool', 'content': 'result_0'},
            {'role': 'user', 'content': 'followup_0'},
            {'role': 'assistant', 'content': 'final_answer'},
        ]
    }
    encoded = template.encode(data)
    turns = [
        ('user', 'question'),
        ('assistant', call),
        ('user', '<tool_response>\nresult_0\n</tool_response>'),
        ('user', 'followup_0'),
        ('assistant', 'final_answer'),
    ]
    expected = ''.join(f'<|im_start|>{role}\n{content}<|im_end|>\n' for role, content in turns)
    assert tokenizer.decode(encoded['input_ids']) == expected
    assert _supervised(tokenizer, encoded) == call + '<|im_end|>\nfinal_answer<|im_end|>\n'


@pytest.mark.parametrize('kind', ['glm4_5', 'glm4_7', 'glm5_1'])
@pytest.mark.parametrize('strategy', ['default', 'last_round'])
def test_glm_multiple_followups(kind, strategy):
    tokenizer = _make_tokenizer(glm=True)
    template = get_template(
        tokenizer,
        template_type=kind,
        loss_scale=strategy,
        preserve_thinking=True,
        add_non_thinking_prefix=False,
    )
    template.set_mode('train')
    call = '<tool_call>weather\n<arg_key>city</arg_key>\n<arg_value>BJ</arg_value>\n</tool_call><|observation|>'
    messages = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': call}]
    messages += [{'role': 'tool', 'content': 'result_0'}]
    messages += [{'role': 'user', 'content': f'followup_{i}'} for i in range(2)]
    messages.append({'role': 'assistant', 'content': 'final_answer'})
    encoded = template.encode({'messages': messages})
    text = tokenizer.decode(encoded['input_ids'])
    newline = '\n' if kind == 'glm4_5' else ''
    user = '<|user|>' + newline
    result = newline + '<tool_response>' + newline + 'result_0' + newline + '</tool_response>'
    expected = ('[gMASK]<sop>' + user + 'question<|assistant|>' + call + result + user + 'followup_0' + user
                + 'followup_1<|assistant|>' + newline + 'final_answer<|user|>')
    assert text == expected
    assert _supervised(tokenizer, encoded) == (call if strategy == 'default' else '') + 'final_answer<|user|>'


def _hunyuan_data(count=2, users=2, style='serialized'):
    calls = [{'name': f'lookup{i}', 'arguments': {'city': '北京'}} for i in range(count)]
    call = '<tool_calls>' + '\n'.join(
        '<tool_call>' + c['name'] + '\n```json\n{"city": "北京"}\n```</tool_call>' for c in calls) + '</tool_calls>'
    if style == 'serialized':
        assistants = [{'role': 'assistant', 'content': call}]
    elif style == 'openai':
        assistants = [{
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {'id': f'call{i}', 'type': 'function', 'function': c} for i, c in enumerate(calls)
            ],
        }]
    else:
        assistants = [{'role': 'tool_call', 'content': c} for c in calls]
    messages = [{'role': 'user', 'content': 'question'}] + assistants
    messages += [{'role': 'tool', 'content': f'result{i}', 'tool_call_id': f'call{i}'} for i in range(count)]
    messages += [{'role': 'user', 'content': f'followup{i}'} for i in range(users)]
    messages.append({'role': 'assistant', 'content': 'answer'})
    return {'messages': messages}, call


@pytest.mark.parametrize('style', ['serialized', 'openai', 'native'])
@pytest.mark.parametrize('strategy', ['default', 'last_round'])
def test_hunyuan_sequences_and_structured_calls(style, strategy):
    tokenizer = _make_tokenizer()
    template = get_template(
        tokenizer,
        template_type='hunyuan',
        loss_scale=strategy,
        preserve_thinking=True,
        add_non_thinking_prefix=False,
        enable_thinking=True,
        response_prefix='',
    )
    template.set_mode('train')
    data, call = _hunyuan_data(style=style)
    encoded = template.encode(data)
    eos, user, assistant = '<｜hy_place▁holder▁no▁2｜>', '<｜hy_User｜>', '<｜hy_Assistant｜>'
    expected = '<｜hy_begin▁of▁sentence｜>' + user + 'question' + assistant + call + eos
    expected += user + '<tool_responses>' + '\n'.join(
        f'<tool_response>result{i}</tool_response>' for i in range(2)) + '</tool_responses>'
    expected += user + 'followup0' + user + 'followup1' + assistant + 'answer' + eos
    assert tokenizer.decode(encoded['input_ids']) == expected
    supervised = _supervised(tokenizer, encoded)
    assert ('answer' in supervised) and ('followup' not in supervised) and ('result' not in supervised)
    assert (call in supervised) == (strategy == 'default')


@pytest.mark.parametrize('strategy', ['default', 'last_round'])
def test_deepseek_followup_boundaries(strategy):
    tokenizer = _make_tokenizer()
    template = get_template(tokenizer, template_type='deepseek_v3_1', loss_scale=strategy, enable_thinking=False)
    template.set_mode('train')
    call = ('<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>weather<｜tool▁sep｜>{}'
            '<｜tool▁call▁end｜><｜tool▁calls▁end｜>')
    messages = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': call}]
    messages += [{'role': 'tool', 'content': f'result_{i}'} for i in range(2)]
    messages += [{'role': 'user', 'content': f'followup_{i}'} for i in range(2)]
    messages.append({'role': 'assistant', 'content': 'answer'})
    encoded = template.encode({'messages': messages})
    results = ''.join(f'<｜tool▁output▁begin｜>result_{i}<｜tool▁output▁end｜>' for i in range(2))
    expected = ('<｜begin▁of▁sentence｜><｜User｜>question<｜Assistant｜></think>' + call
                + '<｜end▁of▁sentence｜>' + results + '<｜User｜>followup_0<｜User｜>followup_1<｜Assistant｜>'
                + '</think>answer<｜end▁of▁sentence｜>')
    assert tokenizer.decode(encoded['input_ids']) == expected
    supervised = _supervised(tokenizer, encoded)
    assert 'answer' in supervised and 'result_' not in supervised and 'followup_' not in supervised
    assert (call in supervised) == (strategy == 'default')


@pytest.mark.parametrize('strategy', ['default', 'last_round'])
def test_youtu_followup_boundaries(strategy):
    tokenizer = _make_tokenizer()
    tokenizer.add_special_tokens({'bos_token': '<|begin_of_text|>'})
    template = get_template(
        tokenizer,
        template_type='youtu_llm',
        loss_scale=strategy,
        preserve_thinking=True,
        add_non_thinking_prefix=False,
    )
    template.set_mode('train')
    call = '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'
    messages = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': call}]
    messages += [{'role': 'tool', 'content': f'result{i}'} for i in range(2)]
    messages += [{'role': 'user', 'content': f'followup{i}'} for i in range(2)]
    messages.append({'role': 'assistant', 'content': 'answer'})
    encoded = template.encode({'messages': messages})
    expected = '<|begin_of_text|><|User|>question<|Assistant|>' + call + '<|end_of_text|>'
    expected += '<|User|><tool_response>result0</tool_response>\n<tool_response>result1</tool_response>'
    expected += '<|User|>followup0<|User|>followup1<|Assistant|>answer<|end_of_text|>'
    assert tokenizer.decode(encoded['input_ids']) == expected
    supervised = _supervised(tokenizer, encoded)
    assert 'answer' in supervised and 'followup' not in supervised
    assert (call in supervised) == (strategy == 'default')


@pytest.mark.parametrize('agent', ['react_en', 'react_zh', 'react_grpo'])
def test_react_followup_keeps_single_observation(agent):
    tokenizer = _make_tokenizer()
    template = get_template(tokenizer, template_type='qwen2_5', agent_template=agent)
    template.set_mode('train')
    keyword = agent_template_map[agent].keyword
    action = f'{keyword.action} lookup\n{keyword.action_input} {{"q": "x"}}'
    data = {
        'messages': [
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': action},
            {'role': 'tool', 'content': 'observation_text'},
            {'role': 'user', 'content': 'followup_text'},
            {'role': 'assistant', 'content': 'final_answer'},
        ]
    }
    encoded = template.encode(data)
    text = tokenizer.decode(encoded['input_ids'])
    assert text.count(keyword.observation) == 1
    supervised = _supervised(tokenizer, encoded)
    assert 'final_answer' in supervised
    assert 'observation_text' not in supervised and 'followup_text' not in supervised


NATIVE_FAMILIES = [
    ('minicpm5', 'minicpm5'),
    ('minimax_m2', 'minimax_m2'),
    ('minimax_m3', 'minimax'),
    ('seed_oss', 'seed_oss'),
]


def _native_call(template):
    call = template.agent_template._format_tool_calls([{
        'role': 'tool_call',
        'content': json.dumps({'name': 'lookup', 'arguments': {'q': 'x'}}),
    }])
    return ''.join(part for part in call if isinstance(part, str)) if isinstance(call, list) else call


@pytest.mark.parametrize('agent,template_type', NATIVE_FAMILIES)
@pytest.mark.parametrize('mode', ['train', 'transformers'])
def test_native_families_accept_followup(agent, template_type, mode):
    tokenizer = _make_tokenizer()
    template = get_template(tokenizer, template_type=template_type, agent_template=agent)
    template.set_mode(mode)
    data = {
        'messages': [
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': _native_call(template)},
            {'role': 'tool', 'content': 'OBS'},
            {'role': 'user', 'content': 'follow'},
            {'role': 'assistant', 'content': 'answer'},
        ]
    }
    if mode != 'train':
        data['messages'].pop()
    encoded = template.encode(data)
    text = tokenizer.decode(encoded['input_ids'])
    assert 'OBS' in text and 'follow' in text
    if mode == 'train':
        assert 'answer' in text and 'answer' in _supervised(tokenizer, encoded)
        assert 'follow' not in _supervised(tokenizer, encoded)
    else:
        assert 'labels' not in encoded


@pytest.mark.parametrize('agent,template_type', NATIVE_FAMILIES + [('hermes', 'qwen2_5'), ('glm4_5', 'qwen2_5')])
@pytest.mark.parametrize('n_users', [1, 2])
def test_every_user_turn_is_closed(agent, template_type, n_users):
    tokenizer = _make_tokenizer()
    template = get_template(tokenizer, template_type=template_type, agent_template=agent)
    template.set_mode('train')
    messages = [
        {'role': 'user', 'content': 'q1'},
        {'role': 'assistant', 'content': _native_call(template)},
        {'role': 'tool', 'content': 'OBS'},
    ]
    messages += [{'role': 'user', 'content': f'f{i}'} for i in range(n_users)]
    messages.append({'role': 'assistant', 'content': 'answer'})
    text = tokenizer.decode(template.encode({'messages': messages})['input_ids'])
    prefix, suffix = ''.join(template.template_meta.prompt).split('{{QUERY}}', 1)
    chat_sep = ''.join(template.template_meta.chat_sep or [])
    for i in range(n_users):
        terminator = suffix if i == n_users - 1 else chat_sep
        assert prefix + f'f{i}' + terminator in text


@pytest.mark.parametrize('agent,template_type', [('hermes', 'qwen2_5'), ('minicpm5', 'minicpm5'), ('glm4_5', 'glm4_5')])
def test_multiple_tool_followup_rounds(agent, template_type):
    tokenizer = _make_tokenizer(glm=agent == 'glm4_5')
    template = get_template(tokenizer, template_type=template_type, agent_template=agent)
    template.set_mode('train')
    call = _native_call(template)
    messages = []
    for i in range(2):
        messages += [
            {'role': 'user', 'content': f'q{i}'},
            {'role': 'assistant', 'content': call},
            {'role': 'tool', 'content': f'OBS{i}'},
            {'role': 'user', 'content': f'follow{i}'},
            {'role': 'assistant', 'content': f'answer{i}'},
        ]
    encoded = template.encode({'messages': messages})
    text = tokenizer.decode(encoded['input_ids'])
    for token in ('OBS0', 'OBS1', 'follow0', 'follow1', 'answer0', 'answer1'):
        assert token in text
    supervised = _supervised(tokenizer, encoded)
    assert 'follow0' not in supervised and 'follow1' not in supervised
    assert 'answer1' in supervised


def test_non_chat_user_followup_requires_chat_template():
    tokenizer = _make_tokenizer()
    template = get_template(tokenizer, template_type='qwen3_5', agent_template='qwen3_5', use_chat_template=False)
    template.set_mode('train')
    data = _qwen_data()
    with pytest.raises(ValueError, match='use_chat_template'):
        template.encode(data)
