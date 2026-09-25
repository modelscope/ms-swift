# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import json
import os
import unittest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from types import SimpleNamespace

from swift import get_processor
from swift.template import get_template


def _make_tokenizer():
    # Byte-level tokens keep this template regression independent of model downloads.
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token='<|im_end|>',
        pad_token='<|endoftext|>',
        additional_special_tokens=['<|im_start|>', '<think>', '</think>'])
    tokenizer.model_info = SimpleNamespace(config=SimpleNamespace(), task_type='causal_lm', max_model_len=8192)
    tokenizer.model_meta = SimpleNamespace(is_multimodal=False)
    return tokenizer


def _make_glm_tokenizer():
    # GLM renders tool results with its own control tokens, so a GLM-flavoured byte tokenizer keeps this
    # regression independent of model downloads while still exercising the `<|user|>` splice.
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token='<|endoftext|>',
        pad_token='<|endoftext|>',
        additional_special_tokens=[
            '[gMASK]', '<sop>', '<|system|>', '<|user|>', '<|assistant|>', '<|observation|>', '<think>', '</think>'
        ])
    tokenizer.model_info = SimpleNamespace(config=SimpleNamespace(), task_type='causal_lm', max_model_len=8192)
    tokenizer.model_meta = SimpleNamespace(is_multimodal=False)
    return tokenizer


class TestQwenToolUserFollowup(unittest.TestCase):
    """Reference byte-match: the root fix must reproduce independent ChatML rendering for qwen3_5."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _make_tokenizer()

    def make_template(self, **kwargs):
        template = get_template(
            self.tokenizer, template_type='qwen3_5', preserve_thinking=True, add_non_thinking_prefix=False, **kwargs)
        template.set_mode('train')
        return template

    @staticmethod
    def make_data(n_tools=1, n_users=1):
        call = '<think>\nplan\n</think>\n\n<tool_call>\n<function=weather>\n</function>\n</tool_call>'
        answer = '<think>\nreply\n</think>\n\nfinal_answer'
        messages = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': call}]
        messages += [{'role': 'tool', 'content': f'result_{i}'} for i in range(n_tools)]
        messages += [{'role': 'user', 'content': f'followup_{i}'} for i in range(n_users)]
        messages.append({'role': 'assistant', 'content': answer})
        return {'messages': messages}

    def test_followup_boundaries_and_labels(self):
        for strategy in ('default', 'last_round'):
            for n_tools, n_users in ((1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2)):
                with self.subTest(strategy=strategy, tools=n_tools, users=n_users):
                    template = self.make_template(loss_scale=strategy)
                    data = self.make_data(n_tools, n_users)
                    original = copy.deepcopy(data)
                    encoded = template.encode(data)
                    call = data['messages'][1]['content']
                    answer = data['messages'][-1]['content']
                    observations = '\n'.join(f'<tool_response>\nresult_{i}\n</tool_response>' for i in range(n_tools))
                    turns = [('user', 'question'), ('assistant', call), ('user', observations)]
                    turns += [('user', f'followup_{i}') for i in range(n_users)]
                    turns.append(('assistant', answer))
                    # Independent ChatML rendering: every user turn retains its own boundary.
                    text = '\n'.join(f'<|im_start|>{role}\n{content}<|im_end|>' for role, content in turns) + '\n'
                    self.assertEqual(self.tokenizer.decode(encoded['input_ids']), text)
                    self.assertEqual(data, original)
                    expected_labels = [-100] * len(encoded['input_ids'])
                    responses = [answer] if strategy == 'last_round' else [call, answer]
                    for response in responses:
                        start = text.index(response)
                        begin = len(self.tokenizer.encode(text[:start], add_special_tokens=False))
                        tokens = self.tokenizer.encode(response + '<|im_end|>\n', add_special_tokens=False)
                        expected_labels[begin:begin + len(tokens)] = tokens
                    self.assertEqual(encoded['labels'], expected_labels)

    def test_no_followup_is_unchanged(self):
        # The fix must be a no-op when no user turn follows the tool results.
        template = self.make_template()
        data = self.make_data(n_tools=2, n_users=0)
        encoded = template.encode(data)
        text = self.tokenizer.decode(encoded['input_ids'])
        self.assertIn('<tool_response>\nresult_0\n</tool_response>', text)

    def test_followup_template_placeholders_are_literal(self):
        template = self.make_template()
        for content in ('{{QUERY}}', '{{RESPONSE}}', '{{ROUND0}} {{ROUND1}}'):
            with self.subTest(content=content):
                data = self.make_data()
                data['messages'][-2]['content'] = content
                encoded = template.encode(data)
                text = self.tokenizer.decode(encoded['input_ids'])
                self.assertIn('<|im_start|>user\n' + content + '<|im_end|>', text)

    def test_openai_and_swift_tool_calls_match(self):
        # The follow-up path must treat OpenAI-format tool_calls and native tool_call/tool_response alike.
        call = {'name': 'weather', 'arguments': {'city': 'Beijing'}}
        data = self.make_data()
        data['messages'][1] = {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': 'call_1',
                'type': 'function',
                'function': call
            }]
        }
        data['messages'][2]['tool_call_id'] = 'call_1'
        native = copy.deepcopy(data)
        native['messages'][1] = {'role': 'tool_call', 'content': call}
        native['messages'][2]['role'] = 'tool_response'
        template = self.make_template()
        self.assertEqual(template.encode(data), template.encode(native))

    def test_response_loss_weight_is_preserved(self):
        # Splicing the follow-up into the query must not disturb the assistant response loss weights.
        template = self.make_template(is_binary_loss_scale=False)
        data = self.make_data()
        data['messages'][-1]['loss_scale'] = 0.4
        encoded = template.encode(data)
        text = self.tokenizer.decode(encoded['input_ids'])
        start = text.index(data['messages'][-1]['content'])
        begin = len(self.tokenizer.encode(text[:start], add_special_tokens=False))
        length = len(self.tokenizer.encode(data['messages'][-1]['content'], add_special_tokens=False))
        self.assertEqual(encoded['loss_scale'][begin:begin + length], [0.4] * length)
        for label, weight in zip(encoded['labels'], encoded['loss_scale']):
            if label == -100:
                self.assertEqual(weight, 0.)

    def test_inference_after_user_followup(self):
        template = self.make_template(enable_thinking=True)
        template.set_mode('transformers')
        data = self.make_data(n_tools=2, n_users=2)
        data['messages'].pop()
        encoded = template.encode(data)
        text = self.tokenizer.decode(encoded['input_ids'])
        self.assertTrue(text.endswith('followup_1<|im_end|>\n<|im_start|>assistant\n<think>\n'))
        self.assertEqual(text.count('<|im_start|>assistant\n'), 2)
        self.assertIn('</tool_response><|im_end|>\n<|im_start|>user\nfollowup_0', text)
        self.assertNotIn('labels', encoded)

    def test_multiple_rounds_keep_last_round_supervision(self):
        data = self.make_data()
        data['messages'] += self.make_data(n_tools=2, n_users=2)['messages']
        template = self.make_template(loss_scale='last_round')
        encoded = template.encode(data)
        text = self.tokenizer.decode(encoded['input_ids'])
        self.assertEqual(text.count('<tool_response>'), 3)
        self.assertEqual(text.count('<|im_start|>assistant\n'), 4)
        supervised = [token for token in encoded['labels'] if token != -100]
        self.assertEqual(self.tokenizer.decode(supervised), data['messages'][-1]['content'] + '<|im_end|>\n')


class TestHermesToolUserFollowup(unittest.TestCase):
    """The root fix generalizes: a non-qwen ChatML template stops crashing and keeps the user boundary."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _make_tokenizer()

    def make_template(self):
        template = get_template(self.tokenizer, template_type='qwen2_5', agent_template='hermes')
        template.set_mode('train')
        return template

    @staticmethod
    def make_data():
        messages = [
            {
                'role': 'user',
                'content': 'question'
            },
            {
                'role': 'assistant',
                'content': '<tool_call>\n{"name": "weather", "arguments": {}}\n</tool_call>'
            },
            {
                'role': 'tool',
                'content': 'result_0'
            },
            {
                'role': 'user',
                'content': 'followup_0'
            },
            {
                'role': 'assistant',
                'content': 'final_answer'
            },
        ]
        return {'messages': messages}

    def test_does_not_crash_and_keeps_boundary(self):
        template = self.make_template()
        data = self.make_data()
        data['messages'].insert(0, {'role': 'system', 'content': ''})
        encoded = template.encode(data)
        text = self.tokenizer.decode(encoded['input_ids'])
        turns = [('user', 'question'), ('assistant', data['messages'][2]['content']),
                 ('user', '<tool_response>\nresult_0\n</tool_response>'), ('user', 'followup_0'),
                 ('assistant', 'final_answer')]
        expected = ''.join(f'<|im_start|>{role}\n{content}<|im_end|>\n' for role, content in turns)
        self.assertEqual(text, expected)
        supervised = self.tokenizer.decode([token for token in encoded['labels'] if token != -100])
        self.assertEqual(supervised, data['messages'][2]['content'] + '<|im_end|>\nfinal_answer<|im_end|>\n')
        self.assertIn('<tool_response>\nresult_0\n</tool_response>', text)
        self.assertIn('followup_0', text)
        # The follow-up user keeps its own turn boundary rather than being glued to the tool result.
        self.assertIn('</tool_response><|im_end|>\n<|im_start|>user\nfollowup_0', text)


class TestGLMToolUserFollowup(unittest.TestCase):
    """GLM renders tool results independently; a follow-up user must become a normal `<|user|>` turn."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _make_glm_tokenizer()

    def make_template(self):
        template = get_template(self.tokenizer, template_type='glm4_5')
        template.set_mode('train')
        return template

    @staticmethod
    def make_data(n_users=1):
        call = '<tool_call>weather\n<arg_key>city</arg_key>\n<arg_value>BJ</arg_value>\n</tool_call>'
        messages = [
            {
                'role': 'user',
                'content': 'question'
            },
            {
                'role': 'assistant',
                'content': call
            },
            {
                'role': 'tool',
                'content': 'result_0'
            },
        ]
        messages += [{'role': 'user', 'content': f'followup_{i}'} for i in range(n_users)]
        messages.append({'role': 'assistant', 'content': 'final_answer'})
        return {'messages': messages}

    def test_followup_spliced_before_assistant(self):
        template = self.make_template()
        encoded = template.encode(self.make_data(n_users=1))
        text = self.tokenizer.decode(encoded['input_ids'])
        # Matches the official jinja: `</tool_response><|user|>\n{followup}<|assistant|>`.
        self.assertIn('</tool_response><|user|>\nfollowup_0<|assistant|>', text)
        # The follow-up belongs to the query side and must not be supervised.
        supervised = self.tokenizer.decode([t for t in encoded['labels'] if t != -100])
        self.assertNotIn('followup_0', supervised)

    def test_multiple_followups_and_last_round(self):
        for kind in ('glm4_5', 'glm4_7', 'glm5_1'):
            for strategy in ('default', 'last_round'):
                with self.subTest(template=kind, loss_scale=strategy):
                    template = get_template(
                        self.tokenizer,
                        template_type=kind,
                        loss_scale=strategy,
                        preserve_thinking=True,
                        add_non_thinking_prefix=False)
                    template.set_mode('train')
                    data = self.make_data(n_users=2)
                    data['messages'][1]['content'] += '<|observation|>'
                    encoded = template.encode(data)
                    text = self.tokenizer.decode(encoded['input_ids'])
                    newline = '\n' if kind == 'glm4_5' else ''
                    user = '<|user|>' + newline
                    assistant = '<|assistant|>'
                    call = data['messages'][1]['content']
                    result = (newline + '<tool_response>' + newline + 'result_0' + newline + '</tool_response>')
                    expected = ('[gMASK]<sop>' + user + 'question' + assistant + call + result + user + 'followup_0'
                                + user + 'followup_1' + assistant + newline + 'final_answer<|user|>')
                    self.assertEqual(text, expected)
                    supervised = self.tokenizer.decode([t for t in encoded['labels'] if t != -100])
                    self.assertEqual(supervised, (call if strategy == 'default' else '') + 'final_answer<|user|>')

    def test_no_followup_is_unchanged(self):
        template = self.make_template()
        encoded = template.encode(self.make_data(n_users=0))
        text = self.tokenizer.decode(encoded['input_ids'])
        self.assertIn('</tool_response><|assistant|>', text)
        self.assertNotIn('<|user|>\nfollowup', text)


class TestInheritedHermesFollowup(unittest.TestCase):

    def test_youtu_native_boundaries_and_labels(self):
        tokenizer = _make_tokenizer()
        tokenizer.add_special_tokens({'bos_token': '<|begin_of_text|>'})
        for strategy in ('default', 'last_round'):
            for mode in ('train', 'transformers'):
                for count in (1, 2):
                    with self.subTest(strategy=strategy, mode=mode, count=count):
                        template = get_template(
                            tokenizer,
                            template_type='youtu_llm',
                            loss_scale=strategy,
                            preserve_thinking=True,
                            add_non_thinking_prefix=False)
                        template.set_mode(mode)
                        call = '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'
                        messages = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': call}]
                        messages += [{'role': 'tool', 'content': f'result{i}'} for i in range(count)]
                        messages += [{'role': 'user', 'content': f'followup{i}'} for i in range(count)]
                        if mode == 'train':
                            messages.append({'role': 'assistant', 'content': 'answer'})
                        encoded = template.encode({'messages': messages})
                        expected = '<|begin_of_text|><|User|>question<|Assistant|>' + call + '<|end_of_text|>'
                        expected += '<|User|>' + '\n'.join(f'<tool_response>result{i}</tool_response>'
                                                           for i in range(count))
                        expected += ''.join(f'<|User|>followup{i}' for i in range(count)) + '<|Assistant|>'
                        if mode == 'train':
                            expected += 'answer<|end_of_text|>'
                        else:
                            expected += '<think>\n\n</think>\n\n'
                        self.assertEqual(tokenizer.decode(encoded['input_ids']), expected)
                        if mode != 'train':
                            self.assertNotIn('labels', encoded)
                            continue
                        labels = [-100] * len(encoded['input_ids'])
                        responses = ['answer<|end_of_text|>']
                        if strategy == 'default':
                            responses.insert(0, call + '<|end_of_text|>')
                        for response in responses:
                            start = expected.index(response)
                            lo = len(tokenizer.encode(expected[:start], add_special_tokens=False))
                            tokens = tokenizer.encode(response, add_special_tokens=False)
                            labels[lo:lo + len(tokens)] = tokens
                        self.assertEqual(encoded['labels'], labels)


class TestHunyuanToolUserFollowup(unittest.TestCase):
    eos = '<｜hy_place▁holder▁no▁2｜>'
    user = '<｜hy_User｜>'
    assistant = '<｜hy_Assistant｜>'
    bos = '<｜hy_begin▁of▁sentence｜>'

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _make_tokenizer()

    def make_template(self, **kwargs):
        return get_template(
            self.tokenizer,
            template_type='hunyuan',
            preserve_thinking=True,
            add_non_thinking_prefix=False,
            enable_thinking=True,
            response_prefix='',
            **kwargs)

    @staticmethod
    def make_data(count=1, users=1, style='serialized'):
        calls = [{'name': f'lookup{i}', 'arguments': {'city': '北京'}} for i in range(count)]
        call = '<tool_calls>' + '\n'.join('<tool_call>' + c['name'] + '\n```json\n{"city": "北京"}\n```</tool_call>'
                                          for c in calls) + '</tool_calls>'
        if style == 'serialized':
            assistants = [{'role': 'assistant', 'content': call}]
        elif style == 'openai':
            assistants = [{
                'role':
                'assistant',
                'content':
                '',
                'tool_calls': [{
                    'id': f'call{i}',
                    'type': 'function',
                    'function': c
                } for i, c in enumerate(calls)]
            }]
        else:
            assistants = [{'role': 'tool_call', 'content': c} for c in calls]
        messages = [{'role': 'user', 'content': 'question'}] + assistants
        messages += [{'role': 'tool', 'content': f'result{i}', 'tool_call_id': f'call{i}'} for i in range(count)]
        messages += [{'role': 'user', 'content': f'followup{i}'} for i in range(users)]
        messages.append({'role': 'assistant', 'content': 'answer'})
        return {'messages': messages}, call

    def test_native_sequences_and_labels(self):
        for count, users in ((1, 0), (2, 0), (1, 1), (2, 2)):
            for mode in ('train', 'transformers'):
                for loss in ('default', 'last_round'):
                    with self.subTest(count=count, users=users, mode=mode, loss=loss):
                        reference = None
                        for style in ('serialized', 'openai', 'native'):
                            # A fresh template prevents earlier structured calls from hiding missing metadata.
                            t = self.make_template(loss_scale=loss)
                            t.set_mode(mode)
                            data, call = self.make_data(count, users, style)
                            if mode != 'train':
                                data['messages'].pop()
                            original = copy.deepcopy(data)
                            encoded = t.encode(data)
                            self.assertEqual(data, original)
                            expected = self.bos + self.user + 'question' + self.assistant + call + self.eos
                            expected += self.user + '<tool_responses>' + '\n'.join(
                                f'<tool_response>result{i}</tool_response>' for i in range(count)) + '</tool_responses>'
                            expected += ''.join(self.user + f'followup{i}' for i in range(users)) + self.assistant
                            if mode == 'train':
                                expected += 'answer' + self.eos
                            self.assertEqual(encoded['input_ids'],
                                             self.tokenizer.encode(expected, add_special_tokens=False))
                            if reference is not None:
                                self.assertEqual(encoded, reference)
                            reference = encoded
                            if mode != 'train':
                                self.assertNotIn('labels', encoded)
                                continue
                            labels = [-100] * len(encoded['input_ids'])
                            spans = ['answer' + self.eos]
                            if loss == 'default':
                                spans.insert(0, call + self.eos)
                            for response in spans:
                                start = expected.index(response)
                                lo = len(self.tokenizer.encode(expected[:start], add_special_tokens=False))
                                tokens = self.tokenizer.encode(response, add_special_tokens=False)
                                labels[lo:lo + len(tokens)] = tokens
                            self.assertEqual(encoded['labels'], labels)

    def test_call_roundtrip_and_template_reuse(self):
        t = self.make_template()
        t.set_mode('train')
        data, call = self.make_data(count=2, users=0, style='native')
        calls = data['messages'][1:3]
        rendered = t.agent_template._format_tool_calls(calls)
        self.assertEqual(rendered, call)
        parsed = t.agent_template.get_toolcall(rendered)
        self.assertEqual([c.name for c in parsed], ['lookup0', 'lookup1'])
        self.assertEqual([json.loads(c.arguments) for c in parsed], [{'city': '北京'}, {'city': '北京'}])
        raw, _ = self.make_data(count=2, users=0)
        before = t.encode(raw)
        t.encode(data)
        self.assertEqual(t.encode(raw), before)

    def test_multiround_loss_and_literal_content(self):
        for loss in ('default', 'last_round'):
            t = self.make_template(loss_scale=loss, is_binary_loss_scale=False)
            t.set_mode('train')
            first, _ = self.make_data()
            second, _ = self.make_data(style='openai')
            first['messages'][-1]['content'] = 'first answer'
            second['messages'][-2]['content'] = '{{QUERY}} {{RESPONSE}} {{ROUND0}}'
            second['messages'][-1].update(content='last answer', loss_scale=0.4)
            encoded = t.encode({'messages': first['messages'] + second['messages']})
            text = self.tokenizer.decode(encoded['input_ids'])
            self.assertIn('{{QUERY}} {{RESPONSE}} {{ROUND0}}', text)
            supervised = self.tokenizer.decode([x for x in encoded['labels'] if x != -100])
            self.assertNotIn('result', supervised)
            self.assertNotIn('{{QUERY}}', supervised)
            self.assertEqual('first answer' in supervised, loss == 'default')
            lo = len(self.tokenizer.encode(text[:text.index('last answer')], add_special_tokens=False))
            n = len(self.tokenizer.encode('last answer', add_special_tokens=False))
            self.assertEqual(encoded['loss_scale'][lo:lo + n], [0.4] * n)


@unittest.skipUnless(os.environ.get('SWIFT_TEST_HUNYUAN_PROCESSOR'), 'Set a local Hunyuan-1.8B tokenizer path')
class TestHunyuanRealTokenizer(TestHunyuanToolUserFollowup):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = get_processor(os.environ['SWIFT_TEST_HUNYUAN_PROCESSOR'], model_type='hunyuan_v1_dense')


class TestDeepSeekToolUserFollowup(unittest.TestCase):

    def test_native_boundaries_and_supervision(self):
        tokenizer = _make_tokenizer()
        call = ('<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>weather<｜tool▁sep｜>{}'
                '<｜tool▁call▁end｜><｜tool▁calls▁end｜>')
        for strategy in ('default', 'last_round'):
            for n_tools, n_users in ((1, 1), (2, 1), (1, 2), (2, 2)):
                for mode in ('train', 'transformers'):
                    for thinking in (False, True):
                        with self.subTest(
                                strategy=strategy, tools=n_tools, users=n_users, mode=mode, thinking=thinking):
                            template = get_template(
                                tokenizer, template_type='deepseek_v3_1', loss_scale=strategy, enable_thinking=thinking)
                            template.set_mode(mode)
                            messages = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': call}]
                            messages += [{'role': 'tool', 'content': f'result_{i}'} for i in range(n_tools)]
                            messages += [{'role': 'user', 'content': f'followup_{i}'} for i in range(n_users)]
                            if mode == 'train':
                                messages.append({'role': 'assistant', 'content': 'answer'})
                            encoded = template.encode({'messages': messages})
                            results = ''.join(f'<｜tool▁output▁begin｜>result_{i}<｜tool▁output▁end｜>'
                                              for i in range(n_tools))
                            users = ''.join(f'<｜User｜>followup_{i}' for i in range(n_users))
                            expected = ('<｜begin▁of▁sentence｜><｜User｜>question<｜Assistant｜></think>' + call
                                        + '<｜end▁of▁sentence｜>' + results + users + '<｜Assistant｜>')
                            if mode == 'train':
                                expected += '</think>answer<｜end▁of▁sentence｜>'
                            else:
                                expected += '<think>' if thinking else '</think>'
                            self.assertEqual(tokenizer.decode(encoded['input_ids']), expected)
                            if mode == 'train':
                                supervised = tokenizer.decode([t for t in encoded['labels'] if t != -100])
                                self.assertIn('answer', supervised)
                                self.assertNotIn('result_', supervised)
                                self.assertNotIn('followup_', supervised)
                                self.assertEqual(call in supervised, strategy == 'default')
                                expected_labels = [-100] * len(encoded['input_ids'])
                                responses = ['</think>answer<｜end▁of▁sentence｜>']
                                if strategy == 'default':
                                    responses.insert(0, '</think>' + call + '<｜end▁of▁sentence｜>')
                                for response in responses:
                                    start = expected.index(response)
                                    lo = len(tokenizer.encode(expected[:start], add_special_tokens=False))
                                    tokens = tokenizer.encode(response, add_special_tokens=False)
                                    expected_labels[lo:lo + len(tokens)] = tokens
                                self.assertEqual(encoded['labels'], expected_labels)
                            else:
                                self.assertNotIn('labels', encoded)

    def test_openai_native_calls_and_response_weight(self):
        tokenizer = _make_tokenizer()
        template = get_template(tokenizer, template_type='deepseek_v3_1', is_binary_loss_scale=False)
        template.set_mode('train')
        call = {'name': 'weather', 'arguments': {'city': 'BJ'}}
        messages = [{
            'role': 'user',
            'content': 'question'
        }, {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': 'call_1',
                'type': 'function',
                'function': call
            }]
        }, {
            'role': 'tool',
            'content': 'result',
            'tool_call_id': 'call_1'
        }, {
            'role': 'user',
            'content': 'followup'
        }, {
            'role': 'assistant',
            'content': 'answer',
            'loss_scale': 0.4
        }]
        native = copy.deepcopy(messages)
        native[1] = {'role': 'tool_call', 'content': call}
        native[2]['role'] = 'tool_response'
        encoded = template.encode({'messages': messages})
        self.assertEqual(encoded, template.encode({'messages': native}))
        text = tokenizer.decode(encoded['input_ids'])
        start = text.index('</think>answer')
        lo = len(tokenizer.encode(text[:start], add_special_tokens=False))
        length = len(tokenizer.encode('</think>answer', add_special_tokens=False))
        self.assertEqual(encoded['loss_scale'][lo:lo + length], [0.4] * length)
        self.assertTrue(
            all(weight == 0 for label, weight in zip(encoded['labels'], encoded['loss_scale']) if label == -100))


class TestToolResultsWithoutChatTemplate(unittest.TestCase):

    def test_tool_results_match_plain_generation_prompt(self):
        cases = (
            ('qwen3_5', 'qwen3_5', '<tool_call>\n<function=lookup>\n</function>\n</tool_call>',
             '<tool_response>\nresult\n</tool_response>'),
            ('hunyuan', 'hunyuan_hermes', '<tool_calls><tool_call>lookup\n```json\n{}\n```</tool_call></tool_calls>',
             '<tool_responses><tool_response>result</tool_response></tool_responses>'),
            ('youtu_llm', 'youtu', '<tool_call>{"name":"lookup","arguments":{}}</tool_call>',
             '<tool_response>result</tool_response>'),
        )
        for model, agent, call, result in cases:
            for mode in ('train', 'transformers'):
                for style in ('serialized', 'structured'):
                    with self.subTest(model=model, mode=mode, style=style):
                        tokenizer = _make_tokenizer()
                        tokenizer.add_special_tokens({'bos_token': '<|begin_of_text|>'})
                        template = get_template(
                            tokenizer, template_type=model, agent_template=agent, use_chat_template=False)
                        template.set_mode(mode)
                        assistant = {'role': 'assistant', 'content': call}
                        if style == 'structured':
                            assistant = {
                                'role': 'assistant',
                                'content': '',
                                'tool_calls': [{
                                    'type': 'function',
                                    'function': {
                                        'name': 'lookup',
                                        'arguments': '{}'
                                    }
                                }]
                            }
                        data = {
                            'messages': [
                                {
                                    'role': 'user',
                                    'content': 'question'
                                },
                                assistant,
                                {
                                    'role': 'tool',
                                    'content': 'result'
                                },
                            ]
                        }
                        plain = {'messages': [{'role': 'user', 'content': result}]}
                        if mode == 'train':
                            data['messages'].append({'role': 'assistant', 'content': 'answer'})
                            plain['messages'].append({'role': 'assistant', 'content': 'answer'})
                        original = copy.deepcopy(data)
                        expected = template.encode(plain)
                        # Generation templates retain the last round, without chat role delimiters.
                        actual = template.encode(data)
                        self.assertEqual(actual['input_ids'], expected['input_ids'])
                        if mode == 'train':
                            self.assertEqual(actual['labels'], expected['labels'])
                        self.assertEqual(data, original)
                        # Encoding again must not depend on metadata left by earlier calls.
                        self.assertEqual(template.encode(data), actual)


if __name__ == '__main__':
    unittest.main()
