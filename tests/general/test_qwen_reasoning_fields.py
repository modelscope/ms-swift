# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import unittest
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from types import SimpleNamespace

from swift.template import get_template


class TestQwenReasoningFields(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = os.environ.get('QWEN_REASONING_TOKENIZER')
        if path:
            cls.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        else:
            vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
            backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
            backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            backend.decoder = decoders.ByteLevel()
            cls.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend,
                eos_token='<|im_end|>',
                pad_token='<|endoftext|>',
                additional_special_tokens=['<|im_start|>'])
            # Match Qwen: added control tokens need not have special=True.
            cls.tokenizer.add_tokens(
                [AddedToken(word, special=False) for word in ('<think>', '</think>', '<tool_call>', '</tool_call>')])

    def template(self, **kwargs):
        processor = SimpleNamespace(
            tokenizer=self.tokenizer,
            model_info=SimpleNamespace(config={}, task_type='causal_lm', max_model_len=262144),
            model_meta=SimpleNamespace(is_multimodal=False))
        template = get_template(processor, template_type='qwen3_8', **kwargs)
        template.set_mode('train')
        return template

    def encode(self, reason, body, **kwargs):
        return self.template(**kwargs).encode({
            'messages': [{
                'role': 'user',
                'content': 'question'
            }, {
                'role': 'assistant',
                'reasoning_content': reason,
                'content': body
            }]
        })

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def test_literal_markers_keep_text_and_control_identity(self):
        reason = '  BEFORE quote "<think>" and "</think>" AFTER 中文🙂\n\n'
        body = 'body <think> </think> <tool_call> </tool_call> <|endoftext|> <|im_end|>'
        encoded = self.encode(reason, body)
        text = self.decode(encoded['input_ids'])
        self.assertIn('<think>\n' + reason + '\n</think>\n\n' + body, text)
        for word in ('<think>', '</think>'):
            self.assertEqual(encoded['input_ids'].count(self.tokenizer.convert_tokens_to_ids(word)), 1)
        for word in ('<tool_call>', '</tool_call>', '<|endoftext|>'):
            self.assertNotIn(self.tokenizer.convert_tokens_to_ids(word), encoded['input_ids'])
        end = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        self.assertEqual(encoded['input_ids'][-1], end)
        self.assertEqual(encoded['labels'][-1], end)
        self.assertIn(reason, self.decode([i for i in encoded['labels'] if i != -100]))

    def test_marker_free_encoding_matches_existing_path(self):
        structured = self.encode('reason', 'answer')
        existing = self.template().encode({
            'messages': [{
                'role': 'user',
                'content': 'question'
            }, {
                'role': 'assistant',
                'content': '<think>\nreason\n</think>\n\nanswer'
            }]
        })
        self.assertEqual(structured['input_ids'], existing['input_ids'])
        self.assertEqual(structured['labels'], existing['labels'])

    def test_empty_reasoning_and_body(self):
        for reason, body in [('', ''), ('', 'answer'), ('reason', ''), ('reason', None)]:
            with self.subTest(reason=reason, body=body):
                encoded = self.encode(reason, body)
                self.assertIn('<think>\n' + reason + '\n</think>\n\n', self.decode(encoded['input_ids']))
                self.assertEqual(encoded['labels'][-1], self.tokenizer.eos_token_id)

    def test_tool_calls_and_message_masks(self):
        for body in ('', 'body </think> tail'):
            messages = [
                {
                    'role': 'user',
                    'content': 'old question'
                },
                {
                    'role': 'assistant',
                    'reasoning_content': 'OLD_REASON',
                    'content': 'OLD_BODY',
                    'loss': False
                },
                {
                    'role': 'user',
                    'content': 'new question'
                },
                {
                    'role':
                    'assistant',
                    'reasoning_content':
                    'NEW_REASON <think> </think>',
                    'content':
                    body,
                    'tool_calls': [{
                        'function': {
                            'name': 'edit',
                            'arguments': {
                                'new_str': ''
                            }
                        }
                    }, {
                        'function': {
                            'name': 'unknown',
                            'arguments': {
                                'value': ' 中文🙂\n '
                            }
                        }
                    }]
                },
                {
                    'role': 'tool',
                    'content': 'RESULT_ONE'
                },
                {
                    'role': 'tool',
                    'content': 'RESULT_TWO'
                },
                {
                    'role': 'assistant',
                    'reasoning_content': 'FINISH_REASON',
                    'content': 'final'
                },
            ]
            encoded = self.template().encode({'messages': messages})
            context = self.decode(encoded['input_ids'])
            supervised = self.decode([i for i in encoded['labels'] if i != -100])
            for value in ('OLD_REASON', 'OLD_BODY', 'RESULT_ONE', 'RESULT_TWO', 'new question'):
                self.assertIn(value, context)
                self.assertNotIn(value, supervised)
            for value in ('NEW_REASON <think> </think>', 'FINISH_REASON', '<parameter=new_str>\n\n</parameter>',
                          '<parameter=value>\n 中文🙂\n \n</parameter>'):
                self.assertEqual(supervised.count(value), 1)
            if body:
                self.assertIn(body + '\n\n<tool_call>', context)

    def test_history_thinking_setting(self):
        messages = [
            {
                'role': 'user',
                'content': 'old'
            },
            {
                'role': 'assistant',
                'reasoning_content': 'OLD_REASON',
                'content': 'OLD_BODY'
            },
            {
                'role': 'user',
                'content': 'new'
            },
            {
                'role': 'assistant',
                'reasoning_content': 'NEW_REASON',
                'content': 'NEW_BODY'
            },
        ]
        encoded = self.template(preserve_thinking=False).encode({'messages': messages})
        text = self.decode(encoded['input_ids'])
        self.assertNotIn('OLD_REASON', text)
        for value in ('OLD_BODY', 'NEW_REASON', 'NEW_BODY'):
            self.assertIn(value, text)

    def test_actual_eos_not_duplicated_for_token_inputs(self):
        tokens = self.tokenizer.encode('answer', add_special_tokens=False) + [self.tokenizer.eos_token_id]
        encoded = self.template().encode({
            'messages': [{
                'role': 'user',
                'content': 'question'
            }, {
                'role': 'assistant',
                'content': {
                    'token_ids': tokens
                }
            }]
        })
        self.assertEqual(encoded['input_ids'][-len(tokens):], tokens)
        self.assertNotEqual(encoded['input_ids'][-2], self.tokenizer.eos_token_id)


if __name__ == '__main__':
    unittest.main()
