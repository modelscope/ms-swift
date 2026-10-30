# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec('ray'), 'Ray is required for completion logging')
class TestRayPromptLogging(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from swift.ray.megatron.base_trainer import BaseRayTrainer
        from swift.rl_core.data import OnPolicySample

        cls.trainer_cls = BaseRayTrainer
        cls.sample_cls = OnPolicySample

    def setUp(self):
        self.trainer = self.trainer_cls.__new__(self.trainer_cls)
        self.trainer.args = SimpleNamespace(log_completions=True, output_dir='unused')
        self.trainer.template = SimpleNamespace(safe_decode=lambda ids: f'tokens:{ids}')
        self.trainer._completions_writer = Mock()
        self.formatter = Mock(side_effect=repr)
        self.trainer._format_log_prompt = self.formatter
        self.prompt = [{'role': 'user', 'content': 'Which button?'}]

    def sample(self, prompt=None, response='answer', prompt_id='same-id'):
        messages = deepcopy(self.prompt if prompt is None else prompt)
        messages.append({'role': 'assistant', 'content': response})
        return self.sample_cls(messages=messages, prompt_id=prompt_id)

    def rows(self):
        return self.trainer._completions_writer.append.call_args.args[0]

    def test_repeated_prompt_is_formatted_once(self):
        samples = [self.sample(response=f'answer {i}', prompt_id=str(i)) for i in range(16)]
        self.trainer._maybe_log_completions(samples, rewards=list(range(16)), gen_step=3)
        self.formatter.assert_called_once_with(self.prompt)
        self.trainer._completions_writer.append.assert_called_once()
        self.assertEqual(self.rows(), [{
            'gen_step': 3,
            'prompt': repr(self.prompt),
            'completion': f'answer {i}',
            'reward': float(i)
        } for i in range(16)])
        self.assertEqual([sample.prompt_id for sample in samples], list(map(str, range(16))))

    def test_all_prompt_fields_participate_in_reuse(self):
        prompts = [[{
            'role': 'system',
            'content': system
        }, {
            'role': 'user',
            'content': [{
                'type': 'image',
                'image': image
            }, {
                'type': 'text',
                'text': 'same text'
            }]
        }, {
            'role': 'assistant',
            'content': history
        }, {
            'role': 'user',
            'content': 'continue'
        }] for system, image, history in [('s', 'a.png', 'old'), ('s', 'b.png',
                                                                  'old'), ('s', 'a.png',
                                                                           'different'), ('other', 'a.png', 'old')]]
        samples = [self.sample(prompt=prompt) for prompt in prompts * 2]
        before = deepcopy(samples)
        self.trainer._maybe_log_completions(samples)
        self.assertEqual(self.formatter.call_count, 4)
        self.assertEqual([row['prompt'] for row in self.rows()], [repr(prompt) for prompt in prompts * 2])
        self.assertEqual(samples, before)

    def test_cache_does_not_survive_a_logging_call(self):
        samples = [self.sample() for _ in range(4)]
        before = set(vars(self.trainer))
        self.formatter.side_effect = lambda messages: 'first:' + repr(messages)
        self.trainer._maybe_log_completions(samples)
        self.formatter.side_effect = lambda messages: 'second:' + repr(messages)
        self.trainer._maybe_log_completions(samples)
        self.assertEqual(self.formatter.call_count, 2)
        self.assertTrue(all(row['prompt'].startswith('second:') for row in self.rows()))
        self.assertEqual(set(vars(self.trainer)), before)

    def test_non_serializable_prompts_keep_original_formatter(self):
        prompt = [{'role': 'user', 'content': object()}]
        samples = [self.sample(prompt) for _ in range(4)]
        self.trainer._maybe_log_completions(samples)
        self.assertEqual(self.formatter.call_count, 4)
        self.assertEqual([row['prompt'] for row in self.rows()], [repr(sample.messages[:-1]) for sample in samples])

    def test_circular_prompts_keep_original_formatter(self):
        content = []
        content.append(content)
        samples = [self.sample([{'role': 'user', 'content': content}]) for _ in range(4)]
        self.trainer._maybe_log_completions(samples)
        self.assertEqual(self.formatter.call_count, 4)
        self.assertEqual([row['prompt'] for row in self.rows()], [repr(sample.messages[:-1]) for sample in samples])

    def test_json_coercions_cannot_merge_different_prompts(self):
        for first, second in [(('one', 'two'), ['one', 'two']), ({1: 'value'}, {'1': 'value'})]:
            with self.subTest(first=first):
                self.formatter.reset_mock()
                prompts = [[{'role': 'user', 'content': content}] for content in (first, second)]
                self.trainer._maybe_log_completions([self.sample(prompt) for prompt in prompts * 4])
                self.assertEqual(self.formatter.call_count, 5)
                self.assertEqual([row['prompt'] for row in self.rows()], [repr(prompt) for prompt in prompts * 4])

    def test_dictionary_order_is_preserved_for_fallback_rendering(self):
        prompts = [[{
            'role': 'user',
            'content': content
        }] for content in [{
            'first': 1,
            'second': 2
        }, {
            'second': 2,
            'first': 1
        }]]
        self.trainer._maybe_log_completions([self.sample(prompt) for prompt in prompts * 2])
        self.assertEqual(self.formatter.call_count, 2)
        self.assertEqual([row['prompt'] for row in self.rows()], [repr(prompt) for prompt in prompts * 2])

    def test_completion_decoding_and_partial_rewards_are_preserved(self):
        other = [{'role': 'user', 'content': 'Other prompt'}]
        samples = [
            self.sample(response='text'),
            self.sample(response=[5, 6]),
            self.sample(other, response={'input_ids': [7]}),
            self.sample_cls(messages=other),
            self.sample_cls(messages=[]),
        ]
        self.trainer._maybe_log_completions(samples, rewards=[1, 2.5], gen_step=9)
        expected = [{
            'gen_step': 9,
            'prompt': prompt,
            'completion': completion
        } for prompt, completion in [(repr(self.prompt), 'text'), (repr(self.prompt),
                                                                   'tokens:[5, 6]'), (repr(other),
                                                                                      'tokens:[7]'), (repr(other),
                                                                                                      ''), ('[]', '')]]
        expected[0]['reward'] = 1.0
        expected[1]['reward'] = 2.5
        self.assertEqual(self.rows(), expected)
        self.assertEqual(self.formatter.call_count, 3)

    def test_disabled_and_empty_logging_are_no_ops(self):
        self.trainer._completions_writer = None
        with patch('swift.ray.megatron.base_trainer.JsonlWriter') as writer:
            self.trainer._maybe_log_completions([])
            self.trainer.args.log_completions = False
            self.trainer._maybe_log_completions([self.sample()])
            writer.assert_not_called()
        self.formatter.assert_not_called()

    def test_formatting_failure_propagates_and_next_call_starts_fresh(self):
        failure = RuntimeError('formatter failed')
        before = set(vars(self.trainer))
        self.formatter.side_effect = failure
        samples = [self.sample() for _ in range(4)]
        with self.assertRaises(RuntimeError) as raised:
            self.trainer._maybe_log_completions(samples)
        self.assertIs(raised.exception, failure)
        self.trainer._completions_writer.append.assert_not_called()
        self.formatter.side_effect = repr
        self.trainer._maybe_log_completions(samples)
        self.assertEqual(self.formatter.call_count, 2)
        self.assertEqual(set(vars(self.trainer)), before)

    def test_writer_failure_propagates_and_next_call_starts_fresh(self):
        failure = OSError('disk full')
        before = set(vars(self.trainer))
        self.trainer._completions_writer.append.side_effect = failure
        samples = [self.sample() for _ in range(4)]
        with self.assertRaises(OSError) as raised:
            self.trainer._maybe_log_completions(samples)
        self.assertIs(raised.exception, failure)
        self.trainer._completions_writer.append.side_effect = None
        self.trainer._maybe_log_completions(samples)
        self.assertEqual(self.formatter.call_count, 2)
        self.assertEqual(set(vars(self.trainer)), before)

    def test_native_formatter_is_reused(self):
        del self.trainer._format_log_prompt
        self.trainer.template.encode = Mock(return_value={'input_ids': [1, 2]})
        self.trainer._maybe_log_completions([self.sample() for _ in range(4)])
        self.trainer.template.encode.assert_called_once()
        self.assertEqual([row['prompt'] for row in self.rows()], ['tokens:[1, 2]'] * 4)

    def test_native_formatter_fallback_is_reused(self):
        del self.trainer._format_log_prompt
        self.trainer.template.encode = Mock(side_effect=ValueError('cannot encode'))
        self.trainer._maybe_log_completions([self.sample() for _ in range(4)])
        self.trainer.template.encode.assert_called_once()
        self.assertEqual([row['prompt'] for row in self.rows()], ['user: Which button?\n'] * 4)

    def test_real_jsonl_writer_preserves_rows(self):
        with tempfile.TemporaryDirectory() as output_dir:
            self.trainer.args.output_dir = output_dir
            self.trainer._completions_writer = None
            with patch('swift.utils.io_utils.is_master', return_value=True):
                self.trainer._maybe_log_completions([self.sample(response=str(i)) for i in range(4)], gen_step=2)
            rows = [json.loads(line) for line in (Path(output_dir) / 'completions.jsonl').read_text().splitlines()]
        self.assertEqual(rows, [{'gen_step': 2, 'prompt': repr(self.prompt), 'completion': str(i)} for i in range(4)])
        self.formatter.assert_called_once()


if __name__ == '__main__':
    unittest.main()
