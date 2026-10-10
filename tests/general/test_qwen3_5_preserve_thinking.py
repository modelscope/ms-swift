# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
from copy import deepcopy
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from types import SimpleNamespace

from swift.template import TEMPLATE_MAPPING, get_template

PAST_THINK = '简单加法'
CURRENT_THINK = '2+3=5'
MESSAGES = [
    {
        'role': 'user',
        'content': '1+1等于几'
    },
    {
        'role': 'assistant',
        'content': f'<think>\n{PAST_THINK}。\n</think>\n\n等于2。'
    },
    {
        'role': 'user',
        'content': '那再加3呢'
    },
    {
        'role': 'assistant',
        'content': f'<think>\n{CURRENT_THINK}。\n</think>\n\n等于5。'
    },
]


class TestQwen3_5PreserveThinking(unittest.TestCase):
    """The official Qwen3.5/3.6 chat_template.jinja only keeps reasoning of the current agent
    round (after the last user message) and drops reasoning of past assistant turns, at both
    training and inference time. See #10255 and the review of #10259: the two remaining
    ``loss_scale='all'`` mismatches come from the ``preserve_thinking`` default."""

    @classmethod
    def setUpClass(cls):
        # A local byte tokenizer keeps whitespace visible without model downloads.
        vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        cls.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, eos_token='<|im_end|>', pad_token='<|endoftext|>')
        cls.processor = SimpleNamespace(
            tokenizer=cls.tokenizer,
            model_info=SimpleNamespace(config={}, task_type='causal_lm', max_model_len=8192),
            model_meta=SimpleNamespace(is_multimodal=False))

    def _encode(self, mode, messages=MESSAGES, chat_template_kwargs=None, template_type='qwen3_5', **kwargs):
        template = get_template(self.processor, template_type=template_type, **kwargs)
        template.set_mode(mode)
        encoded = template.encode({
            'messages': deepcopy(messages),
            'chat_template_kwargs': chat_template_kwargs or {},
        })
        return self.tokenizer.decode(encoded['input_ids'])

    def test_meta_pins_preserve_thinking_false(self):
        # Unlike qwen3_8 (which keeps historical thinking), qwen3_5 follows its official
        # chat_template.jinja and drops past-turn reasoning by default.
        self.assertIs(TEMPLATE_MAPPING['qwen3_8'].preserve_thinking, True)
        self.assertIs(TEMPLATE_MAPPING['qwen3_5'].preserve_thinking, False)

    def test_training_all_drops_past_turn_reasoning(self):
        # With loss_scale='all' the training text used to keep <think> of past assistant turns
        # while the jinja used at inference drops them (train/infer mismatch).
        rendered = self._encode('train', loss_scale='all')
        self.assertNotIn(PAST_THINK, rendered)
        self.assertIn(CURRENT_THINK, rendered)
        self.assertIn('等于2。', rendered)

    def test_training_last_round_drops_past_turn_reasoning(self):
        rendered = self._encode('train', loss_scale='last_round')
        self.assertNotIn(PAST_THINK, rendered)
        self.assertIn(CURRENT_THINK, rendered)

    def test_inference_drops_past_turn_reasoning(self):
        rendered = self._encode('vllm', messages=MESSAGES[:-1])
        self.assertNotIn(PAST_THINK, rendered)
        self.assertIn('等于2。', rendered)

    def test_explicit_opt_in_still_keeps_past_turn_reasoning(self):
        # Users who want to train on the full reasoning history can still opt in explicitly.
        rendered = self._encode('train', loss_scale='all', preserve_thinking=True)
        self.assertIn(PAST_THINK, rendered)
        self.assertIn(CURRENT_THINK, rendered)

    def test_current_agent_round_keeps_reasoning_across_tool_results(self):
        tool_call = '<tool_call>\n<function=search>\n<parameter=q>\nx\n</parameter>\n</function>\n</tool_call>'
        reasoning = '<think>\nSearch first.\n</think>\n\n'
        assistant = {'role': 'assistant', 'content': reasoning + tool_call}
        structured_assistant = {
            'role': 'assistant',
            'content': reasoning,
            'tool_calls': [{
                'type': 'function',
                'function': {
                    'name': 'search',
                    'arguments': {
                        'q': 'x'
                    }
                }
            }],
        }
        tool_result = {'role': 'tool', 'content': 'result'}
        wrapped_result = {'role': 'user', 'content': '  <tool_response>\nresult\n</tool_response>  '}
        query = {'role': 'user', 'content': 'Search twice.'}
        answer = {'role': 'assistant', 'content': '<think>\nDone.\n</think>\n\nanswer'}
        query_text = '<|im_start|>user\nSearch twice.<|im_end|>\n'
        call_text = f'<|im_start|>assistant\n{reasoning}{tool_call}<|im_end|>\n'
        result_text = '<|im_start|>user\n<tool_response>\nresult\n</tool_response><|im_end|>\n'
        answer_text = '<|im_start|>assistant\n<think>\nDone.\n</think>\n\nanswer<|im_end|>\n'
        # These expected strings follow the official jinja: every assistant after the
        # last real user query retains reasoning, including intermediate tool calls.
        for call in (assistant, structured_assistant):
            for result in (tool_result, wrapped_result):
                for steps in (1, 2):
                    with self.subTest(structured=call is structured_assistant, role=result['role'], steps=steps):
                        messages = [query] + [call, result] * steps + [answer]
                        rendered = self._encode('train', messages=messages, loss_scale='all')
                        self.assertEqual(rendered, query_text + (call_text + result_text) * steps + answer_text)

    def test_only_real_user_query_ends_the_agent_round(self):
        messages = [
            {
                'role': 'user',
                'content': 'old query'
            },
            {
                'role': 'assistant',
                'content': '<think>\nPAST_CALL\n</think>\n\nold call'
            },
            {
                'role': 'tool',
                'content': 'old result'
            },
            {
                'role': 'assistant',
                'content': '<think>\nPAST_ANSWER\n</think>\n\nold answer'
            },
            {
                'role': 'user',
                'content': 'current query'
            },
            {
                'role': 'assistant',
                'content': '<think>\nCURRENT_CALL\n</think>\n\ncurrent call'
            },
            {
                'role': 'tool',
                'content': 'current result'
            },
            {
                'role': 'assistant',
                'content': '<think>\nCURRENT_ANSWER\n</think>\n\ncurrent answer'
            },
        ]
        for mode, loss_scale in [('train', 'all'), ('train', 'last_round'), ('vllm', 'last_round')]:
            with self.subTest(mode=mode, loss_scale=loss_scale):
                prompt = messages[:-1] if mode == 'vllm' else messages
                rendered = self._encode(mode, messages=prompt, loss_scale=loss_scale)
                self.assertNotIn('PAST_CALL', rendered)
                self.assertNotIn('PAST_ANSWER', rendered)
                self.assertIn('old call', rendered)
                self.assertIn('old answer', rendered)
                self.assertIn('CURRENT_CALL', rendered)
                if mode == 'train':
                    self.assertIn('CURRENT_ANSWER', rendered)

    def test_per_request_override_takes_precedence(self):
        for preserve in (False, True):
            with self.subTest(preserve=preserve):
                rendered = self._encode(
                    'train',
                    loss_scale='all',
                    preserve_thinking=not preserve,
                    chat_template_kwargs={'preserve_thinking': preserve})
                self.assertEqual(PAST_THINK in rendered, preserve)
                self.assertIn(CURRENT_THINK, rendered)

    def test_qwen3_8_and_shared_metadata_are_unchanged(self):
        self._encode('train', loss_scale='all', preserve_thinking=True)
        rendered = self._encode('train', loss_scale='all')
        self.assertNotIn(PAST_THINK, rendered)
        rendered = self._encode('train', loss_scale='all', template_type='qwen3_8')
        self.assertIn(PAST_THINK, rendered)
        self.assertIs(TEMPLATE_MAPPING['qwen3_5'].preserve_thinking, False)
        self.assertIs(TEMPLATE_MAPPING['qwen3_8'].preserve_thinking, True)
