# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
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

    def _encode(self, mode, **kwargs):
        template = get_template(self.processor, template_type='qwen3_5', **kwargs)
        template.set_mode(mode)
        encoded = template.encode({'messages': [dict(message) for message in MESSAGES]})
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
        rendered = self._encode('vllm')
        self.assertNotIn(PAST_THINK, rendered)

    def test_explicit_opt_in_still_keeps_past_turn_reasoning(self):
        # Users who want to train on the full reasoning history can still opt in explicitly.
        rendered = self._encode('train', loss_scale='all', preserve_thinking=True)
        self.assertIn(PAST_THINK, rendered)
        self.assertIn(CURRENT_THINK, rendered)
