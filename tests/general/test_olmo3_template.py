# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
from copy import deepcopy
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from types import SimpleNamespace

from swift.template import get_template

# The text-only branch of the official Think chat template, including its
# different historical/final assistant terminators. No Hub access is needed.
# https://huggingface.co/allenai/Olmo-3-7B-Think/blob/main/chat_template.jinja
CHAT_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are OLMo, a helpful function-calling AI assistant built by Ai2. "
    'Your date cutoff is November 2024, and your model weights are available at '
    'https://huggingface.co/allenai. You do not currently have access to any functions. '
    "<functions></functions><|im_end|>\n' }}{% endif %}"
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] }}"
    "{% if message['role'] == 'system' %}"
    "{{ ' You do not currently have access to any functions. <functions></functions>' }}{% endif %}"
    "{% if message['role'] == 'assistant' and loop.last %}{{ eos_token }}"
    "{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>' }}{% endif %}")


class TestOlmo3ThinkTemplate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Preserve every byte and whitespace boundary with a tiny local tokenizer.
        vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        cls.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token='<|endoftext|>',
            eos_token='<|endoftext|>',
            pad_token='<|pad|>',
            additional_special_tokens=['<|im_start|>', '<|im_end|>'],
            chat_template=CHAT_TEMPLATE)
        cls.processor = SimpleNamespace(
            tokenizer=cls.tokenizer,
            model_info=SimpleNamespace(config={}, task_type='causal_lm', max_model_len=8192),
            model_meta=SimpleNamespace(is_multimodal=False))

    def make_template(self, mode='train', **kwargs):
        template = get_template(self.processor, template_type='olmo3_think', **kwargs)
        template.set_mode(mode)
        return template

    def assert_matches_hf(self, messages, mode='train'):
        original = deepcopy(messages)
        encoded = self.make_template(mode).encode({'messages': messages})
        expected = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=mode == 'transformers')
        self.assertEqual(self.tokenizer.decode(encoded['input_ids']), expected)
        self.assertEqual(messages, original)
        return encoded

    def test_default_and_explicit_system(self):
        conversation = [{'role': 'user', 'content': 'Question?'},
                        {'role': 'assistant', 'content': '<think>reason</think>\nAnswer.'}]
        for system in [None, '', 'Answer in Chinese.']:
            with self.subTest(system=system):
                messages = deepcopy(conversation)
                if system is not None:
                    messages.insert(0, {'role': 'system', 'content': system})
                encoded = self.assert_matches_hf(messages)
                self.assertEqual(encoded['input_ids'][-1], self.tokenizer.eos_token_id)
                self.assertNotEqual(encoded['input_ids'][0], self.tokenizer.bos_token_id)

    def test_multiround_preserves_thinking_and_uses_distinct_terminators(self):
        messages = [{'role': 'user', 'content': 'First question'},
                    {'role': 'assistant', 'content': '<think>historical reasoning</think>\nFirst answer'},
                    {'role': 'user', 'content': 'Second question'},
                    {'role': 'assistant', 'content': '<think>new reasoning</think>\nSecond answer'}]
        encoded = self.assert_matches_hf(messages)
        text = self.tokenizer.decode(encoded['input_ids'])
        self.assertIn('historical reasoning', text)
        self.assertIn('First answer<|im_end|>\n<|im_start|>user', text)
        self.assertTrue(text.endswith('Second answer<|endoftext|>'))

    def test_inference_prefix_has_no_extra_newline(self):
        messages = [{'role': 'user', 'content': 'First question'},
                    {'role': 'assistant', 'content': '<think>historical reasoning</think>\nFirst answer'},
                    {'role': 'user', 'content': 'Next question'}]
        encoded = self.assert_matches_hf(messages, mode='transformers')
        self.assertTrue(self.tokenizer.decode(encoded['input_ids']).endswith('<|im_start|>assistant\n<think>'))

    def test_prompt_mask_and_assistant_eos_loss(self):
        answer = '<think>reason</think>\nAnswer.'
        encoded = self.assert_matches_hf(
            [{'role': 'user', 'content': 'Question?'}, {'role': 'assistant', 'content': answer}])
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        prompt_len = len(encoded['input_ids']) - len(answer_ids)
        self.assertEqual(encoded['labels'][:prompt_len], [-100] * prompt_len)
        self.assertEqual(encoded['labels'][prompt_len:], answer_ids)

    def test_rejects_tool_protocols(self):
        user = {'role': 'user', 'content': 'Question?'}
        for unsupported in [
                {'tools': [{'type': 'function', 'function': {'name': 'weather', 'parameters': {}}}]},
                {'functions': [{'name': 'weather'}]},
                {'messages': [user, {'role': 'tool_call', 'content': '{"name": "weather"}'}]},
                {'messages': [user, {'role': 'environment', 'content': 'result'}]},
                {'messages': [dict(user, functions='[]')]},
                {'messages': [user, {'role': 'assistant', 'content': '', 'function_calls': 'weather()'}]},
        ]:
            inputs = {'messages': [user]}
            inputs.update(unsupported)
            with self.subTest(inputs=inputs), self.assertRaisesRegex(ValueError, 'without tools'):
                self.make_template().encode(inputs)

    def test_rejects_multimodal_inputs_before_loading_media(self):
        for media_type in ['images', 'videos', 'audios']:
            with self.subTest(media_type=media_type), self.assertRaisesRegex(ValueError, 'only supports text'):
                self.make_template().encode({
                    'messages': [{'role': 'user', 'content': 'Describe this.'}],
                    media_type: ['does-not-exist'],
                })

    def test_disables_unsupported_hybrid_attention_training_paths(self):
        # SFT checks this flag for both --padding_free and --packing.
        self.assertFalse(self.make_template().support_padding_free)
        with self.assertRaisesRegex(ValueError, 'sequence_parallel_size > 1'):
            self.make_template(sequence_parallel_size=2)


if __name__ == '__main__':
    unittest.main()
