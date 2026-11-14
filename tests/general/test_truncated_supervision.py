import unittest
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from types import SimpleNamespace
from unittest.mock import Mock

from swift.dataset import EncodePreprocessor
from swift.dataset.packing import PackingDataset
from swift.dataset.utils import AddLengthPreprocessor, LazyLLMDataset
from swift.pipelines.train.sft import SwiftSft
from swift.template import MaxLengthError, get_template


def make_template(**kwargs):
    tokens = [
        '[UNK]', '[PAD]', '<|eot_id|>', '<|begin_of_text|>', '<|start_header_id|>', '<|end_header_id|>', 'question',
        'answer', 'user', 'assistant'
    ]
    backend = Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token='[UNK]',
        pad_token='[PAD]',
        eos_token='<|eot_id|>',
        additional_special_tokens=tokens[3:6])
    tokenizer.model_info = SimpleNamespace(config=SimpleNamespace(), task_type='causal_lm', max_model_len=512)
    tokenizer.model_meta = SimpleNamespace(is_multimodal=False)
    template = get_template(tokenizer, template_type='llama3', max_length=32, truncation_strategy='right', **kwargs)
    template.set_mode('train')
    return template


def sample(words):
    return {
        'messages': [{
            'role': 'user',
            'content': 'question ' * words
        }, {
            'role': 'assistant',
            'content': 'answer ' * 4
        }]
    }


class TruncatedSupervisionTest(unittest.TestCase):

    def setUp(self):
        self.template = make_template()
        self.long = sample(64)
        self.short = sample(1)

    def test_right_truncation_rejects_only_fully_removed_targets(self):
        with self.assertRaisesRegex(MaxLengthError, 'all supervised tokens'):
            self.template.encode(self.long)
        self.assertTrue(any(x != -100 for x in self.template.encode(self.short)['labels']))
        self.template.max_length = 512
        full = self.template.encode(self.long)
        first = next(i for i, x in enumerate(full['labels']) if x != -100)
        self.template.max_length = first + 1
        partial = self.template.encode(self.long)
        self.assertEqual(sum(x != -100 for x in partial['labels']), 1)
        self.assertEqual(partial['input_ids'], full['input_ids'][:first + 1])

    def test_left_truncation_preserves_targets(self):
        self.template.truncation_strategy = 'left'
        encoded = self.template.encode(self.long)
        self.assertEqual(len(encoded['input_ids']), 32)
        self.assertTrue(any(x != -100 for x in encoded['labels']))

    def test_prompt_training_still_has_supervision(self):
        self.template = make_template(loss_scale='all')
        encoded = self.template.encode(self.long)
        self.assertTrue(any(x != -100 for x in encoded['labels']))

    def test_inference_and_intentionally_masked_responses_are_not_rejected(self):
        self.template.set_mode('transformers')
        encoded = self.template.encode({'messages': self.long['messages'][:1]})
        self.assertEqual(len(encoded['input_ids']), 32)
        self.assertNotIn('labels', encoded)
        self.template.set_mode('train')
        # Explicitly all-masked raw responses are used outside ordinary SFT.
        row = sample(64)
        row['add_eos'] = False
        row['messages'][-1]['content'] = {'token_ids': [6, 7], 'loss_scale': [0, 0]}
        encoded = self.template.encode(row)
        self.assertTrue(all(x == -100 for x in encoded['labels']))

    def test_classification_and_preference_encoding_keep_existing_behavior(self):
        self.template.task_type = 'seq_cls'
        self.template.config.problem_type = 'single_label_classification'
        row = sample(64)
        row['label'] = 1
        self.assertEqual(self.template.encode(row)['labels'], 1)
        self.template.task_type = 'causal_lm'
        self.template.set_mode('rlhf')
        row = sample(64)
        row['rejected_response'] = 'answer'
        encoded = self.template.encode(row)
        self.assertIn('chosen_input_ids', encoded)
        self.assertIn('rejected_input_ids', encoded)

    def test_preprocessing_filters_and_strict_mode_raises(self):
        data = Dataset.from_list([self.long, self.short])
        for cls in (AddLengthPreprocessor, EncodePreprocessor):
            with self.subTest(preprocessor=cls.__name__):
                encoded = cls(self.template)(data, num_proc=1, load_from_cache_file=False, strict=False)
                self.assertEqual(len(encoded), 1)
                with self.assertRaisesRegex(MaxLengthError, 'all supervised tokens'):
                    cls(self.template)(data, num_proc=1, load_from_cache_file=False, strict=True)

    def test_streaming_filters_and_lazy_encoding_resamples(self):
        data = Dataset.from_list([self.long, self.short])
        encoded = EncodePreprocessor(self.template)(data.to_iterable_dataset(), strict=False)
        self.assertEqual(len(list(encoded)), 1)
        for seed in (0, 1):
            with self.subTest(seed=seed):
                lazy = LazyLLMDataset(data, self.template.encode, random_state=seed)
                self.assertTrue(any(x != -100 for x in lazy[0]['labels']))
        with self.assertRaisesRegex(MaxLengthError, 'all supervised tokens'):
            LazyLLMDataset(data, self.template.encode, strict=True)[0]

    def test_packing_keeps_supervision_after_filtering(self):
        data = Dataset.from_list([self.long, self.short, self.short])
        filtered = AddLengthPreprocessor(self.template)(data, num_proc=1, load_from_cache_file=False, strict=False)
        lazy = LazyLLMDataset(filtered, self.template.encode, strict=True)
        packed = PackingDataset(self.template, lazy, packing_length=128, packing_strategy='sequential')
        self.assertEqual(len(packed), 1)
        self.assertEqual(len(packed[0]), 2)
        expected_labels = lazy[0]['labels'] + lazy[1]['labels']
        batch = self.template.data_collator([packed[0]])
        self.assertEqual(batch['labels'].tolist(), [expected_labels])
        self.assertTrue((batch['labels'] != -100).any())

    def test_empty_streaming_training_dataset_has_clear_error(self):
        pipeline = SwiftSft.__new__(SwiftSft)
        pipeline.args = SimpleNamespace(streaming=True, lazy_tokenize=False)
        pipeline.template = self.template
        dataset = Dataset.from_list([self.long]).to_iterable_dataset()
        encoded = EncodePreprocessor(self.template)(dataset, strict=False)
        with self.assertRaisesRegex(ValueError, 'No valid training samples'):
            pipeline._show_dataset(encoded, None)

    def test_streaming_preview_preserves_valid_samples(self):
        pipeline = SwiftSft.__new__(SwiftSft)
        pipeline.args = SimpleNamespace(streaming=True, lazy_tokenize=False)
        pipeline.template = self.template
        dataset = Dataset.from_list([self.long, self.short]).to_iterable_dataset()
        encoded = EncodePreprocessor(self.template)(dataset, strict=False)
        pipeline._show_dataset(encoded, None)
        remaining = list(encoded)
        self.assertEqual(len(remaining), 1)
        self.assertTrue(any(x != -100 for x in remaining[0]['labels']))

    def test_empty_filtered_training_dataset_has_clear_error(self):
        pipeline = SwiftSft.__new__(SwiftSft)
        pipeline.args = SimpleNamespace(
            cached_dataset=[], cached_val_dataset=[], dataset=['local'], val_dataset=[], truncation_strategy='right')
        pipeline._get_dataset = Mock(return_value=(Dataset.from_list([self.long]), None))
        # Exercise the actual preprocessing result through the pipeline's empty-data guard.
        filtered = AddLengthPreprocessor(self.template)(
            Dataset.from_list([self.long]), num_proc=1, load_from_cache_file=False, strict=False)
        self.assertEqual(len(filtered), 0)
        pipeline._encode_dataset = Mock(return_value=(None, None))
        with self.assertRaisesRegex(ValueError, 'No valid training samples'):
            pipeline._prepare_dataset()


if __name__ == '__main__':
    unittest.main()
