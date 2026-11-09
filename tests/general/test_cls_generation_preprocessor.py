# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
from datasets import Dataset

from swift.dataset.preprocessor import ClsGenerationPreprocessor


class TestClsGenerationPreprocessor(unittest.TestCase):

    def test_literal_prompt_text(self):
        cases = [
            ('Choose {negative, positive}', ['negative', 'positive']),
            ('Classify {"text": "review"}', ['negative', 'positive']),
            ('Keep {{task}}', ['negative', 'positive']),
            ('Classification', ['{"kind": "negative"}', '{positive}']),
            ('Classification', ['{{negative}}', '{{positive}}']),
            ('Sentiment Classification', ['negative', 'positive']),
            ('', ['negative', 'positive']),
        ]
        for task, labels in cases:
            for pair in [False, True]:
                for strict in [False, True]:
                    with self.subTest(task=task, labels=labels, pair=pair, strict=strict):
                        row = {
                            'sentence1': 'First {{literal}}',
                            'sentence2': 'Second {literal}',
                            'label': 1
                        } if pair else {
                            'sentence': 'Review {{literal}} {"x": 1}',
                            'label': 1
                        }
                        inputs = 'Sentence1: First {{literal}}\nSentence2: Second {literal}' if pair else (
                            'Sentence: Review {{literal}} {"x": 1}')
                        preprocessor = ClsGenerationPreprocessor(labels, task=task, is_pair_seq=pair)
                        output = preprocessor(Dataset.from_list([row]), strict=strict, load_from_cache_file=False)
                        self.assertEqual(len(output), 1)
                        self.assertEqual(output[0]['messages'], [
                            {
                                'role': 'user',
                                'content': f'Task: {task}\n{inputs}\nCategory: {", ".join(labels)}\nOutput:'
                            },
                            {
                                'role': 'assistant',
                                'content': labels[1]
                            },
                        ])

    def test_missing_label(self):
        preprocessor = ClsGenerationPreprocessor(['negative', 'positive'], task='Classification')
        self.assertIsNone(preprocessor.preprocess({'sentence': 'Review'}))

    def test_unknown_label(self):
        preprocessor = ClsGenerationPreprocessor(['negative', 'positive'], task='Classification')
        dataset = Dataset.from_list([{'sentence': 'Review', 'label': 2}])
        with self.assertRaises(IndexError):
            preprocessor(dataset, strict=True, load_from_cache_file=False)
        self.assertEqual(len(preprocessor(dataset, load_from_cache_file=False)), 0)

    def test_task_remains_required(self):
        with self.assertRaises(TypeError):
            ClsGenerationPreprocessor(['negative', 'positive'])


if __name__ == '__main__':
    unittest.main()
