# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import numpy as np
import os
import tempfile
import unittest
from datasets import Dataset as HfDataset
from unittest import mock

from swift.dataset import DatasetMeta, dataset_syntax, load_dataset, register_dataset
from swift.dataset.dataset_meta import DATASET_MAPPING
from swift.dataset.loader import DatasetLoader
from swift.dataset.preprocessor.decision import DecisionAugmentConfig, JevPreprocessor
from swift.utils import get_seed


class TestDatasetRng(unittest.TestCase):

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = os.path.join(directory.name, 'messages.jsonl')
        self.rows = [{
            'messages': [{
                'role': 'user',
                'content': f'question {i}'
            }, {
                'role': 'assistant',
                'content': f'answer {i}'
            }]
        } for i in range(20)]
        with open(self.path, 'w', encoding='utf-8') as stream:
            for row in self.rows:
                stream.write(json.dumps(row) + '\n')
        patch = mock.patch.dict(os.environ, {'MODELSCOPE_CACHE': os.path.join(directory.name, 'cache')})
        patch.start()
        self.addCleanup(patch.stop)

    def _reference(self, random_state, split_ratio, shuffle):
        train, val = DatasetLoader.post_process(
            HfDataset.from_list(self.rows), split_dataset_ratio=split_ratio, shuffle=shuffle, random_state=random_state)
        if shuffle:
            train = DatasetLoader.shuffle_dataset(train, seed=get_seed(random_state))
            if val is not None:
                val = DatasetLoader.shuffle_dataset(val, seed=get_seed(random_state))
        return train, val

    def test_disabled_augmentation_preserves_split_shuffle_and_rng(self):
        for split_ratio, shuffle in [(0., False), (0., True), (0.2, False), (0.2, True)]:
            with self.subTest(split_ratio=split_ratio, shuffle=shuffle):
                reference_rng = np.random.RandomState(42)
                expected_train, expected_val = self._reference(reference_rng, split_ratio, shuffle)
                actual_rng = np.random.RandomState(42)
                train, val = load_dataset(
                    self.path,
                    split_dataset_ratio=split_ratio,
                    shuffle=shuffle,
                    seed=actual_rng,
                    num_proc=1,
                    strict=True,
                    load_from_cache_file=False)

                self.assertEqual(list(train['messages']), list(expected_train['messages']))
                if expected_val is None:
                    self.assertIsNone(val)
                else:
                    self.assertEqual(list(val['messages']), list(expected_val['messages']))
                np.testing.assert_array_equal(
                    actual_rng.randint(0, 1000, size=10), reference_rng.randint(0, 1000, size=10))

    def test_enabled_augmentation_keeps_its_seed_and_behavior(self):
        path = os.path.join(os.path.dirname(self.path), 'decision.jsonl')
        with open(path, 'w', encoding='utf-8') as stream:
            for i in range(20):
                stream.write(
                    json.dumps({
                        'state':
                        f'case {i}',
                        'questions': [{
                            'kind': 'choice',
                            'question': 'Choose an option',
                            'options': ['a', 'b', 'c'],
                            'gold': 0
                        }]
                    }) + '\n')
        config = DecisionAugmentConfig(drop_instructions=0.5)
        expected_rng = np.random.RandomState(42)
        get_seed(expected_rng)
        actual_rng = np.random.RandomState(42)
        with mock.patch.dict(DATASET_MAPPING), mock.patch.object(dataset_syntax, '_dataset_meta_mapping', None):
            register_dataset(DatasetMeta(dataset_path=path, preprocess_func=JevPreprocessor()), exist_ok=True)
            train, val = load_dataset(
                path,
                seed=actual_rng,
                shuffle=False,
                num_proc=1,
                strict=True,
                load_from_cache_file=False,
                decision_augment_config=config)
        self.assertIsNone(val)
        self.assertEqual(len(train), 20)
        questions = [row['questions'][0] for row in train]
        self.assertIn('', questions)
        self.assertIn('Choose an option', questions)
        np.testing.assert_array_equal(actual_rng.randint(0, 1000, size=10), expected_rng.randint(0, 1000, size=10))


if __name__ == '__main__':
    unittest.main()
