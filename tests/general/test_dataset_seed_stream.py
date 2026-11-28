import unittest
from unittest.mock import patch

import numpy as np
from datasets import Dataset as HfDataset

from swift.dataset import DatasetMeta, DatasetSyntax, load_dataset
from swift.dataset.loader import DatasetLoader
from swift.dataset.preprocessor.decision import DecisionAugmentConfig


def _make_dataset() -> HfDataset:
    return HfDataset.from_dict({
        'messages': [[{
            'role': 'user',
            'content': f'question {i}',
        }, {
            'role': 'assistant',
            'content': f'answer {i}',
        }] for i in range(20)]
    })


class RecordingRandomState(np.random.RandomState):
    """RandomState that records every randint draw, in call order."""

    def __init__(self, seed, calls):
        super().__init__(seed)
        self.calls = calls

    def randint(self, *args, **kwargs):
        self.calls.append('randint')
        return super().randint(*args, **kwargs)


class TestDecisionAugmentSeedStream(unittest.TestCase):

    def _load(self, calls, decision_augment_config=None):
        original_post_process = DatasetLoader.post_process

        def recording_post_process(loader, dataset, **kwargs):
            calls.append('post_process')
            return original_post_process(dataset, **kwargs)

        def fake_load_repo_dataset(loader, dataset_id, subset, *, use_hf=None, revision=None):
            return _make_dataset()

        with patch.object(DatasetSyntax, 'get_dataset_meta', return_value=DatasetMeta()), \
                patch.object(DatasetLoader, '_load_repo_dataset', new=fake_load_repo_dataset), \
                patch.object(DatasetLoader, 'post_process', new=recording_post_process):
            train_dataset, val_dataset = load_dataset(
                'ms::example-org/dummy',
                seed=RecordingRandomState(42, calls),
                shuffle=True,
                split_dataset_ratio=0.5,
                decision_augment_config=decision_augment_config,
            )
        self.assertEqual(len(train_dataset) + (len(val_dataset) or 0), 20)
        return train_dataset, val_dataset

    def test_augment_disabled_does_not_consume_seed_stream(self):
        # get_seed(seed) used to be evaluated eagerly even when decision augmentation was
        # disabled, shifting every downstream consumer (train/val split, shuffle) by one draw.
        calls = []
        self._load(calls)
        # The very first draw must belong to post_process, not to the augment seeding.
        self.assertEqual(calls[0], 'post_process')

    def test_augment_enabled_still_seeds_before_post_process(self):
        calls = []
        self._load(calls, decision_augment_config=DecisionAugmentConfig())
        self.assertEqual(calls[0], 'randint')
        self.assertEqual(calls[1], 'post_process')

    def test_augment_disabled_split_is_stable_across_calls(self):
        calls_a, calls_b = [], []
        train_a, val_a = self._load(calls_a)
        train_b, val_b = self._load(calls_b)
        self.assertEqual(list(train_a['messages']), list(train_b['messages']))
        self.assertEqual(list(val_a['messages']), list(val_b['messages']))


if __name__ == '__main__':
    unittest.main()
