import collections
import unittest

import torch


class TestDynamicMixBatchSampler(unittest.TestCase):

    def _make_sampler(self, domain_indices=None, batch_size=4, shuffle=True,
                      drop_last=False, data_seed=42, num_batches=10):
        from swift.dataloader.shard import DynamicMixBatchSampler
        if domain_indices is None:
            domain_indices = {
                'math': list(range(0, 100)),
                'code': list(range(100, 200)),
                'general': list(range(200, 400)),
            }
        return DynamicMixBatchSampler(
            domain_indices=domain_indices,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            data_seed=data_seed,
            num_batches=num_batches,
        )

    def test_initial_probabilities_proportional_to_size(self):
        sampler = self._make_sampler()
        # math:100, code:100, general:200 => 0.25, 0.25, 0.5
        self.assertAlmostEqual(sampler.probabilities['math'], 0.25)
        self.assertAlmostEqual(sampler.probabilities['code'], 0.25)
        self.assertAlmostEqual(sampler.probabilities['general'], 0.5)

    def test_yields_correct_number_of_batches(self):
        sampler = self._make_sampler(num_batches=20, batch_size=8)
        batches = list(sampler)
        self.assertEqual(len(batches), 20)
        for batch in batches:
            self.assertEqual(len(batch), 8)

    def test_len_equals_num_batches(self):
        sampler = self._make_sampler(num_batches=15)
        self.assertEqual(len(sampler), 15)

    def test_all_indices_belong_to_domains(self):
        domain_indices = {
            'a': list(range(0, 50)),
            'b': list(range(50, 100)),
        }
        sampler = self._make_sampler(domain_indices=domain_indices, num_batches=30)
        all_valid = set(range(100))
        for batch in sampler:
            for idx in batch:
                self.assertIn(idx, all_valid)

    def test_set_probabilities_changes_distribution(self):
        sampler = self._make_sampler(num_batches=200, batch_size=8, data_seed=123)
        # Set extreme probabilities: almost all samples from 'math'
        sampler.set_probabilities({'math': 0.98, 'code': 0.01, 'general': 0.01})

        math_indices = set(range(0, 100))
        math_count = 0
        total_count = 0
        for batch in sampler:
            for idx in batch:
                total_count += 1
                if idx in math_indices:
                    math_count += 1

        math_ratio = math_count / total_count
        # With 98% probability, math should dominate
        self.assertGreater(math_ratio, 0.85)

    def test_set_probabilities_normalizes(self):
        sampler = self._make_sampler()
        sampler.set_probabilities({'math': 3.0, 'code': 1.0, 'general': 1.0})
        total = sum(sampler.probabilities.values())
        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertAlmostEqual(sampler.probabilities['math'], 0.6, places=6)

    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical batches (important for distributed consistency)."""
        sampler1 = self._make_sampler(data_seed=99, num_batches=10, batch_size=4)
        sampler2 = self._make_sampler(data_seed=99, num_batches=10, batch_size=4)
        batches1 = list(sampler1)
        batches2 = list(sampler2)
        self.assertEqual(batches1, batches2)

    def test_different_seed_different_batches(self):
        sampler1 = self._make_sampler(data_seed=1, num_batches=5)
        sampler2 = self._make_sampler(data_seed=2, num_batches=5)
        batches1 = list(sampler1)
        batches2 = list(sampler2)
        self.assertNotEqual(batches1, batches2)

    def test_set_epoch_changes_seed(self):
        sampler = self._make_sampler(data_seed=42, num_batches=5)
        batches_epoch0 = list(sampler)
        sampler.set_epoch(1)
        batches_epoch1 = list(sampler)
        self.assertNotEqual(batches_epoch0, batches_epoch1)

    def test_no_shuffle_deterministic(self):
        sampler = self._make_sampler(shuffle=False, num_batches=5)
        batches1 = list(sampler)
        batches2 = list(sampler)
        self.assertEqual(batches1, batches2)

    def test_domain_exhaustion_reshuffle(self):
        """When a small domain is exhausted, it should reshuffle and continue."""
        domain_indices = {
            'small': [0, 1],
            'large': list(range(2, 102)),
        }
        sampler = self._make_sampler(
            domain_indices=domain_indices, num_batches=50, batch_size=4)
        # Set probability heavily towards the small domain
        sampler.set_probabilities({'small': 0.9, 'large': 0.1})
        # Should not raise, even though 'small' only has 2 samples
        batches = list(sampler)
        self.assertEqual(len(batches), 50)


class TestDynamicMixingCallback(unittest.TestCase):

    def test_update_probabilities_softmax(self):
        """Verify that _update_probabilities applies softmax(loss/T) correctly."""
        from swift.callbacks.dynamic_mix import DynamicMixingCallback

        class FakeArgs:
            dynamic_mix_update_steps = 10
            dynamic_mix_temperature = 1.0
            dynamic_mix_warmup_steps = 0

        class FakeMeanMetric:
            def __init__(self):
                self.values = []

            def update(self, v):
                self.values.append(v)

        class FakeTrainer:
            custom_metrics = {
                'train': collections.defaultdict(FakeMeanMetric),
            }

        class FakeSampler:
            domain_names = ['code', 'math']
            domain_indices = {'code': list(range(50)), 'math': list(range(50, 100))}
            probabilities = {'code': 0.5, 'math': 0.5}

            def set_probabilities(self, probs):
                self.probabilities = probs

        callback = DynamicMixingCallback(FakeArgs(), FakeTrainer())
        callback._sampler = FakeSampler()
        callback._domain_names = ['code', 'math']

        # Simulate loss values: math has higher loss
        callback._loss_buffer['code'] = [1.0, 1.0]
        callback._loss_buffer['math'] = [3.0, 3.0]

        callback._update_probabilities(global_step=10)

        # math (loss=3) should get higher probability than code (loss=1)
        self.assertGreater(
            callback._sampler.probabilities['math'],
            callback._sampler.probabilities['code'])

        # With T=1: softmax([1,3]) = [exp(1)/(exp(1)+exp(3)), exp(3)/(exp(1)+exp(3))]
        expected_math = torch.softmax(torch.tensor([1.0, 3.0]), dim=0)[1].item()
        self.assertAlmostEqual(
            callback._sampler.probabilities['math'], expected_math, places=5)

    def test_high_temperature_more_uniform(self):
        """Higher temperature should produce more uniform distribution."""
        from swift.callbacks.dynamic_mix import DynamicMixingCallback

        class FakeArgs:
            dynamic_mix_update_steps = 10
            dynamic_mix_temperature = 100.0  # Very high T
            dynamic_mix_warmup_steps = 0

        class FakeMeanMetric:
            def update(self, v):
                pass

        class FakeTrainer:
            custom_metrics = {
                'train': collections.defaultdict(FakeMeanMetric),
            }

        class FakeSampler:
            domain_names = ['a', 'b']
            domain_indices = {'a': list(range(50)), 'b': list(range(50, 100))}
            probabilities = {'a': 0.5, 'b': 0.5}

            def set_probabilities(self, probs):
                self.probabilities = probs

        callback = DynamicMixingCallback(FakeArgs(), FakeTrainer())
        callback._sampler = FakeSampler()
        callback._domain_names = ['a', 'b']

        callback._loss_buffer['a'] = [1.0]
        callback._loss_buffer['b'] = [10.0]

        callback._update_probabilities(global_step=10)

        # With very high T, both should be close to 0.5
        self.assertAlmostEqual(
            callback._sampler.probabilities['a'], 0.5, places=1)
        self.assertAlmostEqual(
            callback._sampler.probabilities['b'], 0.5, places=1)


if __name__ == '__main__':
    unittest.main()
