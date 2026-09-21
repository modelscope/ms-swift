# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
from datasets import Dataset
from torch.utils.data import DataLoader

from swift.dataset import LazyLLMDataset
from swift.template import MaxLengthError


class TestLazyDatasetRetries(unittest.TestCase):

    def _dataset(self, valid, error=ValueError, **kwargs):
        self.attempts = []

        def encode(row, return_length=False):
            self.assertTrue(return_length)
            self.attempts.append(row['id'])
            if not row['valid']:
                raise error('invalid training row')
            return {'id': row['id']}

        data = Dataset.from_dict({'id': list(range(len(valid))), 'valid': valid})
        return LazyLLMDataset(data, encode, traceback_limit=0, **kwargs)

    def test_retry_reaches_valid_row(self):
        for error in [ValueError, MaxLengthError]:
            for index in [0, -2]:
                with self.subTest(error=error.__name__, index=index):
                    data = self._dataset([False, True], error=error, random_state=1)
                    self.assertEqual(data[index], {'id': 1})
                    self.assertEqual(self.attempts, [0, 1])

    def test_last_fallback_is_not_skipped(self):
        data = self._dataset([True, False, False], random_state=0)
        self.assertEqual(data[2], {'id': 0})
        self.assertEqual(len(self.attempts), 3)
        self.assertEqual(set(self.attempts), {0, 1, 2})

    def test_retry_cursor_wraps_without_repeating_initial_row(self):
        data = self._dataset([False, True], random_state=1)
        for _ in range(4):
            self.attempts.clear()
            self.assertEqual(data[0], {'id': 1})
            self.assertEqual(self.attempts, [0, 1])

    def test_exhaustion_and_attempt_limit(self):
        for count in [1, 2, 3]:
            with self.subTest(count=count):
                data = self._dataset([False] * 3, random_state=0, n_try_fetch=count)
                with self.assertRaisesRegex(ValueError, 'Failed to retrieve'):
                    data[2]
                self.assertEqual(len(self.attempts), count)
                self.assertEqual(len(set(self.attempts)), count)
        data = self._dataset([False], random_state=0)
        with self.assertRaisesRegex(ValueError, 'Failed to retrieve'):
            data[0]
        self.assertEqual(self.attempts, [0])

    def test_strict_and_valid_first_attempt(self):
        data = self._dataset([False, True], random_state=1, strict=True)
        with self.assertRaisesRegex(ValueError, 'invalid training row'):
            data[0]
        self.assertEqual(self.attempts, [0])
        self.attempts.clear()
        self.assertEqual(data[1], {'id': 1})
        self.assertEqual(self.attempts, [1])
        self.assertEqual(data['id'], [0, 1])

    def test_dataloader_can_skip_bad_rows(self):
        data = self._dataset([False, True], random_state=1)
        loader = DataLoader(data, batch_size=2, num_workers=0)
        self.assertEqual(next(iter(loader))['id'].tolist(), [1, 1])


if __name__ == '__main__':
    unittest.main()
