import os
import unittest
from datasets import Dataset as HfDataset
from tempfile import TemporaryDirectory
from unittest.mock import patch

from swift.dataset.preprocessor import RowPreprocessor


class _StrictRowPreprocessor(RowPreprocessor):
    """Drops a row marked invalid, and raises on it when ``strict`` is set."""

    INVALID = 'DROP_ME'

    def preprocess(self, row):
        messages = row.get('messages') or []
        if any(message.get('content') == self.INVALID for message in messages):
            raise ValueError(f'invalid row: {messages}')
        return {'messages': messages}


_MESSAGE_HELLO = {'role': 'user', 'content': 'hello'}
_MESSAGE_INVALID = {'role': 'user', 'content': _StrictRowPreprocessor.INVALID}
ROWS = [{'messages': [_MESSAGE_HELLO]}, {'messages': [_MESSAGE_INVALID]}]


class TestMapCacheFileName(unittest.TestCase):

    def _preprocess(self, cache_dir, strict, enable_auto_mapping=False):
        dataset = HfDataset.from_list([dict(row) for row in ROWS])
        preprocessor = _StrictRowPreprocessor()
        with patch('swift.dataset.preprocessor.core.get_cache_dir', return_value=cache_dir):
            return preprocessor(dataset, num_proc=1, strict=strict, enable_auto_mapping=enable_auto_mapping)

    def _cache_files(self, cache_dir):
        map_cache_dir = os.path.join(cache_dir, 'datasets', 'map_cache')
        if not os.path.isdir(map_cache_dir):
            return []
        return sorted(f for f in os.listdir(map_cache_dir) if f.endswith('.arrow'))

    def test_changing_the_map_arguments_reruns_the_map(self):
        with TemporaryDirectory() as cache_dir:
            # strict=False: the invalid row is dropped and the result is cached.
            dataset = self._preprocess(cache_dir, strict=False)
            self.assertEqual(len(dataset), 1)

            # strict=True must raise instead of replaying the cached, already filtered result.
            with self.assertRaises(ValueError):
                self._preprocess(cache_dir, strict=True)

    def test_the_same_arguments_reuse_the_cache(self):
        with TemporaryDirectory() as cache_dir:
            dataset = self._preprocess(cache_dir, strict=False)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(self._cache_files(cache_dir)), 1)

            dataset = self._preprocess(cache_dir, strict=False)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(self._cache_files(cache_dir)), 1)

    def test_every_map_argument_is_part_of_the_cache_file_name(self):
        dataset = HfDataset.from_list([dict(row) for row in ROWS])
        preprocessor = _StrictRowPreprocessor()

        def cache_file_name(**kwargs):
            return preprocessor._get_map_cache_file_name(dataset, **kwargs)

        names = {
            cache_file_name(strict=False, enable_auto_mapping=False),
            cache_file_name(strict=False, enable_auto_mapping=True),
            cache_file_name(strict=True, enable_auto_mapping=False),
            cache_file_name(strict=True, enable_auto_mapping=True),
        }
        self.assertEqual(len(names), 4)

    def test_the_column_mapping_is_part_of_the_cache_file_name(self):
        dataset = HfDataset.from_list([dict(row) for row in ROWS])
        default = _StrictRowPreprocessor()._get_map_cache_file_name(dataset, False, False)
        mapped = _StrictRowPreprocessor(columns={'query': 'messages'})._get_map_cache_file_name(dataset, False, False)
        self.assertNotEqual(default, mapped)


if __name__ == '__main__':
    unittest.main()
