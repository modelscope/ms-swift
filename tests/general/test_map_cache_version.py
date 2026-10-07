# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import unittest
from datasets import Dataset as HfDataset
from tempfile import TemporaryDirectory
from unittest.mock import patch

from swift.dataset.preprocessor import RowPreprocessor


class _VersionedPreprocessor(RowPreprocessor):
    """Simulates a preprocessing schema that changed between ms-swift versions."""

    def __init__(self, *, schema_version: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.schema_version = schema_version

    def preprocess(self, row):
        row = dict(row)
        if self.schema_version >= 2:
            # e.g. the routing `dataset` column added after 4.0.0
            row['dataset'] = 'new-schema-column'
        return row


ROWS = [{'messages': [{'role': 'user', 'content': 'hello'}]}]


class TestMapCacheVersion(unittest.TestCase):

    def _preprocess(self, cache_dir, *, version, schema_version):
        dataset = HfDataset.from_list([dict(row) for row in ROWS])
        preprocessor = _VersionedPreprocessor(schema_version=schema_version)
        with patch('swift.dataset.preprocessor.core.get_cache_dir', return_value=cache_dir), \
                patch('swift.dataset.preprocessor.core.__version__', version, create=True):
            return preprocessor(dataset, num_proc=1, strict=False)

    @staticmethod
    def _cache_files(cache_dir):
        map_cache_dir = os.path.join(cache_dir, 'datasets', 'map_cache')
        if not os.path.isdir(map_cache_dir):
            return []
        return sorted(f for f in os.listdir(map_cache_dir) if f.endswith('.arrow'))

    def test_cache_of_another_version_is_not_reused(self):
        with TemporaryDirectory() as cache_dir:
            old = self._preprocess(cache_dir, version='4.0.0', schema_version=1)
            self.assertNotIn('dataset', old.column_names)
            self.assertEqual(len(self._cache_files(cache_dir)), 1)

            # The raw dataset (`dataset._fingerprint`) is unchanged; only the
            # preprocessing schema of the newer version changed. The cache written by
            # the old version must not be replayed.
            new = self._preprocess(cache_dir, version='4.4.0', schema_version=2)
            self.assertIn('dataset', new.column_names)
            self.assertEqual(len(self._cache_files(cache_dir)), 2)

    def test_cache_of_the_same_version_is_reused(self):
        with TemporaryDirectory() as cache_dir:
            for _ in range(2):
                dataset = self._preprocess(cache_dir, version='4.0.0', schema_version=1)
                self.assertEqual(dataset.column_names, ['messages'])
            self.assertEqual(len(self._cache_files(cache_dir)), 1)


if __name__ == '__main__':
    unittest.main()
