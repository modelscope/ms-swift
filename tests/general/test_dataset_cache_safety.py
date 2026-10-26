import hashlib
import json
import os
import tempfile
import unittest
from datasets import Dataset, Features, Value

from swift.dataset import MessagesPreprocessor, RowPreprocessor

SAMPLE = [
    {
        'messages': [{
            'role': 'user',
            'content': '1+1=?'
        }, {
            'role': 'assistant',
            'content': '2'
        }]
    },
    {
        'messages': [{
            'role': 'user',
            'content': 'say hi'
        }, {
            'role': 'assistant',
            'content': 'hi!',
            'loss': False
        }]
    },
    {
        'messages': [{
            'role': 'user',
            'content': 'poem'
        }, {
            'role': 'assistant',
            'content': 'roses',
            'loss_scale': 0.5
        }]
    },
]


def _md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def _iter_messages(row):
    """Yield message dicts whether the cache feature materializes dicts or JSON strings."""
    for message in row['messages']:
        yield json.loads(message) if isinstance(message, str) else message


class TestDatasetCacheSafety(unittest.TestCase):
    """Cache-file identity tests for the preprocessed (map_cache) arrow files.

    The map cache file name used to be `{input_fingerprint}.arrow`, so caches written by
    an older ms-swift with a different preprocessed schema (e.g. pre-#9214 `messages[].loss:
    float64` without `loss_scale`) were silently reused by a newer ms-swift: datasets does
    not validate an explicitly passed `cache_file_name`, it only checks that the file
    exists (modelscope/ms-swift#10286). `RowPreprocessor.cache_format_version` busts the
    cache the same way `Template._version` does for encoded caches.
    """

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.cache_dir = self._temp.name
        self._old_cache_dir = os.getenv('MODELSCOPE_CACHE')
        os.environ['MODELSCOPE_CACHE'] = self.cache_dir
        self.addCleanup(self._restore_cache_dir_env)

    def _restore_cache_dir_env(self):
        if self._old_cache_dir is None:
            os.environ.pop('MODELSCOPE_CACHE', None)
        else:
            os.environ['MODELSCOPE_CACHE'] = self._old_cache_dir

    def _map_cache_dir(self):
        return os.path.join(self.cache_dir, 'datasets', 'map_cache')

    def test_cache_file_name_is_versioned(self):
        dataset = Dataset.from_list(SAMPLE)
        preprocessor = MessagesPreprocessor()
        expected_suffix = f'_{preprocessor.cache_format_version}.arrow'
        cache_name = os.path.basename(preprocessor._map_cache_file(dataset._fingerprint))
        self.assertTrue(cache_name.endswith(expected_suffix), cache_name)
        self.assertNotEqual(cache_name, f'{dataset._fingerprint}.arrow')

    def test_cache_version_bump_prevents_name_collision(self):
        dataset = Dataset.from_list(SAMPLE)
        preprocessor = MessagesPreprocessor()
        current_name = os.path.basename(preprocessor._map_cache_file(dataset._fingerprint))

        class LegacyPreprocessor(RowPreprocessor):
            cache_format_version = 'v0'

        legacy_name = os.path.basename(LegacyPreprocessor()._map_cache_file(dataset._fingerprint))
        self.assertNotEqual(current_name, legacy_name)

    def test_stale_cache_from_previous_format_is_not_reused(self):
        from datasets.arrow_writer import ArrowWriter

        dataset = Dataset.from_list(SAMPLE)
        preprocessor = MessagesPreprocessor()
        map_cache_dir = self._map_cache_dir()
        os.makedirs(map_cache_dir, exist_ok=True)
        # Simulate a cache left at the legacy (unversioned) path by a previous format.
        stale_path = os.path.join(map_cache_dir, f'{dataset._fingerprint}.arrow')
        writer = ArrowWriter(features=Features({'poison': Value('string')}), path=stale_path)
        writer.write({'poison': 'stale-cache'})
        writer.finalize()
        stale_md5 = _md5(stale_path)

        result = preprocessor(dataset, num_proc=1)

        versioned_path = os.path.join(self._map_cache_dir(),
                                      f'{dataset._fingerprint}_{preprocessor.cache_format_version}.arrow')
        self.assertTrue(os.path.exists(versioned_path))
        self.assertEqual(_md5(stale_path), stale_md5)
        self.assertIn('messages', result.column_names)
        self.assertNotIn('poison', result.column_names)
        # Fresh preprocessing kept the new per-message loss metadata (see #9214).
        scaled = list(_iter_messages(result[2]))[-1]
        self.assertEqual(scaled['loss_scale'], 0.5)


if __name__ == '__main__':
    unittest.main()
