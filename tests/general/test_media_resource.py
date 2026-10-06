# Copyright (c) ModelScope Contributors. All rights reserved.
import io
import os
import shutil
import tarfile
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from swift.dataset import MediaResource


class TestMediaResource(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.cache = self.root / 'media_cache'
        self.cache_patch = patch.object(MediaResource, 'cache_dir', str(self.cache))
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        self.env_patch = patch.dict(os.environ, {'MODELSCOPE_CACHE': str(self.root / 'modelscope')})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def _archive(self, name, files):
        path = self.root / name
        with tarfile.open(path, 'w') as archive:
            for filename, text in files.items():
                data = text.encode('utf-8')
                info = tarfile.TarInfo(filename)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return str(path)

    def test_failed_shard_is_not_published_and_retry_recovers_all_files(self):
        first = self._archive('first.tar', {'nested/first.txt': 'first'})
        second = str(self.root / 'second.tar')
        with self.assertRaises(FileNotFoundError):
            MediaResource.download([first, second], 'videos', file_type='sharded')
        self.assertFalse((self.cache / 'videos').exists())
        # The attempt's private staging directory must also have been cleaned.
        self.assertFalse(list(self.cache.glob('tmp*')))
        self._archive('second.tar', {'nested/second.txt': 'second'})
        result = Path(MediaResource.download([first, second], 'videos', file_type='sharded'))
        self.assertEqual((result / 'nested/first.txt').read_text(), 'first')
        self.assertEqual((result / 'nested/second.txt').read_text(), 'second')

    def test_completed_cache_is_reused_without_source_archives(self):
        first = self._archive('first.tar', {'first.txt': 'first'})
        second = self._archive('second.tar', {'second.txt': 'second'})
        result = MediaResource.download([first, second], 'videos', file_type='sharded')
        os.remove(first)
        os.remove(second)
        self.assertEqual(MediaResource.download([first, second], 'videos', file_type='sharded'), result)
        self.assertEqual((Path(result) / 'second.txt').read_text(), 'second')

    def test_move_failure_can_retry_consumed_extraction_cache(self):
        first = self._archive('first.tar', {'first.txt': 'first'})
        second = self._archive('second.tar', {'second.txt': 'second'})
        move = shutil.move

        def fail_second(source, destination):
            if Path(source).name == 'second.txt':
                raise OSError('interrupted move')
            return move(source, destination)

        with patch('swift.dataset.media.shutil.move', side_effect=fail_second):
            with self.assertRaisesRegex(OSError, 'interrupted move'):
                MediaResource.download([first, second], 'videos', file_type='sharded')
        self.assertFalse((self.cache / 'videos').exists())
        result = Path(MediaResource.download([first, second], 'videos', file_type='sharded'))
        self.assertEqual((result / 'first.txt').read_text(), 'first')
        self.assertEqual((result / 'second.txt').read_text(), 'second')

    def test_publish_failure_can_retry_all_consumed_shards(self):
        first = self._archive('first.tar', {'first.txt': 'first'})
        second = self._archive('second.tar', {'second.txt': 'second'})
        rename = os.rename

        def fail_publish(source, destination):
            if str(destination) == str(self.cache / 'videos'):
                raise OSError('interrupted publish')
            return rename(source, destination)

        with patch('swift.dataset.media.os.rename', side_effect=fail_publish):
            with self.assertRaisesRegex(OSError, 'interrupted publish'):
                MediaResource.download([first, second], 'videos', file_type='sharded')
        self.assertFalse((self.cache / 'videos').exists())
        result = Path(MediaResource.download([first, second], 'videos', file_type='sharded'))
        self.assertEqual((result / 'first.txt').read_text(), 'first')
        self.assertEqual((result / 'second.txt').read_text(), 'second')

    def test_concurrent_downloads_reuse_completed_cache(self):
        first = self._archive('first.tar', {'first.txt': 'first'})
        second = self._archive('second.tar', {'second.txt': 'second'})
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(lambda _: MediaResource.download([first, second], 'videos', file_type='sharded'), range(2)))
        self.assertEqual(results[0], results[1])
        self.assertEqual((Path(results[0]) / 'first.txt').read_text(), 'first')
        self.assertEqual((Path(results[0]) / 'second.txt').read_text(), 'second')

    def test_shard_order_preserves_last_file_and_nested_directories(self):
        first = self._archive('first.tar', {'nested/shared.txt': 'first', 'a.txt': 'a'})
        second = self._archive('second.tar', {'nested/shared.txt': 'second', 'b.txt': 'b'})
        result = Path(MediaResource.download([first, second], 'videos', file_type='sharded'))
        self.assertEqual((result / 'nested/shared.txt').read_text(), 'second')
        self.assertEqual((result / 'a.txt').read_text(), 'a')
        self.assertEqual((result / 'b.txt').read_text(), 'b')

    def test_nested_local_alias_is_preserved(self):
        first = self._archive('first.tar', {'first.txt': 'first'})
        result = Path(MediaResource.download([first], 'nested/videos', file_type='sharded'))
        self.assertEqual(result, self.cache / 'nested/videos')
        self.assertEqual((result / 'first.txt').read_text(), 'first')

    def test_existing_legacy_cache_is_preserved(self):
        result = self.cache / 'videos'
        result.mkdir(parents=True)
        (result / 'user.txt').write_text('user content')
        missing = str(self.root / 'missing.tar')
        self.assertEqual(MediaResource.download([missing], 'videos', file_type='sharded'), str(result))
        self.assertEqual((result / 'user.txt').read_text(), 'user content')

    def test_single_archive_and_file_downloads_are_unchanged(self):
        archive = self._archive('single.tar', {'one.txt': 'one'})
        result = Path(MediaResource.download(archive, 'single'))
        self.assertEqual((result / 'one.txt').read_text(), 'one')
        source = self.root / 'plain.txt'
        source.write_text('plain')
        result = Path(MediaResource.download(str(source), 'plain', file_type='file'))
        self.assertEqual((result / 'plain.txt').read_text(), 'plain')


if __name__ == '__main__':
    unittest.main()
