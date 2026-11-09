import os
import tempfile
import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

from swift.utils.hub_utils import download_file, download_ms_file, safe_snapshot_download


class TestHubUtils(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        self._tmp_dir.cleanup()

    @staticmethod
    def _mock_response(chunks):
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = chunks
        response.iter_lines.return_value = [chunk.rstrip(b'\r\n') for chunk in chunks]
        response.headers = {'content-length': str(sum(map(len, chunks)))}
        return response

    @patch('swift.utils.hub_utils.ModelScopeConfig.get_cookies', return_value={'session': 'test'})
    @patch('swift.utils.hub_utils.requests.get')
    def test_download_ms_file_preserves_bytes_and_closes_response(self, mock_get, mock_cookies):
        chunks = [b'first line\n', b'\x00second line\r\n']
        response = self._mock_response(chunks)
        mock_get.return_value = response
        local_path = os.path.join(self.tmp_dir, 'artifact.bin')

        download_ms_file('https://example.com/artifact.bin', local_path)

        with open(local_path, 'rb') as f:
            self.assertEqual(f.read(), b''.join(chunks))
        mock_get.assert_called_once_with('https://example.com/artifact.bin', cookies={'session': 'test'}, stream=True)
        response.raise_for_status.assert_called_once_with()
        response.__exit__.assert_called_once()

    @patch('swift.utils.hub_utils.requests.get')
    @patch('swift.utils.hub_utils.get_cache_dir')
    def test_download_file_closes_response(self, mock_cache_dir, mock_get):
        mock_cache_dir.return_value = self.tmp_dir
        chunks = [b'model', b' data']
        response = self._mock_response(chunks)
        mock_get.return_value = response

        file_path = download_file('https://example.com/model.bin')

        with open(file_path, 'rb') as f:
            self.assertEqual(f.read(), b''.join(chunks))
        response.__exit__.assert_called_once()

    @patch('swift.utils.hub_utils.requests.get')
    @patch('swift.utils.hub_utils.get_cache_dir')
    def test_download_file_removes_partial_cache_on_failure(self, mock_cache_dir, mock_get):
        mock_cache_dir.return_value = self.tmp_dir
        response = self._mock_response([b'partial'])
        response.iter_content.side_effect = OSError('connection reset')
        mock_get.return_value = response

        with self.assertRaisesRegex(OSError, 'connection reset'):
            download_file('https://example.com/model.bin')

        cache_dir = os.path.join(self.tmp_dir, 'files')
        self.assertEqual(os.listdir(cache_dir), [])

    @patch('swift.utils.hub_utils.requests.get')
    @patch('swift.utils.hub_utils.get_cache_dir')
    def test_download_file_uses_a_distinct_cache_path_for_each_url(self, mock_cache_dir, mock_get):
        mock_cache_dir.return_value = self.tmp_dir
        mock_get.side_effect = [self._mock_response([b'first']), self._mock_response([b'second'])]

        first_path = download_file('https://first.example.com/releases/artifact.bin')
        second_path = download_file('https://second.example.com/releases/artifact.bin')
        cached_first_path = download_file('https://first.example.com/releases/artifact.bin')

        self.assertNotEqual(first_path, second_path)
        self.assertEqual(cached_first_path, first_path)
        with open(first_path, 'rb') as f:
            self.assertEqual(f.read(), b'first')
        with open(second_path, 'rb') as f:
            self.assertEqual(f.read(), b'second')
        self.assertEqual(mock_get.call_count, 2)

    @patch('swift.utils.hub_utils.safe_ddp_context', side_effect=lambda **kwargs: nullcontext())
    @patch('swift.hub.get_hub')
    def test_metadata_download_does_not_mutate_caller_ignore_patterns(self, mock_get_hub, _mock_safe_ddp_context):
        hub = MagicMock()
        hub.download_model.return_value = self.tmp_dir
        mock_get_hub.return_value = hub
        ignore_patterns = ['custom/*']

        safe_snapshot_download('org/model', download_model=False, ignore_patterns=ignore_patterns)

        self.assertEqual(ignore_patterns, ['custom/*'])
        self.assertEqual(hub.download_model.call_args_list[0].args[2], ['custom/*', '*.bin', '*.safetensors'])

        safe_snapshot_download('org/model', download_model=True, ignore_patterns=ignore_patterns)

        self.assertEqual(hub.download_model.call_args_list[1].args[2], ['custom/*'])


if __name__ == '__main__':
    unittest.main()
