import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from swift.utils.hub_utils import download_file, download_ms_file


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


if __name__ == '__main__':
    unittest.main()
