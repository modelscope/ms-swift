import unittest
from unittest.mock import call, patch

from swift.utils import find_free_port


class TestFindFreePort(unittest.TestCase):

    @patch('swift.utils.utils.socket.socket')
    def test_returns_next_available_port(self, socket_cls):
        sock = socket_cls.return_value.__enter__.return_value
        sock.bind.side_effect = [OSError('occupied'), None]
        sock.getsockname.return_value = ('0.0.0.0', 30001)

        port = find_free_port(30000, retry=2)

        self.assertEqual(port, 30001)
        self.assertEqual(sock.bind.call_args_list, [call(('', 30000)), call(('', 30001))])

    @patch('swift.utils.utils.socket.socket')
    def test_raises_when_candidate_range_is_exhausted(self, socket_cls):
        sock = socket_cls.return_value.__enter__.return_value
        sock.bind.side_effect = OSError('occupied')

        with self.assertRaisesRegex(OSError, r'\[30000, 30003\)'):
            find_free_port(30000, retry=3)

        self.assertEqual(sock.bind.call_count, 3)

    @patch('swift.utils.utils.socket.socket')
    def test_does_not_scan_past_max_port(self, socket_cls):
        sock = socket_cls.return_value.__enter__.return_value
        sock.bind.side_effect = OSError('occupied')

        with self.assertRaisesRegex(OSError, r'\[65535, 65536\)'):
            find_free_port(65535, retry=2)

        sock.bind.assert_called_once_with(('', 65535))


if __name__ == '__main__':
    unittest.main()
