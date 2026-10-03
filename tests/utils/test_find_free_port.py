import socket
import unittest

from swift.utils import find_free_port


def _reserve(count):
    """Bind ``count`` consecutive free ports and keep those sockets open.

    Returns ``(sockets, base_port)``, with ``sockets[i]`` holding ``base_port + i``. The caller
    owns the sockets and must close them; a socket that only ever bound (never listened or
    connected) leaves no TIME_WAIT behind, so a released port is bindable again right away.
    """
    for _ in range(200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(('', 0))
            base = probe.getsockname()[1]
        socks = []
        for port in range(base, base + count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('', port))
            except OSError:
                for held in socks:
                    held.close()
                break
            socks.append(sock)
        else:
            return socks, base
    raise AssertionError(f'Could not reserve {count} consecutive free ports.')


class TestFindFreePort(unittest.TestCase):

    def _release(self, socks, index):
        socks[index].close()
        socks[index] = None

    def tearDown(self):
        for sock in getattr(self, 'socks', []):
            if sock is not None:
                sock.close()
        self.socks = []

    def test_default_call_returns_a_bindable_port(self):
        # control: the no-argument call keeps handing out a free ephemeral port
        port = find_free_port()
        self.assertGreater(port, 0)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('', port))

    def test_explicit_free_port_is_returned(self):
        # control: a requested port that is free must come back unchanged
        self.socks, base = _reserve(1)
        self._release(self.socks, 0)
        self.assertEqual(find_free_port(base, retry=3), base)

    def test_partially_used_range_skips_the_busy_port(self):
        # control: base and base+2 are held, base+1 is known free -> the scan must report base+1
        self.socks, base = _reserve(3)
        self._release(self.socks, 1)
        self.assertEqual(find_free_port(base, retry=3), base + 1)

    def test_exhausted_range_does_not_report_a_busy_port(self):
        # every port of the scan window is taken, so no free port exists to report
        self.socks, base = _reserve(3)
        with self.assertRaises(OSError) as ctx:
            find_free_port(base, retry=3)
        self.assertIn(str(base), str(ctx.exception))

    def test_zero_retry_raises_oserror(self):
        with self.assertRaises(OSError):
            find_free_port(find_free_port(), retry=0)


if __name__ == '__main__':
    unittest.main()
