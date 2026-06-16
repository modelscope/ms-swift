import os
import time
import unittest

from swift.template.vision_utils import _decode_with_timeout


def _sleep_forever(*args, **kwargs):
    time.sleep(3600)


def _echo(value):
    return value


def _raise_value_error(*args, **kwargs):
    raise ValueError('boom')


def _big_payload(*args, **kwargs):
    # ~1 MB, far above the ~64 KB OS pipe buffer that made a SimpleQueue worker block its put()
    # while the parent waited in join() -- a deadlock that false-timed-out every real media clip.
    return b'x' * (1024 * 1024)


class TestDecodeWithTimeout(unittest.TestCase):

    def setUp(self):
        self._saved_timeout = os.environ.get('MEDIA_DECODE_TIMEOUT')

    def tearDown(self):
        if self._saved_timeout is None:
            os.environ.pop('MEDIA_DECODE_TIMEOUT', None)
        else:
            os.environ['MEDIA_DECODE_TIMEOUT'] = self._saved_timeout

    def test_kills_hung_decode(self):
        # A decode that never returns must be killed and surface a TimeoutError rather than hang.
        os.environ['MEDIA_DECODE_TIMEOUT'] = '2'
        start = time.time()
        with self.assertRaises(TimeoutError):
            _decode_with_timeout(_sleep_forever)
        self.assertLess(time.time() - start, 30)

    def test_returns_result_when_fast(self):
        os.environ['MEDIA_DECODE_TIMEOUT'] = '10'
        self.assertEqual(_decode_with_timeout(_echo, 'ok'), 'ok')

    def test_propagates_decode_error(self):
        os.environ['MEDIA_DECODE_TIMEOUT'] = '10'
        with self.assertRaises(ValueError):
            _decode_with_timeout(_raise_value_error)

    def test_disabled_calls_directly(self):
        # Default (unset / 0): no subprocess, original behavior and zero overhead.
        os.environ.pop('MEDIA_DECODE_TIMEOUT', None)
        self.assertEqual(_decode_with_timeout(_echo, 'direct'), 'direct')

    def test_handles_large_payload(self):
        # A decoded payload larger than the OS pipe buffer must transfer without deadlocking the
        # worker (regression: the previous SimpleQueue implementation false-timed-out here).
        os.environ['MEDIA_DECODE_TIMEOUT'] = '10'
        self.assertEqual(_decode_with_timeout(_big_payload), b'x' * (1024 * 1024))


if __name__ == '__main__':
    unittest.main()
