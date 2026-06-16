import pytest
import time

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


def test_decode_with_timeout_kills_hung_decode(monkeypatch):
    # A decode that never returns must be killed and surface a TimeoutError rather than hang.
    monkeypatch.setenv('MEDIA_DECODE_TIMEOUT', '2')
    start = time.time()
    with pytest.raises(TimeoutError):
        _decode_with_timeout(_sleep_forever)
    assert time.time() - start < 30


def test_decode_with_timeout_returns_result_when_fast(monkeypatch):
    monkeypatch.setenv('MEDIA_DECODE_TIMEOUT', '10')
    assert _decode_with_timeout(_echo, 'ok') == 'ok'


def test_decode_with_timeout_propagates_decode_error(monkeypatch):
    monkeypatch.setenv('MEDIA_DECODE_TIMEOUT', '10')
    with pytest.raises(ValueError, match='boom'):
        _decode_with_timeout(_raise_value_error)


def test_decode_with_timeout_disabled_calls_directly(monkeypatch):
    # Default (unset / 0): no subprocess, original behavior and zero overhead.
    monkeypatch.delenv('MEDIA_DECODE_TIMEOUT', raising=False)
    assert _decode_with_timeout(_echo, 'direct') == 'direct'


def test_decode_with_timeout_handles_large_payload(monkeypatch):
    # A decoded payload larger than the OS pipe buffer must transfer without deadlocking the
    # worker (regression: the previous SimpleQueue implementation false-timed-out here).
    monkeypatch.setenv('MEDIA_DECODE_TIMEOUT', '10')
    assert _decode_with_timeout(_big_payload) == b'x' * (1024 * 1024)
