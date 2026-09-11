# Copyright (c) ModelScope Contributors. All rights reserved.
import os

from swift.template.vision_utils import _assert_media_url_allowed


def _is_blocked(url):
    try:
        _assert_media_url_allowed(url)
        return False
    except ValueError:
        return True


def test_media_url_ssrf_blocked():
    """``load_file`` is reachable unauthenticated from ``swift deploy`` via
    ``image_url`` / ``audio_url`` / ``video_url``, so media URLs pointing at
    loopback, private, link-local (cloud instance metadata), unspecified, or a
    non-http scheme must be rejected. See issue #9740 / GHSA-7mpg-jvqj-ccxr.
    """
    blocked = [
        'http://169.254.169.254/latest/meta-data/',  # cloud instance metadata (link-local)
        'http://127.0.0.1:8456/secret.png',  # loopback (issue PoC)
        'http://localhost/x',  # loopback via name
        'http://10.0.0.5/x',
        'http://192.168.1.1/x',
        'http://172.16.0.1/x',
        'http://0.0.0.0/x',  # unspecified
        'file:///etc/passwd',  # non-http scheme
        'gopher://127.0.0.1/x',
    ]
    for url in blocked:
        assert _is_blocked(url), f'expected {url!r} to be rejected as SSRF'


def test_media_url_public_allowed():
    # Public addresses (given as literals so the check needs no external DNS) stay allowed.
    for url in ['http://8.8.8.8/img.png', 'https://1.1.1.1/a.jpg']:
        assert not _is_blocked(url), f'expected public {url!r} to be allowed'


def test_media_url_opt_out():
    # An explicit opt-out restores the old behaviour for trusted deployments.
    prev = os.environ.get('SWIFT_ALLOW_INTERNAL_MEDIA_URLS')
    os.environ['SWIFT_ALLOW_INTERNAL_MEDIA_URLS'] = '1'
    try:
        _assert_media_url_allowed('http://127.0.0.1/x')  # must not raise
    finally:
        if prev is None:
            os.environ.pop('SWIFT_ALLOW_INTERNAL_MEDIA_URLS', None)
        else:
            os.environ['SWIFT_ALLOW_INTERNAL_MEDIA_URLS'] = prev


def test_load_audio_librosa_blocks_internal_url_before_fallback():
    """``_load_audio_librosa`` wraps ``load_file`` in ``except Exception`` and, for an
    ``http(s)`` URL, re-fetches it via ``audioread``/ffmpeg. That fallback would swallow
    the SSRF ``ValueError`` load_file raises and fetch the internal URL anyway, so the
    URL must be validated up front, before the fallback can run. See #9740.

    ``librosa`` and ``audioread`` are stubbed via ``sys.modules`` so the test needs no
    audio deps, ffmpeg, or network; the ffmpeg stub records if it is ever reached.
    """
    import sys
    import types

    from swift.template import vision_utils

    reached = []
    fake_audioread = types.ModuleType('audioread')
    fake_audioread.ffdec = types.SimpleNamespace(FFmpegAudioFile=lambda url: reached.append(url))
    fake_librosa = types.ModuleType('librosa')
    fake_librosa.load = lambda *args, **kwargs: (None, None)

    saved = {name: sys.modules.get(name) for name in ('audioread', 'librosa')}
    sys.modules['audioread'] = fake_audioread
    sys.modules['librosa'] = fake_librosa
    try:
        blocked = False
        try:
            vision_utils._load_audio_librosa('http://127.0.0.1:9000/x.wav', 16000)
        except ValueError:
            blocked = True
        assert blocked, 'internal audio_url was not rejected'
        assert reached == [], 'internal audio_url reached the ffmpeg fallback (SSRF bypass)'
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
