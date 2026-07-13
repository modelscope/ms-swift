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
