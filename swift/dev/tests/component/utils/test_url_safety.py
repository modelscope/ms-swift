# Copyright (c) ModelScope Contributors. All rights reserved.
"""C3-MM04: SafeUrlFetcher / SafeMediaPath guards against untrusted media inputs.

Real loopback HTTP servers with request counters -- a refusal must mean the request never went out,
not merely that an exception surfaced after the fact. Every refusal case is paired with the
environment escape hatch that legitimately allows the same shape.
"""
import http.server
import os
import threading

import pytest

from swift.dev.utils.url_safety import SafeMediaPath, SafeUrlFetcher

BODY = b'media-bytes-0123456789' * 4


class _Counter(http.server.BaseHTTPRequestHandler):
    hits = 0
    redirect_to = None

    def do_GET(self):
        type(self).hits += 1
        if self.redirect_to:
            self.send_response(302)
            self.send_header('Location', self.redirect_to)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Length', str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    handler = type('Handler', (_Counter,), {'hits': 0, 'redirect_to': None})
    httpd = http.server.HTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, handler
    httpd.shutdown()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ('SWIFT_ALLOW_INTERNAL_URL', 'SWIFT_URL_ALLOWED_HOSTS', 'SWIFT_MAX_DOWNLOAD_SIZE_MB',
                 'SWIFT_MEDIA_ALLOWED_DIRS'):
        monkeypatch.delenv(name, raising=False)


def test_loopback_refused_by_default_without_a_request(server, monkeypatch):
    httpd, handler = server
    with pytest.raises(ValueError, match='not a public address'):
        SafeUrlFetcher.read(f'http://127.0.0.1:{httpd.server_port}/x')
    assert handler.hits == 0


def test_loopback_allowed_with_env_reads_real_bytes(server, monkeypatch):
    httpd, handler = server
    monkeypatch.setenv('SWIFT_ALLOW_INTERNAL_URL', '1')
    assert SafeUrlFetcher.read(f'http://127.0.0.1:{httpd.server_port}/x') == BODY
    assert handler.hits == 1


def test_allowlist_permits_listed_host_and_refuses_others(server, monkeypatch):
    httpd, handler = server
    monkeypatch.setenv('SWIFT_URL_ALLOWED_HOSTS', '127.0.0.1')
    assert SafeUrlFetcher.read(f'http://127.0.0.1:{httpd.server_port}/x') == BODY
    with pytest.raises(ValueError, match='not listed'):
        SafeUrlFetcher.read(f'http://localhost:{httpd.server_port}/x')
    assert handler.hits == 1  # the refused host never produced a request


def test_metadata_address_never_allowed_even_with_internal_escape(monkeypatch):
    monkeypatch.setenv('SWIFT_ALLOW_INTERNAL_URL', '1')
    with pytest.raises(ValueError, match='metadata'):
        SafeUrlFetcher.check_url('http://169.254.169.254/latest/meta-data/')


def test_redirect_hop_is_rechecked(server, monkeypatch):
    httpd_a, handler_a = server
    handler_b_type = type('H2', (_Counter,), {'hits': 0, 'redirect_to': None})
    httpd_b = http.server.HTTPServer(('127.0.0.1', 0), handler_b_type)
    threading.Thread(target=httpd_b.serve_forever, daemon=True).start()
    handler_a.redirect_to = f'http://127.0.0.1:{httpd_b.server_port}/'
    monkeypatch.setenv('SWIFT_URL_ALLOWED_HOSTS', 'localhost')
    try:
        with pytest.raises(ValueError, match='not listed'):
            SafeUrlFetcher.read(f'http://localhost:{httpd_a.server_port}/x')
        assert handler_b_type.hits == 0  # the redirect target was never requested
    finally:
        httpd_b.shutdown()


def test_download_size_cap_refuses_oversize_body(server, monkeypatch):
    httpd, handler = server
    monkeypatch.setenv('SWIFT_ALLOW_INTERNAL_URL', '1')
    monkeypatch.setenv('SWIFT_MAX_DOWNLOAD_SIZE_MB', '0.000001')  # 1 byte
    with pytest.raises(ValueError, match='larger than'):
        SafeUrlFetcher.read(f'http://127.0.0.1:{httpd.server_port}/x')


def test_non_http_scheme_refused():
    with pytest.raises(ValueError, match='URLs are supported'):
        SafeUrlFetcher.check_url('file:///etc/passwd')


def test_media_path_guard_realpath(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    root.mkdir()
    inside = root / 'ok.png'
    inside.write_bytes(BODY)
    outside = tmp_path / 'secret.png'
    outside.write_bytes(BODY)
    link = root / 'link.png'
    link.symlink_to(outside)
    monkeypatch.setenv('SWIFT_MEDIA_ALLOWED_DIRS', str(root))
    assert SafeMediaPath.check(str(inside)) == str(inside)
    with pytest.raises(ValueError, match='outside'):
        SafeMediaPath.check(str(outside))
    with pytest.raises(ValueError, match='outside'):
        SafeMediaPath.check(str(link))
    with pytest.raises(ValueError, match='outside'):
        SafeMediaPath.check(str(root / '..' / 'secret.png'))
    monkeypatch.delenv('SWIFT_MEDIA_ALLOWED_DIRS')
    assert SafeMediaPath.check(str(outside)) == str(outside)  # unset = training default, pass-through