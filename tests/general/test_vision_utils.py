import unittest
from unittest.mock import patch

from swift.template.vision_utils import UnsafeMediaURLError, _load_audio_librosa, _validate_media_url, load_file


class TestMediaUrlSsrfGuard(unittest.TestCase):
    """Regression coverage for #9740.

    ``swift deploy`` fetches ``image_url`` / ``audio_url`` / ``video_url`` server-side and is
    unauthenticated by default, so media URLs must not be able to reach loopback, private
    ranges or the cloud metadata endpoint.
    """

    # (url, allowed) -- IP literals only, so the cases need no DNS and stay deterministic.
    IP_LITERAL_CASES = [
        ('http://169.254.169.254/latest/meta-data/iam/security-credentials/', False),  # metadata
        ('http://127.0.0.1:8456/secret.png', False),
        ('http://127.1/secret.png', False),  # shorthand loopback
        ('http://2130706433/secret.png', False),  # decimal loopback
        ('http://10.0.0.5/x.png', False),
        ('http://192.168.1.10/x.png', False),
        ('http://172.16.0.1/x.png', False),
        ('http://0.0.0.0/x.png', False),
        ('http://[::1]/x.png', False),  # IPv6 loopback
        ('http://[fc00::1]/x.png', False),  # IPv6 unique-local
        ('ftp://example.com/x.png', False),  # non-http scheme
        ('http://93.184.216.34/x.png', True),  # public
        ('https://93.184.216.34:8443/x.png', True),
    ]

    def test_ip_literal_urls(self):
        for url, allowed in self.IP_LITERAL_CASES:
            with self.subTest(url=url):
                if allowed:
                    self.assertEqual(_validate_media_url(url), url)
                else:
                    with self.assertRaises(UnsafeMediaURLError):
                        _validate_media_url(url)

    def test_hostname_resolving_to_private_address(self):
        """A public hostname that resolves into private space is rejected too."""
        addrinfos = [(2, 1, 6, '', ('127.0.0.1', 80))]
        with patch('swift.template.vision_utils.socket.getaddrinfo', return_value=addrinfos):
            with self.assertRaises(UnsafeMediaURLError):
                _validate_media_url('http://localtest.me/secret.png')

    def test_load_file_rejects_loopback_before_any_request(self):
        with patch('swift.template.vision_utils.requests.Session.get') as mock_get:
            with self.assertRaises(UnsafeMediaURLError):
                load_file('http://127.0.0.1:8456/secret.png')
        mock_get.assert_not_called()

    def test_load_file_validates_every_redirect_hop(self):
        """A public host must not be able to 302 to the metadata endpoint."""
        public_addrinfos = [(2, 1, 6, '', ('93.184.216.34', 80))]
        redirect = unittest.mock.Mock(is_redirect=True, headers={'location': 'http://169.254.169.254/latest/'})
        with patch('swift.template.vision_utils.requests.Session.get', return_value=redirect) as mock_get, \
                patch('swift.template.vision_utils.socket.getaddrinfo', return_value=public_addrinfos):
            with self.assertRaises(UnsafeMediaURLError):
                load_file('http://example.com/redirect')
        self.assertEqual(mock_get.call_count, 1)

    def test_load_file_follows_safe_redirect(self):
        """A redirect that stays on public addresses is still followed."""
        public_addrinfos = [(2, 1, 6, '', ('93.184.216.34', 80))]
        redirect = unittest.mock.Mock(is_redirect=True, headers={'location': 'http://93.184.216.34/final.png'})
        final = unittest.mock.Mock(is_redirect=False, content=b'png-bytes')
        with patch('swift.template.vision_utils.requests.Session.get', side_effect=[redirect, final]) as mock_get, \
                patch('swift.template.vision_utils.socket.getaddrinfo', return_value=public_addrinfos):
            res = load_file('http://example.com/redirect')
        self.assertEqual(res.read(), b'png-bytes')
        self.assertEqual(mock_get.call_count, 2)
        for call in mock_get.call_args_list:
            self.assertIs(call.kwargs['allow_redirects'], False)

    def test_opt_out_env_var(self):
        """Deployments that serve media internally can opt out explicitly."""
        with patch.dict('os.environ', {'SWIFT_ALLOW_PRIVATE_MEDIA_URLS': '1'}):
            self.assertEqual(_validate_media_url('http://10.0.0.5/x.png'), 'http://10.0.0.5/x.png')

    def test_audio_fallback_does_not_bypass_guard(self):
        """A blocked url must not be retried through the ffmpeg fallback."""
        try:
            import librosa  # noqa: F401
        except ImportError:
            self.skipTest('librosa is not installed')

        with patch('swift.template.vision_utils.load_file', side_effect=UnsafeMediaURLError('blocked')):
            with self.assertRaises(UnsafeMediaURLError):
                _load_audio_librosa('http://10.0.0.5/a.wav', 16000)


if __name__ == '__main__':
    unittest.main()
