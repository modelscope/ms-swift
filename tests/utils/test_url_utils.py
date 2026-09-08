import os
import unittest
from unittest.mock import MagicMock, patch

from swift.utils.url_utils import SafeUrlFetcher


class TestSafeUrlFetcher(unittest.TestCase):

    def setUp(self):
        # Isolate each test from any ambient configuration.
        for key in ['SWIFT_ALLOW_INTERNAL_URL', 'SWIFT_URL_ALLOWED_HOSTS']:
            os.environ.pop(key, None)

    def tearDown(self):
        for key in ['SWIFT_ALLOW_INTERNAL_URL', 'SWIFT_URL_ALLOWED_HOSTS']:
            os.environ.pop(key, None)

    def test_metadata_endpoints_are_blocked(self):
        for url in [
                'http://169.254.169.254/latest/meta-data/iam/security-credentials/',
                'http://100.100.100.200/latest/meta-data/',  # Alibaba Cloud
                'http://metadata.tencentyun.com/latest/meta-data/',  # Tencent Cloud
                'http://[::ffff:169.254.169.254]/x',  # IPv4-mapped IPv6
        ]:
            with self.assertRaises(ValueError):
                SafeUrlFetcher.check_url(url)

    def test_metadata_blocked_even_when_internal_url_allowed(self):
        os.environ['SWIFT_ALLOW_INTERNAL_URL'] = '1'
        with self.assertRaises(ValueError):
            SafeUrlFetcher.check_url('http://169.254.169.254/latest/meta-data/')

    def test_private_and_loopback_are_blocked(self):
        for url in [
                'http://127.0.0.1:8456/secret.png',
                'http://localhost/x',
                'http://[::1]/x',
                'http://192.168.1.1/x',
                'http://10.0.0.5/x',
                'http://172.16.0.1/x',
        ]:
            with self.assertRaises(ValueError):
                SafeUrlFetcher.check_url(url)

    def test_non_http_schemes_are_blocked(self):
        for url in ['file:///etc/passwd', 'gopher://x/1', 'ftp://host/x']:
            with self.assertRaises(ValueError):
                SafeUrlFetcher.check_url(url)

    def test_public_urls_are_allowed(self):
        for url in ['https://www.modelscope.cn/', 'http://example.com/a.png']:
            SafeUrlFetcher.check_url(url)

    def test_allow_internal_url_env(self):
        os.environ['SWIFT_ALLOW_INTERNAL_URL'] = '1'
        SafeUrlFetcher.check_url('http://127.0.0.1:8456/x')
        SafeUrlFetcher.check_url('http://192.168.1.1/x')

    def test_allowed_hosts_env(self):
        os.environ['SWIFT_URL_ALLOWED_HOSTS'] = 'mybucket.oss.com, cdn.example.com'
        SafeUrlFetcher.check_url('http://mybucket.oss.com/x')
        SafeUrlFetcher.check_url('http://cdn.example.com/x')
        with self.assertRaises(ValueError):
            SafeUrlFetcher.check_url('http://evil.com/x')
        # Metadata is blocked even if it were (mis)placed in the allowlist path.
        with self.assertRaises(ValueError):
            SafeUrlFetcher.check_url('http://169.254.169.254/x')

    @staticmethod
    def _redirect_response(location):
        response = MagicMock()
        response.is_redirect = True
        response.headers = {'location': location}
        return response

    def test_redirect_to_internal_address_is_blocked(self):
        """A public URL must not be able to bounce the request into the internal network via a redirect."""
        session = MagicMock()
        session.__enter__.return_value = session
        session.get.return_value = self._redirect_response('http://169.254.169.254/latest/meta-data/')
        with patch('swift.utils.url_utils.requests.Session', return_value=session):
            with self.assertRaises(ValueError):
                SafeUrlFetcher.get('http://example.com/redir')

    def test_get_returns_final_response(self):
        ok = MagicMock()
        ok.is_redirect = False
        session = MagicMock()
        session.__enter__.return_value = session
        session.get.return_value = ok
        with patch('swift.utils.url_utils.requests.Session', return_value=session):
            res = SafeUrlFetcher.get('http://example.com/a.png')
        self.assertIs(res, ok)
        ok.raise_for_status.assert_called_once()


if __name__ == '__main__':
    unittest.main()
