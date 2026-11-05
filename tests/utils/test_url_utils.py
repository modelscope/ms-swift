import ipaddress
import os
import unittest
from unittest.mock import MagicMock, patch

from swift.utils.url_utils import SafeUrlFetcher

ENV_KEYS = ['SWIFT_ALLOW_INTERNAL_URL', 'SWIFT_URL_ALLOWED_HOSTS', 'SWIFT_MAX_DOWNLOAD_SIZE_MB']


class TestSafeUrlFetcher(unittest.TestCase):

    def setUp(self):
        # Isolate each test from any ambient configuration.
        for key in ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self):
        for key in ENV_KEYS:
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

    def test_ambiguous_authorities_are_blocked(self):
        """Reject syntax for which urllib.parse and requests can disagree about the connection host."""
        for url in [
                'http://127.0.0.1\\@example.com/x',
                'http://example.com\\@127.0.0.1/x',
                'http://example.com\n@127.0.0.1/x',
                'http://example.com\x7f@127.0.0.1/x',
        ]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                SafeUrlFetcher.check_url(url)

    def test_obscured_loopback_addresses_are_blocked(self):
        loopback = ipaddress.ip_address('127.0.0.1')
        for url in ['http://0177.0.0.1/x', 'http://2130706433/x', 'http://[::ffff:127.0.0.1]/x']:
            with self.subTest(url=url), patch.object(SafeUrlFetcher, '_resolve', return_value=[loopback]):
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
        public_ip = ipaddress.ip_address('93.184.216.34')
        with patch.object(SafeUrlFetcher, '_resolve', return_value=[public_ip]):
            SafeUrlFetcher.check_url('http://mybucket.oss.com/x')
            SafeUrlFetcher.check_url('http://cdn.example.com/x')
        with self.assertRaises(ValueError):
            SafeUrlFetcher.check_url('http://evil.com/x')

    def test_metadata_ip_is_blocked_even_when_allowlisted(self):
        os.environ['SWIFT_URL_ALLOWED_HOSTS'] = '169.254.169.254,100.100.100.200'
        for url in ['http://169.254.169.254/x', 'http://100.100.100.200/x']:
            with self.assertRaises(ValueError):
                SafeUrlFetcher.check_url(url)

    def test_allowlisted_dns_alias_to_metadata_is_blocked(self):
        os.environ['SWIFT_URL_ALLOWED_HOSTS'] = 'trusted.example'
        with patch.object(SafeUrlFetcher, '_resolve', return_value=[ipaddress.ip_address('169.254.169.254')]):
            with self.assertRaises(ValueError):
                SafeUrlFetcher.check_url('http://trusted.example/latest/meta-data/')

    @staticmethod
    def _response(*, is_redirect=False, location=None, content=b'', headers=None):
        response = MagicMock()
        response.__enter__.return_value = response
        response.is_redirect = is_redirect
        response.headers = dict(headers or {})
        if location is not None:
            response.headers['location'] = location
        response.content = content
        response.iter_content.return_value = [content]
        return response

    @staticmethod
    def _session(response):
        session = MagicMock()
        session.__enter__.return_value = session
        session.get.return_value = response
        return session

    def test_redirect_to_internal_address_is_blocked(self):
        """A public URL must not be able to bounce the request into the internal network via a redirect."""
        session = self._session(self._response(is_redirect=True, location='http://169.254.169.254/latest/meta-data/'))
        with patch('swift.utils.url_utils.requests.Session', return_value=session):
            with self.assertRaises(ValueError):
                SafeUrlFetcher.read('http://example.com/redir')

    def test_read_returns_body(self):
        response = self._response(content=b'hello')
        session = self._session(response)
        with patch('swift.utils.url_utils.requests.Session', return_value=session):
            res = SafeUrlFetcher.read('http://example.com/a.png')
        self.assertEqual(res, b'hello')
        response.raise_for_status.assert_called_once()

    def test_oversized_body_is_rejected_while_reading(self):
        """A response that omits (or understates) content-length must still be capped."""
        os.environ['SWIFT_MAX_DOWNLOAD_SIZE_MB'] = '0.001'  # ~1KB
        session = self._session(self._response(content=b'x' * 4096))
        with patch('swift.utils.url_utils.requests.Session', return_value=session):
            with self.assertRaises(ValueError):
                SafeUrlFetcher.read('http://example.com/big.png')

    def test_oversized_content_length_is_rejected_before_reading(self):
        os.environ['SWIFT_MAX_DOWNLOAD_SIZE_MB'] = '0.001'  # ~1KB
        response = self._response(content=b'x', headers={'content-length': str(100 * 1024**2)})
        session = self._session(response)
        with patch('swift.utils.url_utils.requests.Session', return_value=session):
            with self.assertRaises(ValueError):
                SafeUrlFetcher.read('http://example.com/big.png')
        response.iter_content.assert_not_called()

    def test_size_cap_can_be_disabled(self):
        os.environ['SWIFT_MAX_DOWNLOAD_SIZE_MB'] = '0'
        session = self._session(self._response(content=b'x' * 4096))
        with patch('swift.utils.url_utils.requests.Session', return_value=session):
            self.assertEqual(len(SafeUrlFetcher.read('http://example.com/big.png')), 4096)


if __name__ == '__main__':
    unittest.main()
