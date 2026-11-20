#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.

import unittest
from urllib.parse import urlsplit

from swift.rlhf_trainers.utils import format_host_for_url, is_valid_ipv6_address


class TestIsValidIPv6Address(unittest.TestCase):

    def test_accepts_ipv6(self):
        self.assertTrue(is_valid_ipv6_address('::1'))
        self.assertTrue(is_valid_ipv6_address('2001:db8::1'))
        self.assertTrue(is_valid_ipv6_address('fe80::1%eth0'))

    def test_rejects_ipv4_and_hostnames(self):
        self.assertFalse(is_valid_ipv6_address('127.0.0.1'))
        self.assertFalse(is_valid_ipv6_address('localhost'))

    def test_rejects_bracketed_form(self):
        # `ipaddress` does not accept the bracketed URL form, so a host that is
        # already wrapped must not be wrapped a second time.
        self.assertFalse(is_valid_ipv6_address('[::1]'))


class TestFormatHostForUrl(unittest.TestCase):

    def test_ipv4_is_untouched(self):
        self.assertEqual(format_host_for_url('127.0.0.1'), '127.0.0.1')

    def test_hostname_is_untouched(self):
        self.assertEqual(format_host_for_url('localhost'), 'localhost')

    def test_ipv6_is_bracketed(self):
        self.assertEqual(format_host_for_url('::1'), '[::1]')
        self.assertEqual(format_host_for_url('2001:db8::1'), '[2001:db8::1]')

    def test_scoped_ipv6_zone_is_percent_encoded(self):
        # RFC 6874: `fe80::1%eth0` becomes `[fe80::1%25eth0]`. Without the
        # encoding the resulting URL holds `%et`, which HTTP clients treat as a
        # malformed percent-escape.
        self.assertEqual(format_host_for_url('fe80::1%eth0'), '[fe80::1%25eth0]')
        self.assertEqual(format_host_for_url('fe80::1%lo0'), '[fe80::1%25lo0]')

    def test_already_bracketed_is_not_double_wrapped(self):
        self.assertEqual(format_host_for_url('[::1]'), '[::1]')

    def test_built_url_is_parseable(self):
        host = format_host_for_url('fe80::1%eth0')
        url = f'http://{host}:8000/v1'
        parsed = urlsplit(url)
        self.assertEqual(parsed.port, 8000)
        self.assertEqual(parsed.path, '/v1')


if __name__ == '__main__':
    unittest.main()
