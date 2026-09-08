# Copyright (c) ModelScope Contributors. All rights reserved.
import ipaddress
import requests
import socket
from requests.adapters import HTTPAdapter
from typing import List, Optional, Set, Union
from urllib3.util.retry import Retry
from urllib.parse import urljoin, urlparse

from .utils import get_env_args

IpAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]


class SafeUrlFetcher:
    """Fetch a URL that may have been chosen by an untrusted caller.

    A bare `requests.get` on such a URL turns the server into a proxy for its own network: the caller can
    point it at the cloud instance metadata endpoint (which hands out the instance credentials), at internal
    HTTP APIs, or at localhost-only admin ports, and read the result back through the model (SSRF, CWE-918).
    This is reachable from `swift deploy`, where the media URLs in a chat request are downloaded server-side.

    Every URL is therefore resolved and checked before the request goes out, and redirects are followed one
    hop at a time so that a public URL cannot bounce the request into the internal network. Two environment
    variables relax the checks for trusted setups:

    - `SWIFT_ALLOW_INTERNAL_URL=1`: allow private/loopback addresses, e.g. an internal image server used by
      an offline training job. Cloud metadata endpoints stay blocked.
    - `SWIFT_URL_ALLOWED_HOSTS=host1,host2`: only fetch from these hosts, and trust them regardless of the
      addresses they resolve to. Recommended when a deployment serves media from a known bucket domain.

    Note that a host resolved here is resolved again by the OS when the connection is made, so a DNS entry
    that changes between the two (DNS rebinding) is not covered; use `SWIFT_URL_ALLOWED_HOSTS` plus an egress
    firewall if that is part of your threat model.
    """
    ALLOWED_SCHEMES = ('http', 'https')
    MAX_REDIRECTS = 5
    RETRY_TOTAL = 3
    # Cloud instance metadata services, reachable from inside virtually every cloud VM and container and
    # holding short-lived credentials for the whole account, so they are blocked unconditionally.
    # 169.254.0.0/16: AWS/Azure/GCP/Huawei (and Tencent's metadata.tencentyun.com); 100.100.100.200: Alibaba
    # Cloud, which `is_global` reports as public since Python 3.13; fd00:ec2::254: AWS over IPv6.
    METADATA_HOSTS = frozenset({'metadata.google.internal', 'metadata.tencentyun.com'})
    METADATA_NETWORKS = tuple(
        ipaddress.ip_network(network) for network in ('169.254.0.0/16', '100.100.100.200/32', 'fd00:ec2::254/128'))

    @classmethod
    def get(cls, url: str, **kwargs) -> requests.Response:
        """`requests.get`, but every hop of the request is checked by `check_url` first."""
        kwargs.pop('allow_redirects', None)  # redirects are followed below, one checked hop at a time
        retries = Retry(total=cls.RETRY_TOTAL, backoff_factor=1, allowed_methods=['GET'])
        with requests.Session() as session:
            session.mount('http://', HTTPAdapter(max_retries=retries))
            session.mount('https://', HTTPAdapter(max_retries=retries))
            for _ in range(cls.MAX_REDIRECTS + 1):
                cls.check_url(url)
                response = session.get(url, allow_redirects=False, **kwargs)
                if not response.is_redirect:
                    response.raise_for_status()
                    return response
                location = response.headers['location']
                response.close()
                url = urljoin(url, location)
        raise ValueError(f'The URL {url!r} exceeded the limit of {cls.MAX_REDIRECTS} redirects.')

    @classmethod
    def check_url(cls, url: str) -> None:
        """Raise a ValueError if `url` must not be fetched on behalf of an untrusted caller."""
        parsed = urlparse(url)
        if parsed.scheme.lower() not in cls.ALLOWED_SCHEMES:
            raise ValueError(f'Refusing to fetch {url!r}: only '
                             f'{"/".join(scheme + "://" for scheme in cls.ALLOWED_SCHEMES)} URLs are supported.')
        host = (parsed.hostname or '').rstrip('.').lower()
        if not host:
            raise ValueError(f'Refusing to fetch {url!r}: the URL does not contain a host.')
        if host in cls.METADATA_HOSTS:
            cls._raise_metadata(url, f'{host!r} is a cloud instance metadata endpoint')
        allowed_hosts = cls._get_allowed_hosts()
        if allowed_hosts is not None:
            # An explicit allowlist is an explicit trust decision, so the address checks below are skipped.
            if host not in allowed_hosts:
                raise ValueError(f'Refusing to fetch {url!r}: the host {host!r} is not listed in '
                                 'the `SWIFT_URL_ALLOWED_HOSTS` environment variable.')
            return
        for ip in cls._resolve(host):
            cls._check_ip(url, ip)

    @classmethod
    def _check_ip(cls, url: str, ip: IpAddress) -> None:
        if getattr(ip, 'ipv4_mapped', None) is not None:
            ip = ip.ipv4_mapped  # e.g. ::ffff:169.254.169.254
        for network in cls.METADATA_NETWORKS:
            if ip.version == network.version and ip in network:
                cls._raise_metadata(url, f'it resolves to {ip}, inside the metadata range {network}')
        if cls._allow_internal_url():
            return
        if not ip.is_global:
            raise ValueError(
                f'Refusing to fetch {url!r}: it resolves to {ip}, which is not a public address. Fetching '
                'addresses that are only reachable from the server exposes internal services (SSRF). If this '
                'URL is trusted, set the environment variable `SWIFT_ALLOW_INTERNAL_URL=1`, or list the hosts '
                'you trust in `SWIFT_URL_ALLOWED_HOSTS`.')

    @classmethod
    def _resolve(cls, host: str) -> List[IpAddress]:
        try:
            return [ipaddress.ip_address(host)]
        except ValueError:
            pass  # not an IP literal, resolve it
        try:
            addr_infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        except socket.gaierror as e:
            raise ValueError(f'Refusing to fetch from the host {host!r}: it cannot be resolved ({e}).') from e
        # Check every address the host resolves to; the OS may connect to any of them.
        res = [ipaddress.ip_address(addr_info[4][0].split('%', 1)[0]) for addr_info in addr_infos]
        if not res:
            raise ValueError(f'Refusing to fetch from the host {host!r}: it does not resolve to any address.')
        return res

    @classmethod
    def _raise_metadata(cls, url: str, reason: str) -> None:
        raise ValueError(f'Refusing to fetch {url!r}: {reason}. Cloud metadata endpoints hand out the '
                         'credentials of the instance and are never fetched on behalf of a caller.')

    @staticmethod
    def _allow_internal_url() -> bool:
        return bool(get_env_args('swift_allow_internal_url', bool, False))

    @staticmethod
    def _get_allowed_hosts() -> Optional[Set[str]]:
        allowed_hosts = get_env_args('swift_url_allowed_hosts', str, None)
        if not allowed_hosts:
            return None
        res = {host.strip().rstrip('.').lower() for host in allowed_hosts.split(',')}
        res.discard('')
        return res or None
