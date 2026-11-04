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

    - `SWIFT_ALLOW_INTERNAL_URL=1`: allow private/loopback addresses, e.g. a trusted internal media server
      used by a deployment. Cloud metadata endpoints stay blocked.
    - `SWIFT_URL_ALLOWED_HOSTS=host1,host2`: only fetch from these hosts. They may resolve to ordinary private
      addresses, but cloud metadata addresses remain blocked. Recommended for a known media bucket domain.
    - `SWIFT_MAX_DOWNLOAD_SIZE_MB`: cap on the response body, so that a caller cannot exhaust memory (and,
      for video, disk) by pointing the server at an endless response. Set to `0` to disable the cap.

    Note that a host resolved here is resolved again by the OS when the connection is made, so a DNS entry
    that changes between the two (DNS rebinding) is not covered; use `SWIFT_URL_ALLOWED_HOSTS` plus an egress
    firewall if that is part of your threat model.
    """
    ALLOWED_SCHEMES = ('http', 'https')
    MAX_REDIRECTS = 5
    RETRY_TOTAL = 3
    CHUNK_SIZE = 1024 * 1024
    DEFAULT_MAX_DOWNLOAD_SIZE_MB = 1024
    # Cloud instance metadata services, reachable from inside virtually every cloud VM and container and
    # holding short-lived credentials for the whole account, so they are blocked unconditionally.
    # 169.254.0.0/16: AWS/Azure/GCP/Huawei (and Tencent's metadata.tencentyun.com); 100.100.100.200: Alibaba
    # Cloud, which `is_global` reports as public since Python 3.13; fd00:ec2::254: AWS over IPv6.
    METADATA_HOSTS = frozenset({'metadata.google.internal', 'metadata.tencentyun.com'})
    METADATA_NETWORKS = tuple(
        ipaddress.ip_network(network) for network in ('169.254.0.0/16', '100.100.100.200/32', 'fd00:ec2::254/128'))

    @classmethod
    def read(cls, url: str, **kwargs) -> bytes:
        """Fetch `url` and return its body, checking every hop with `check_url` and capping the size."""
        kwargs.pop('allow_redirects', None)  # redirects are followed below, one checked hop at a time
        kwargs.pop('stream', None)  # the body is streamed so that the size cap can be enforced while reading
        max_size = cls._max_download_size()
        retries = Retry(total=cls.RETRY_TOTAL, backoff_factor=1, allowed_methods=['GET'])
        with requests.Session() as session:
            session.mount('http://', HTTPAdapter(max_retries=retries))
            session.mount('https://', HTTPAdapter(max_retries=retries))
            for _ in range(cls.MAX_REDIRECTS + 1):
                cls.check_url(url)
                with session.get(url, allow_redirects=False, stream=True, **kwargs) as response:
                    if not response.is_redirect:
                        response.raise_for_status()
                        return cls._read_capped(response, url, max_size)
                    location = response.headers['location']
                url = urljoin(url, location)
        raise ValueError(f'The URL {url!r} exceeded the limit of {cls.MAX_REDIRECTS} redirects.')

    @classmethod
    def _read_capped(cls, response: requests.Response, url: str, max_size: int) -> bytes:
        if max_size <= 0:
            return response.content
        content_length = response.headers.get('content-length')
        if content_length is not None and content_length.isdigit() and int(content_length) > max_size:
            cls._raise_too_large(url, max_size)
        res = bytearray()
        for chunk in response.iter_content(cls.CHUNK_SIZE):
            res += chunk
            # A response may omit or understate content-length, so the cap is also enforced while reading.
            if len(res) > max_size:
                cls._raise_too_large(url, max_size)
        return bytes(res)

    @staticmethod
    def _raise_too_large(url: str, max_size: int) -> None:
        raise ValueError(f'Refusing to fetch {url!r}: the response is larger than '
                         f'{max_size / 1024 ** 2:.0f}MB. Raise or disable this cap with the environment '
                         'variable `SWIFT_MAX_DOWNLOAD_SIZE_MB` (`0` disables it).')

    @classmethod
    def _max_download_size(cls) -> int:
        size_mb = get_env_args('swift_max_download_size_mb', float, cls.DEFAULT_MAX_DOWNLOAD_SIZE_MB)
        return int(size_mb * 1024**2)

    @classmethod
    def check_url(cls, url: str) -> None:
        """Raise a ValueError if `url` must not be fetched on behalf of an untrusted caller."""
        if not isinstance(url, str):
            raise ValueError(f'Refusing to fetch {url!r}: the URL must be a string.')
        if '\\' in url or any(ord(char) < 32 or ord(char) == 127 for char in url):
            raise ValueError(f'Refusing to fetch {url!r}: ambiguous URL characters are not allowed.')
        try:
            # Validate the URL after applying requests' own normalization. Otherwise urllib.parse and requests
            # can disagree about the authority (for example, a backslash before `@` can move the real host).
            prepared_url = requests.Request('GET', url).prepare().url
        except (requests.RequestException, UnicodeError) as e:
            raise ValueError(f'Refusing to fetch {url!r}: the URL is invalid ({e}).') from e
        parsed = urlparse(prepared_url)
        if parsed.scheme.lower() not in cls.ALLOWED_SCHEMES:
            raise ValueError(f'Refusing to fetch {url!r}: only '
                             f'{"/".join(scheme + "://" for scheme in cls.ALLOWED_SCHEMES)} URLs are supported.')
        host = (parsed.hostname or '').rstrip('.').lower()
        if not host:
            raise ValueError(f'Refusing to fetch {url!r}: the URL does not contain a host.')
        if host in cls.METADATA_HOSTS:
            cls._raise_metadata(url, f'{host!r} is a cloud instance metadata endpoint')
        allowed_hosts = cls._get_allowed_hosts()
        if allowed_hosts is not None and host not in allowed_hosts:
            raise ValueError(f'Refusing to fetch {url!r}: the host {host!r} is not listed in '
                             'the `SWIFT_URL_ALLOWED_HOSTS` environment variable.')
        # Resolve allowlisted hosts too: the allowlist may trust a private storage endpoint, but cloud metadata
        # networks are never trusted, even if an IP literal or a DNS alias was accidentally put in the list.
        for ip in cls._resolve(host):
            cls._check_ip(url, ip, allow_internal=allowed_hosts is not None)

    @classmethod
    def _check_ip(cls, url: str, ip: IpAddress, allow_internal: bool = False) -> None:
        if getattr(ip, 'ipv4_mapped', None) is not None:
            ip = ip.ipv4_mapped  # e.g. ::ffff:169.254.169.254
        for network in cls.METADATA_NETWORKS:
            if ip.version == network.version and ip in network:
                cls._raise_metadata(url, f'it resolves to {ip}, inside the metadata range {network}')
        if allow_internal or cls._allow_internal_url():
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
