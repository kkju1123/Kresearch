"""Fetch a web page and extract its main text, with SSRF guarding.

Every hostname (the original URL and every redirect hop) is resolved and
checked against private/loopback/link-local/metadata IP ranges before a
request is sent, per plan.md 5.3's "权限最小化" rule.
"""

import ipaddress
import socket
from urllib.parse import urlparse

import httpx
import trafilatura

ALLOWED_SCHEMES = ("http", "https")
BLOCKED_HOSTNAMES = {"localhost"}
CLOUD_METADATA_IPS = {"169.254.169.254"}


class UnsafeURLError(Exception):
    pass


def _is_blocked_ip(ip_str: str) -> bool:
    if ip_str in CLOUD_METADATA_IPS:
        return True
    ip = ipaddress.ip_address(ip_str)
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def check_url(url: str) -> None:
    """Raise UnsafeURLError if the URL's scheme or resolved IP is unsafe."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"blocked scheme: {parsed.scheme!r}")
    hostname = parsed.hostname
    if not hostname:
        raise UnsafeURLError("URL has no hostname")
    if hostname.lower() in BLOCKED_HOSTNAMES:
        raise UnsafeURLError(f"blocked hostname: {hostname}")
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"cannot resolve hostname: {hostname}") from exc
    for _family, _type, _proto, _canonname, sockaddr in infos:
        ip_str = sockaddr[0]
        if _is_blocked_ip(ip_str):
            raise UnsafeURLError(f"blocked IP {ip_str} for hostname {hostname}")


async def fetch_and_extract(url: str, max_redirects: int = 5, timeout: float = 15.0) -> tuple[str, str]:
    """Fetch `url`, following redirects manually (re-checked each hop).

    Returns (extracted_main_text, final_url). Raises UnsafeURLError if any
    hop resolves to a disallowed address, httpx.HTTPStatusError on a non-2xx
    final response.
    """
    check_url(url)
    async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as client:
        request = client.build_request("GET", url)
        for _ in range(max_redirects + 1):
            response = await client.send(request)
            if response.is_redirect:
                next_request = response.next_request
                if next_request is None:
                    raise UnsafeURLError("redirect response without a location")
                check_url(str(next_request.url))
                request = next_request
                continue
            response.raise_for_status()
            html = response.text
            final_url = str(response.url)
            break
        else:
            raise UnsafeURLError(f"too many redirects (> {max_redirects}) for {url}")

    text = trafilatura.extract(html) or ""
    return text, final_url
