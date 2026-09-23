"""Fetch a web page and extract its main text, with SSRF guarding.

Every hostname (the original URL and every redirect hop) is resolved and
checked against private/loopback/link-local/metadata IP ranges before a
request is sent, per plan.md 5.3's "权限最小化" rule.
"""

import ipaddress
import re
import socket
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
import trafilatura

ALLOWED_SCHEMES = ("http", "https")
BLOCKED_HOSTNAMES = {"localhost"}
CLOUD_METADATA_IPS = {"169.254.169.254"}

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "ref", "mc_cid", "mc_eid",
}

_SUSPICIOUS_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore (all |the )?(previous|prior|above) instructions",
        r"disregard (all |the )?(previous|prior|above) (instructions|prompt)",
        r"you are now",
        r"system prompt",
        r"new instructions?:",
        r"act as (if|an?)\b.*\b(assistant|ai|model)\b",
    ]
]


def canonicalize_url(url: str) -> str:
    """Normalize a URL for dedup: lowercase scheme/host, strip fragment and
    trailing slash, drop known tracking query params, sort remaining ones.
    """
    parsed = urlparse(url)
    query_pairs = sorted(
        (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() not in _TRACKING_PARAMS
    )
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", urlencode(query_pairs), ""))


def contains_suspicious_pattern(text: str) -> bool:
    """Lightweight regex scan for prompt-injection-style text (plan.md 5.3).

    Not a security guarantee — a hit just downgrades the source's
    credibility score rather than blocking it outright.
    """
    return any(p.search(text) for p in _SUSPICIOUS_PATTERNS)


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
