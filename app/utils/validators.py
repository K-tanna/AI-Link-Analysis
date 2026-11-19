"""URL validation and SSRF protection.

Every URL we fetch (the submitted one, every redirect hop, every extra page) goes through
`validate_public_url`, which:
  1. normalises the URL and rejects bad schemes, embedded credentials and odd ports,
  2. resolves the hostname and rejects it if ANY resolved address is not a public IP
     (loopback, RFC1918, link-local/cloud-metadata 169.254.x.x, CGNAT, reserved, multicast...).

Known limitation (documented in README): there is a tiny window between our DNS check and the
actual connection (DNS rebinding). Mitigate in production with an egress proxy / network policy.
"""
import asyncio
import ipaddress
import socket
from typing import Optional, Union
from urllib.parse import urlsplit, urlunsplit

from app.errors import BlockedURLError, InvalidURLError

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {80, 443}
MAX_URL_LENGTH = 2048

IPAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]


def _parse_ip(value: str) -> Optional[IPAddress]:
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def is_public_ip(ip: IPAddress) -> bool:
    if ip.version == 6 and ip.ipv4_mapped is not None:  # ::ffff:127.0.0.1 etc.
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def normalize_url(raw: str) -> str:
    """Syntactic validation + normalisation (no network access)."""
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidURLError("A website URL is required.")
    raw = raw.strip()
    if len(raw) > MAX_URL_LENGTH:
        raise InvalidURLError("URL is too long.")
    if any(c.isspace() or ord(c) < 32 for c in raw):
        raise InvalidURLError("URL must not contain whitespace or control characters.")
    if "://" not in raw:
        raw = "https://" + raw  # be forgiving: "example.com" -> "https://example.com"

    try:
        parts = urlsplit(raw)
        host = parts.hostname
        port = parts.port
    except ValueError:
        raise InvalidURLError("URL is malformed.")

    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise InvalidURLError("Only http:// and https:// URLs are supported.")
    if not host:
        raise InvalidURLError("URL has no hostname.")
    if parts.username is not None or parts.password is not None:
        raise InvalidURLError("URLs with embedded credentials are not allowed.")

    effective_port = port or (443 if scheme == "https" else 80)
    if effective_port not in ALLOWED_PORTS:
        raise BlockedURLError("Only ports 80 and 443 are allowed.")

    try:
        host_ascii = host.encode("idna").decode("ascii")
    except UnicodeError:
        raise InvalidURLError("Hostname is invalid.")

    if _parse_ip(host_ascii) is None and "." not in host_ascii:
        raise InvalidURLError("Please provide a full domain name, e.g. example.com.")

    netloc = f"[{host_ascii}]" if ":" in host_ascii else host_ascii
    if port:
        netloc += f":{port}"
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))


async def resolve_host(host: str, port: int) -> list[str]:
    """Resolve a hostname to IP strings. Separate function so tests can patch it."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


async def validate_public_url(raw: str) -> str:
    """Return the normalised URL if it is safe to fetch, else raise InvalidURLError/BlockedURLError."""
    url = normalize_url(raw)
    parts = urlsplit(url)
    host = parts.hostname
    port = parts.port or (443 if parts.scheme == "https" else 80)

    literal = _parse_ip(host)
    if literal is not None:
        addresses = [str(literal)]
    else:
        try:
            addresses = await resolve_host(host, port)
        except (socket.gaierror, UnicodeError, OSError):
            raise InvalidURLError(f"Could not resolve the domain '{host}'. Check the URL and try again.")

    if not addresses:
        raise InvalidURLError(f"Could not resolve the domain '{host}'.")

    for addr in addresses:
        ip = _parse_ip(addr.split("%")[0])  # strip IPv6 zone id
        if ip is None or not is_public_ip(ip):
            raise BlockedURLError(
                "This URL points to a private, loopback or otherwise non-public address and cannot be analysed."
            )
    return url
