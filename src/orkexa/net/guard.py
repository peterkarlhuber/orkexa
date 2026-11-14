"""The egress guard: which hosts and addresses an outbound HTTP request may reach (ADR 0017).

Two layers, both needed:

* :class:`EgressPolicy` checks a URL before anything is sent: scheme, port, user info, host-name syntax
  (numeric forms such as ``2130706433``, ``0x7f000001``, ``127.1`` or ``0177.0.0.1`` are refused), the
  operator's allow and deny lists, cloud metadata names and literal addresses.
* :class:`GuardedBackend` is an ``httpcore`` network backend. Its connect step resolves the host name once,
  checks **every** resolved address with :func:`address_problem` and connects to an address that passed. The
  address checked is the address connected, so DNS rebinding (answers that change between lookups) cannot slip
  an internal address in between the check and the connect. TLS still uses the host name (SNI and certificate
  verification); only the socket uses the checked address.

Redirects are never followed by the HTTP client: :mod:`orkexa.net.http` follows them by hand and every hop
passes both layers again. Proxies from the environment are ignored (``trust_env=False``).

This module imports only the standard library; ``httpcore`` and ``anyio`` are imported when a connection is
opened. :class:`GuardedBackend` works with ``httpcore`` and API-compatible forks (``httpcore_module``), and
its errors can be replaced (:meth:`GuardedBackend.error`), so a host application can reuse it unchanged.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
import importlib
import ipaddress
import re
import socket
from typing import Any, NamedTuple
from urllib.parse import parse_qsl, urlsplit

from ..errors import OrkexaError

__all__ = [
    "DEFAULT_MAX_BYTES",
    "LOOPBACK_HOSTS",
    "METADATA_ADDRESSES",
    "METADATA_HOSTS",
    "PRIVATE_NETWORKS",
    "EgressError",
    "EgressPolicy",
    "GuardedBackend",
    "IPAddress",
    "Origin",
    "Resolver",
    "address_problem",
    "host_problem",
    "redact_url",
    "system_resolver",
    "url_secret_problem",
]

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

#: Explicitly refused, whatever range they are in and whatever the policy says (cloud metadata services).
METADATA_ADDRESSES = frozenset(
    ipaddress.ip_address(value)
    for value in ("169.254.169.254", "169.254.170.2", "169.254.170.23", "fd00:ec2::254", "100.100.100.200")
)
#: Cloud metadata host names; refused by every policy, before any lookup.
METADATA_HOSTS = frozenset({"metadata", "metadata.google.internal", "metadata.azure.com", "instance-data"})
#: Names of this computer (RFC 6761 also reserves every name under ``localhost``).
LOOPBACK_HOSTS = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"})
#: The ranges ``allow_private=True`` opens: loopback, RFC 1918, shared address space (RFC 6598, e.g.
#: carrier-grade NAT or VPN overlays) and IPv6 unique local addresses. Never link-local, metadata, multicast
#: or reserved ranges.
PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in (
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "100.64.0.0/10",
        "::1/128",
        "fc00::/7",
    )
)
#: The default for :attr:`EgressPolicy.max_bytes` (5 MiB).
DEFAULT_MAX_BYTES = 5 * 1024 * 1024

_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_NAT64_LOCAL = ipaddress.ip_network("64:ff9b:1::/48")
_TEREDO = ipaddress.ip_network("2001::/32")
_SHARED = ipaddress.ip_network("100.64.0.0/10")

Resolver = Callable[[str, int], Awaitable[list[str]]]
"""``async (host, port) -> [address, ...]``: the name lookup :class:`GuardedBackend` uses."""


class EgressError(OrkexaError, RuntimeError):
    """An outbound request was refused or stopped by the egress guard.

    ``code`` says why:

    * ``url_invalid``: not an absolute http(s) URL, or a host name with characters a DNS name cannot have.
    * ``url_credentials``: the URL carries a user name or password.
    * ``scheme_refused`` / ``port_refused``: not in :attr:`EgressPolicy.schemes` / :attr:`EgressPolicy.ports`.
    * ``host_refused``: ``reason`` ``denied`` (deny list) or ``not_allowed`` (not on the allow list).
    * ``address_refused``: ``reason`` names the rule: ``metadata``, ``loopback``, ``private``, ``link_local``,
      ``numeric_host`` and the other reasons of :func:`address_problem`.
    * ``dns_failed``: the name has no address.
    * ``too_many_redirects`` / ``redirect_invalid``: more hops than :attr:`EgressPolicy.max_redirects`, or a
      redirect without a usable ``Location``.
    * ``response_too_large``: more than :attr:`EgressPolicy.max_bytes` bytes arrived.
    * ``content_encoding``: the server compressed the answer although ``identity`` was requested.
    * ``timeout``: the request took longer than :attr:`EgressPolicy.timeout` (DNS, connect, redirects, body).

    Messages never contain query strings or fragments of URLs.
    """

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details

    @property
    def reason(self) -> str | None:
        """The rule that refused an address or host, if any."""
        value = self.details.get("reason")
        return None if value is None else str(value)


# ---- address rules -----------------------------------------------------------------------------------------


def _embedded_v4(address: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 address carries (mapped, NAT64, 6to4, IPv4-compatible), if any."""
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address in _NAT64 or address in _NAT64_LOCAL:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    if address.sixtofour is not None:
        return address.sixtofour
    packed = int(address)
    if (
        packed >> 32 == 0 and packed > 1
    ):  # IPv4-compatible ::a.b.c.d (deprecated, still routable by some stacks)
        return ipaddress.IPv4Address(packed)
    return None


def _in_private_networks(address: IPAddress) -> bool:
    if isinstance(address, ipaddress.IPv6Address):
        embedded = _embedded_v4(address)
        if embedded is not None:
            return _in_private_networks(embedded)
    return any(address in network for network in PRIVATE_NETWORKS)


def address_problem(
    address: IPAddress, *, allow_loopback: bool = False, allow_private: bool = False
) -> str | None:
    """Why ``address`` may not be connected to, or ``None`` when it is a global unicast address.

    ``allow_loopback`` permits loopback addresses only (a local development server with consent).
    ``allow_private`` permits the ranges of :data:`PRIVATE_NETWORKS` (a self-hosted service on a LAN);
    metadata, link-local, unspecified, multicast and reserved addresses stay refused.

    Reasons: ``metadata``, ``scoped``, ``loopback``, ``unspecified``, ``multicast``, ``link_local``,
    ``site_local``, ``private``, ``shared``, ``reserved``, ``not_global``. IPv6 addresses that embed an IPv4
    address (mapped, NAT64, 6to4, IPv4-compatible) are judged by that IPv4 address; Teredo (``2001::/32``) is
    refused as ``private``.
    """
    reason = _address_reason(address, allow_loopback=allow_loopback)
    if reason in ("loopback", "private", "shared") and allow_private and _in_private_networks(address):
        return None
    return reason


def _address_reason(address: IPAddress, *, allow_loopback: bool) -> str | None:
    if address in METADATA_ADDRESSES:
        return "metadata"
    if isinstance(address, ipaddress.IPv6Address):
        if address.scope_id:
            return "scoped"
        embedded = _embedded_v4(address)
        if embedded is not None:
            return _address_reason(embedded, allow_loopback=allow_loopback)
        if address in _TEREDO:
            # A tunnel endpoint with an obfuscated client address; the standard library also calls it private,
            # but the rule must not depend on the Python version's tables.
            return "private"
    if address.is_loopback:
        return None if allow_loopback else "loopback"
    if address.is_unspecified:
        return "unspecified"
    if address.is_multicast:
        return "multicast"
    if address.is_link_local:
        return "link_local"
    if isinstance(address, ipaddress.IPv6Address) and address.is_site_local:
        # Deprecated site-local fec0::/10 (RFC 3879): internal by definition, but ``is_global`` says global.
        return "site_local"
    if address.is_private:
        return "private"
    if isinstance(address, ipaddress.IPv4Address) and address in _SHARED:
        return "shared"
    if address.is_reserved:
        return "reserved"
    if not address.is_global:
        return "not_global"
    return None


def _literal(host: str) -> IPAddress | None:
    try:
        return ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return None


# A label made of digits or a 0x hex number: getaddrinfo (inet_aton) reads hosts such as 2130706433,
# 0x7f000001, 127.1 or 0177.0.0.1 as IPv4 addresses, and so do browsers (WHATWG URL "ends in a number").
_NUMERIC_LABEL = re.compile(r"^(?:0x[0-9a-f]*|[0-9]+)$")
_HOST_NAME = re.compile(r"^[a-z0-9_-]+(?:\.[a-z0-9_-]+)*\.?$")


def _normal_host(host: str) -> str:
    """``host`` in lower case ASCII (IDNA) without a trailing dot; ``ValueError`` if it is no host name."""
    host = host.strip().lower()
    if ":" in host:  # an IPv6 literal, with or without brackets (httpx gives it without)
        if _literal(host) is None:
            raise ValueError(host)
        return host.strip("[]")
    try:
        ascii_host = host.encode("idna").decode("ascii") if not host.isascii() else host
    except UnicodeError:
        raise ValueError(host) from None
    if not _HOST_NAME.match(ascii_host):
        raise ValueError(host)
    return ascii_host.rstrip(".")


def host_problem(host: str) -> str | None:
    """Why the host name ``host`` (no IP literal) is refused before any lookup, or ``None``.

    ``metadata`` for cloud metadata names, ``loopback`` for names of this computer, ``numeric_host`` for a
    host whose last label is a number (other spellings of an IPv4 address). Expects a normalized host.
    """
    if host in METADATA_HOSTS:
        return "metadata"
    if host in LOOPBACK_HOSTS or host.endswith(".localhost"):
        return "loopback"
    if _NUMERIC_LABEL.match(host.rsplit(".", 1)[-1]):
        return "numeric_host"
    return None


def _host_matches(host: str, patterns: frozenset[str]) -> bool:
    """``host`` is one of ``patterns``; a pattern ``*.example.com`` matches every name below example.com."""
    for pattern in patterns:
        if pattern.startswith("*."):
            if host.endswith(pattern[1:]):
                return True
        elif host == pattern:
            return True
    return False


# ---- request-level policy ----------------------------------------------------------------------------------


class Origin(NamedTuple):
    """The normalized scheme, host and port of a checked URL."""

    scheme: str
    host: str
    port: int


def _host_set(values: Iterable[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    result = set()
    for value in values:
        wildcard = value.strip().startswith("*.")
        name = _normal_host(value.strip()[2:] if wildcard else value)
        result.add(f"*.{name}" if wildcard else name)
    return frozenset(result)


@dataclass(frozen=True)
class EgressPolicy:
    """Which URLs an outbound request may reach, and its limits.

    * ``schemes``: allowed URL schemes (``http`` and ``https``; ``{"https"}`` for API clients).
    * ``allow_private``: permit :data:`PRIVATE_NETWORKS` and names of this computer (a self-hosted service).
      Metadata names and addresses stay refused.
    * ``allow_hosts``: when set, only these host names (``*.example.com`` for a domain and its subdomains); IP
      literals must be listed as written. ``None`` means any public host.
    * ``deny_hosts``: refused host names (same syntax), besides :data:`METADATA_HOSTS`, which every policy
      refuses.
    * ``ports``: allowed ports (the scheme's default port counts); ``None`` allows any port.
    * ``max_redirects``: redirects followed by :mod:`orkexa.net.http` (each hop is checked again).
    * ``max_bytes``: the most bytes of one response body, counted on the wire while it is read.
    * ``timeout``: seconds for the whole request: name lookup, connect, every redirect and the body.
    """

    schemes: frozenset[str] = frozenset({"http", "https"})
    allow_private: bool = False
    allow_hosts: frozenset[str] | None = None
    deny_hosts: frozenset[str] = frozenset()
    ports: frozenset[int] | None = frozenset({80, 443})
    max_redirects: int = 5
    max_bytes: int = DEFAULT_MAX_BYTES
    timeout: float = 30.0

    def __post_init__(self) -> None:
        schemes = frozenset(scheme.lower() for scheme in self.schemes)
        if not schemes or not schemes <= {"http", "https"}:
            raise ValueError("EgressPolicy.schemes must be a non-empty subset of {'http', 'https'}")
        if self.max_redirects < 0 or self.max_bytes < 1 or self.timeout <= 0:
            raise ValueError("EgressPolicy limits must be positive (max_redirects may be 0)")
        if self.ports is not None and any(not 0 < port < 65536 for port in self.ports):
            raise ValueError("EgressPolicy.ports must be TCP ports (1-65535)")
        object.__setattr__(self, "schemes", schemes)
        object.__setattr__(self, "allow_hosts", _host_set(self.allow_hosts))
        object.__setattr__(self, "deny_hosts", _host_set(self.deny_hosts) or frozenset())
        if self.ports is not None:
            object.__setattr__(self, "ports", frozenset(self.ports))

    def check_url(self, url: str) -> Origin:
        """Refuse a URL that must never be requested (:class:`EgressError`); returns its normalized origin.

        The host's addresses are checked later, when :class:`GuardedBackend` connects."""
        shown = redact_url(url)
        try:
            parts = urlsplit(url.strip())
            scheme = parts.scheme.lower()
            raw_host = parts.hostname or ""
            port = parts.port
        except ValueError:
            raise EgressError("url_invalid", f"Not a valid address: {shown}", url=shown) from None
        if scheme not in ("http", "https") or not raw_host:
            raise EgressError("url_invalid", f"Not a valid http(s) address: {shown}", url=shown)
        if parts.username is not None or parts.password is not None:
            raise EgressError(
                "url_credentials", "The address must not contain a user name or password", url=shown
            )
        if scheme not in self.schemes:
            raise EgressError(
                "scheme_refused", f"The scheme {scheme} is not allowed here", url=shown, scheme=scheme
            )
        try:
            host = _normal_host(raw_host)
        except ValueError:
            raise EgressError("url_invalid", f"Not a valid host name: {shown}", url=shown) from None
        port = port if port is not None else (443 if scheme == "https" else 80)
        if self.ports is not None and port not in self.ports:
            raise EgressError("port_refused", f"The port {port} is not allowed", url=shown, port=port)
        self.check_host(host)
        return Origin(scheme, host, port)

    def check_host(self, host: str) -> None:
        """The host rules of :meth:`check_url` for a normalized host name or IP literal."""
        literal = _literal(host)
        if literal is None:
            reason = host_problem(host)
            if reason == "loopback" and self.allow_private:
                reason = None
            if reason is not None:
                raise EgressError(
                    "address_refused", f"The address {host} is refused", host=host, reason=reason
                )
        else:
            reason = address_problem(literal, allow_private=self.allow_private)
            if reason is not None:
                raise EgressError(
                    "address_refused", f"The address {host} is refused", host=host, reason=reason
                )
            host = str(literal)
        if _host_matches(host, self.deny_hosts):
            raise EgressError(
                "host_refused", f"The host {host} is on the deny list", host=host, reason="denied"
            )
        if self.allow_hosts is not None and not _host_matches(host, self.allow_hosts):
            raise EgressError(
                "host_refused", f"The host {host} is not on the allow list", host=host, reason="not_allowed"
            )


# ---- connection-level guard --------------------------------------------------------------------------------


async def system_resolver(host: str, port: int) -> list[str]:
    """The operating system's name lookup (``getaddrinfo``), run without blocking the event loop."""
    import anyio

    infos = await anyio.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


@dataclass
class GuardedBackend:
    """An ``httpcore`` network backend that connects only to checked addresses (see the module documentation).

    ``allow_loopback`` and ``allow_private`` go to :func:`address_problem`. ``httpcore_module`` names the
    ``httpcore`` package (or an API-compatible fork) whose exceptions and stream class are used. Refusals are
    built by :meth:`error`; a subclass may return its own exception type.
    """

    allow_loopback: bool = False
    resolver: Resolver = system_resolver
    #: Every address connected to, for tests and the audit trail.
    connected: list[str] = field(default_factory=list)
    #: HTTP status codes of the answers, for hosts that record them (the backend itself does not).
    statuses: list[int] = field(default_factory=list)
    allow_private: bool = False
    httpcore_module: str = "httpcore"

    def error(self, code: str, message: str, **details: Any) -> Exception:
        """The exception raised for a refusal (``code`` ``address_refused`` or ``dns_failed``)."""
        return EgressError(code, message, **details)

    async def resolve(self, host: str, port: int) -> list[IPAddress]:
        """Every address of ``host``, each checked; one refused answer refuses the name."""
        literal = _literal(host)
        if literal is not None:
            candidates = [literal]
        else:
            if host.lower().rstrip(".") in METADATA_HOSTS:
                raise self.error(
                    "address_refused", f"The address {host} is refused", host=host, reason="metadata"
                )
            try:
                answers = await self.resolver(host, port)
            except OSError as exc:
                raise self.error("dns_failed", f"The name {host} cannot be resolved", host=host) from exc
            candidates = []
            for answer in answers:
                parsed = _literal(answer.split("%")[0]) if "%" in answer else _literal(answer)
                if parsed is None:
                    continue
                if "%" in answer:  # a scoped (link-local) answer
                    raise self.error(
                        "address_refused", f"{host} resolves to a scoped address", host=host, reason="scoped"
                    )
                candidates.append(parsed)
        if not candidates:
            raise self.error("dns_failed", f"The name {host} has no address", host=host)
        for candidate in candidates:
            reason = address_problem(
                candidate, allow_loopback=self.allow_loopback, allow_private=self.allow_private
            )
            if reason is not None:
                # One refused answer refuses the name (mixed public and internal answers are an attack).
                raise self.error(
                    "address_refused",
                    f"{host} resolves to a refused address ({reason})",
                    host=host,
                    reason=reason,
                )
        return candidates

    async def open_stream(self, address: str, port: int, local_address: str | None) -> Any:
        """Open a TCP connection to the checked literal ``address`` (an ``anyio`` byte stream; no lookup)."""
        import anyio

        return await anyio.connect_tcp(address, port, local_host=local_address)

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> Any:
        import anyio

        httpcore = importlib.import_module(self.httpcore_module)
        stream_class = importlib.import_module(f"{self.httpcore_module}._backends.anyio").AnyIOStream

        addresses = await self.resolve(host, port)
        last: Exception | None = None
        for address in addresses:
            try:
                with anyio.fail_after(timeout):
                    stream = await self.open_stream(str(address), port, local_address)
            except TimeoutError as exc:
                last = httpcore.ConnectTimeout(str(exc))
                continue
            except OSError as exc:
                last = httpcore.ConnectError(str(exc))
                continue
            for option in socket_options or ():
                stream._raw_socket.setsockopt(*option)
            # The connected peer is the checked address: anyio connects to the literal address, no lookup.
            self.connected.append(str(address))
            return stream_class(stream)
        assert last is not None
        raise last

    async def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: Any = None
    ) -> Any:
        raise self.error("address_refused", "Unix sockets are refused", host=path, reason="unix")

    async def sleep(self, seconds: float) -> None:
        import anyio

        await anyio.sleep(seconds)


# ---- URLs in messages --------------------------------------------------------------------------------------

_SECRET_NAMES = re.compile(
    r"(key|token|secret|passw|pwd|auth|signature|sig$|credential|session|bearer|jwt|access|private|code$)",
    re.I,
)
_SECRET_VALUE = re.compile(r"^[A-Za-z0-9_\-.~+/=]{20,}$")


def _looks_secret(value: str) -> bool:
    """A long token-like value: at least 20 characters of a token alphabet with letters and digits mixed."""
    value = value.strip()
    if not _SECRET_VALUE.match(value):
        return False
    return any(c.isdigit() for c in value) and any(c.isalpha() for c in value)


def url_secret_problem(url: str) -> str | None:
    """Why ``url`` seems to carry a credential (``query``, ``path``, ``fragment``), or ``None``.

    For addresses that are stored and shown in the clear: query parameters with credential-like names or
    token-like values, long token-like path segments and any fragment count."""
    parts = urlsplit(url.strip())
    if parts.fragment:
        return "fragment"
    for name, value in parse_qsl(parts.query, keep_blank_values=True):
        if _SECRET_NAMES.search(name) or _looks_secret(value):
            return "query"
    if parts.query and not parse_qsl(parts.query, keep_blank_values=True) and _looks_secret(parts.query):
        return "query"
    for segment in parts.path.split("/"):
        if len(segment) >= 24 and _looks_secret(segment):
            return "path"
    return None


def redact_url(url: str) -> str:
    """``url`` without user info, query and fragment, for events, logs and error texts."""
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        return "(invalid address)"
    if not parts.scheme or not parts.netloc:
        return url.split("?", 1)[0].split("#", 1)[0][:200]
    if ":" in host:
        host = f"[{host}]"
    path = "/".join(
        "…" if len(segment) >= 24 and _looks_secret(segment) else segment for segment in parts.path.split("/")
    )
    hidden = "?…" if parts.query else ""
    return f"{parts.scheme}://{host}{port}{path}{hidden}"
