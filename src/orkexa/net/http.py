"""An ``httpx`` client behind the egress guard (ADR 0017). Needs ``pip install "orkexa[web]"``.

:func:`guarded_client` returns an ``httpx.AsyncClient`` whose only transport is :class:`GuardedTransport`:

* every request URL is checked by :meth:`EgressPolicy.check_url <orkexa.net.guard.EgressPolicy.check_url>`;
* connections are opened by :class:`~orkexa.net.guard.GuardedBackend` (resolve once, check every address,
  connect to the checked address, TLS with the host name);
* proxies from the environment are ignored, the client never follows redirects by itself
  (:func:`send_following_redirects` does, checking each hop), compressed answers are refused (``identity`` is
  requested, so the bytes counted are the bytes parsed), and the body is cut off with ``response_too_large``
  once more than :attr:`EgressPolicy.max_bytes` bytes arrived;
* :func:`deadline` bounds a whole exchange, name lookup and redirects included.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
import ssl
from typing import Any, cast

try:
    import anyio
    import httpcore
    import httpx
except ImportError as exc:  # pragma: no cover - exercised by the core suite without extras
    from ..errors import MissingDependencyError

    raise MissingDependencyError('orkexa.net.http needs httpx: pip install "orkexa[web]"') from exc

from .guard import EgressError, EgressPolicy, GuardedBackend, Resolver, redact_url

__all__ = ["GuardedTransport", "deadline", "guarded_client", "send_following_redirects"]


class _CappedStream(httpx.AsyncByteStream):
    """A response body that raises ``response_too_large`` once more than ``limit`` bytes arrived."""

    def __init__(self, stream: Any, limit: int) -> None:
        self._stream = stream
        self._limit = limit

    async def __aiter__(self) -> AsyncIterator[bytes]:
        total = 0
        async for chunk in self._stream:
            total += len(chunk)
            if total > self._limit:
                raise EgressError(
                    "response_too_large",
                    f"The answer is larger than {self._limit} bytes and was not read to the end",
                    limit=self._limit,
                )
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()


class _PinnedTransport(httpx.AsyncHTTPTransport):
    """``httpx``'s transport with a pool that opens every connection through a :class:`GuardedBackend`."""

    def __init__(self, backend: GuardedBackend, verify: ssl.SSLContext | bool) -> None:
        super().__init__(trust_env=False, verify=verify, retries=0, proxy=None)
        ssl_context = self._pool._ssl_context
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl_context,
            max_connections=8,
            max_keepalive_connections=4,
            network_backend=cast(Any, backend),
        )


class GuardedTransport(httpx.AsyncBaseTransport):
    """The transport of :func:`guarded_client` (see the module documentation).

    ``mock_transport`` replaces the network part for tests (``httpx.MockTransport``): URL checks, the encoding
    rule and the size limit still apply, but nothing is resolved or connected. Never pass a real network
    transport there; it would bypass the address checks.
    """

    def __init__(
        self,
        policy: EgressPolicy,
        *,
        resolver: Resolver | None = None,
        verify: ssl.SSLContext | bool = True,
        mock_transport: httpx.AsyncBaseTransport | None = None,
        backend: GuardedBackend | None = None,
    ) -> None:
        self.policy = policy
        self.backend = backend or GuardedBackend(
            allow_loopback=policy.allow_private, allow_private=policy.allow_private
        )
        if resolver is not None:
            self.backend.resolver = resolver
        self._inner = mock_transport if mock_transport is not None else _PinnedTransport(self.backend, verify)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        origin = self.policy.check_url(str(request.url))
        if origin.host != request.url.raw_host.decode("ascii", "replace").lower().rstrip(".").strip("[]"):
            # Two URL parsers disagree about the host: refuse rather than guess which one the connection uses.
            raise EgressError("url_invalid", f"Ambiguous host in {redact_url(str(request.url))}")
        request.headers["Accept-Encoding"] = "identity"
        response = await self._inner.handle_async_request(request)
        self.backend.statuses.append(int(response.status_code))
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if encoding not in ("", "identity"):
            await response.aclose()
            raise EgressError(
                "content_encoding",
                f"The server sent a compressed answer ({encoding}), which is not accepted",
                encoding=encoding,
            )
        # A new response, so that no body read in advance (as httpx.MockTransport does) escapes the limit.
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            stream=_CappedStream(response.stream, self.policy.max_bytes),
            extensions=response.extensions,
            request=request,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


def guarded_client(
    policy: EgressPolicy,
    *,
    headers: dict[str, str] | None = None,
    resolver: Resolver | None = None,
    verify: ssl.SSLContext | bool = True,
    mock_transport: httpx.AsyncBaseTransport | None = None,
    backend: GuardedBackend | None = None,
) -> httpx.AsyncClient:
    """An ``httpx.AsyncClient`` that can only reach what ``policy`` allows (see the module documentation).

    Use it as ``async with guarded_client(policy) as client: ...`` and wrap each exchange in :func:`deadline`.
    ``resolver`` replaces the system's name lookup (tests); ``mock_transport`` see :class:`GuardedTransport`;
    ``backend`` replaces the default :class:`GuardedBackend` (one built from the policy's ``allow_private``).
    """
    transport = GuardedTransport(
        policy, resolver=resolver, verify=verify, mock_transport=mock_transport, backend=backend
    )
    return httpx.AsyncClient(
        transport=transport,
        trust_env=False,
        follow_redirects=False,
        timeout=httpx.Timeout(policy.timeout),
        headers=headers,
    )


@contextmanager
def deadline(policy: EgressPolicy) -> Iterator[None]:
    """Bound everything inside to :attr:`EgressPolicy.timeout` seconds; raises ``EgressError("timeout")``."""
    try:
        with anyio.fail_after(policy.timeout):
            yield
    except TimeoutError:
        raise EgressError("timeout", f"The request took longer than {policy.timeout:g} seconds") from None


async def send_following_redirects(
    client: httpx.AsyncClient, request: httpx.Request, policy: EgressPolicy
) -> tuple[httpx.Response, list[str]]:
    """Send ``request`` (streamed) and follow at most :attr:`EgressPolicy.max_redirects` redirects by hand.

    Each hop is checked by the client's :class:`GuardedTransport` before it is sent (URL and every resolved
    address). ``httpx`` builds each next request (method change for 301/302/303, ``Authorization`` dropped on
    a change of origin). Returns the final, unread response (the caller reads and closes it) and the URLs that
    redirected, in order.
    """
    redirects: list[str] = []
    response = await client.send(request, stream=True, follow_redirects=False)
    while response.is_redirect:
        next_request = response.next_request
        await response.aclose()
        if next_request is None:
            raise EgressError("redirect_invalid", "A redirect without a usable Location header")
        if len(redirects) >= policy.max_redirects:
            raise EgressError(
                "too_many_redirects",
                f"More than {policy.max_redirects} redirects",
                limit=policy.max_redirects,
            )
        redirects.append(str(response.request.url))
        policy.check_url(str(next_request.url))  # a clear error before anything else happens
        response = await client.send(next_request, stream=True, follow_redirects=False)
    return response, redirects
