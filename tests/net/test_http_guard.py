"""The SSRF test catalog, part 2: HTTP exchanges through ``orkexa.net.http`` (ADR 0017).

A loopback test server stands in for "the internet": the backend's :meth:`GuardedBackend.open_stream` is
redirected to it, so the names and addresses the guard checks are the real ones of the test (for example
``93.184.216.34``) while no packet leaves the machine.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from typing import Any

import pytest

pytest.importorskip("httpx")

import anyio
import httpx

from orkexa.net.guard import EgressError, EgressPolicy, GuardedBackend
from orkexa.net.http import deadline, guarded_client, send_following_redirects


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        self.server.requests.append((self.headers.get("Host", ""), self.path, dict(self.headers)))
        path = self.path
        if path.startswith("/redirect?to="):
            self.send_response(302)
            self.send_header("Location", path.split("=", 1)[1])
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path == "/loop":
            self.send_response(302)
            self.send_header("Location", "/loop")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path == "/big":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()  # no Content-Length: only the counting stops this body
            try:
                for _ in range(64):
                    self.wfile.write(b"x" * 16384)
            except OSError:
                pass
        elif path == "/gzip":
            body = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03"
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/slow":
            self.send_response(200)
            self.send_header("Content-Length", "10")
            self.end_headers()
            self.wfile.flush()
            time.sleep(2)
            self.wfile.write(b"0123456789")
        else:
            body = b"hello"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    requests: list[tuple[str, str, dict[str, str]]]


@pytest.fixture
def server() -> Iterator[_Server]:
    httpd = _Server(("127.0.0.1", 0), _Handler)
    httpd.requests = []
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


@dataclass
class LoopbackBackend(GuardedBackend):
    """Checks names and addresses as usual, then connects to the test server, not the checked address."""

    port: int = 0
    opened: list[str] = field(default_factory=list)

    async def open_stream(self, address: str, port: int, local_address: str | None) -> Any:
        self.opened.append(address)
        return await anyio.connect_tcp("127.0.0.1", self.port)


DNS = {
    "public.test": [["93.184.216.34"]],
    "other.test": [["93.184.216.35"]],
    "internal.test": [["10.0.0.7"]],
    "rebind.test": [["93.184.216.36"], ["127.0.0.1"]],
}


def make_backend(server: _Server, answers: dict[str, list[list[str]]] | None = None) -> LoopbackBackend:
    calls: dict[str, int] = {}
    table = answers or DNS

    async def resolve(host: str, port: int) -> list[str]:
        index = calls.get(host, 0)
        calls[host] = index + 1
        if host not in table:
            raise OSError("unknown test name")
        rounds = table[host]
        return rounds[min(index, len(rounds) - 1)]

    return LoopbackBackend(port=server.server_address[1], resolver=resolve)


def fetch(
    url: str, backend: GuardedBackend, policy: EgressPolicy | None = None
) -> tuple[int, bytes, list[str]]:
    policy = policy or EgressPolicy()

    async def go() -> tuple[int, bytes, list[str]]:
        async with guarded_client(policy, backend=backend) as client:
            with deadline(policy):
                response, redirects = await send_following_redirects(
                    client, client.build_request("GET", url), policy
                )
                try:
                    body = b"".join([chunk async for chunk in response.aiter_raw()])
                finally:
                    await response.aclose()
        return response.status_code, body, redirects

    return asyncio.run(go())


def refused(call: Any) -> EgressError:
    with pytest.raises(EgressError) as caught:
        call()
    return caught.value


def test_a_public_name_connects_to_the_checked_address(server: _Server) -> None:
    backend = make_backend(server)
    status, body, redirects = fetch("http://public.test/ok", backend)
    assert (status, body, redirects) == (200, b"hello", [])
    assert backend.opened == ["93.184.216.34"] == backend.connected
    host, path, headers = server.requests[0]
    assert (host, path) == ("public.test", "/ok")  # the Host header is the name, not the address
    assert headers["Accept-Encoding"] == "identity"


def test_a_name_resolving_to_an_internal_address_never_connects(server: _Server) -> None:
    backend = make_backend(server)
    error = refused(lambda: fetch("http://internal.test/", backend))
    assert (error.code, error.reason) == ("address_refused", "private")
    assert backend.opened == [] and server.requests == []


def test_a_302_to_loopback_is_refused(server: _Server) -> None:
    backend = make_backend(server)
    error = refused(lambda: fetch("http://public.test/redirect?to=http://127.0.0.1/admin", backend))
    assert (error.code, error.reason) == ("address_refused", "loopback")
    assert [path for _host, path, _headers in server.requests] == ["/redirect?to=http://127.0.0.1/admin"]


@pytest.mark.parametrize(
    ("target", "code", "reason"),
    [
        ("http://169.254.169.254/latest/meta-data/", "address_refused", "metadata"),
        ("http://metadata.google.internal/", "address_refused", "metadata"),
        ("http://2130706433/", "address_refused", "numeric_host"),
        ("http://[::ffff:10.0.0.1]/", "address_refused", "private"),
        ("http://internal.test/", "address_refused", "private"),
        ("http://public.test:22/", "port_refused", None),
        ("file:///etc/passwd", "url_invalid", None),
    ],
)
def test_every_redirect_hop_is_checked(server: _Server, target: str, code: str, reason: str | None) -> None:
    backend = make_backend(server)
    error = refused(lambda: fetch(f"http://public.test/redirect?to={target}", backend))
    assert (error.code, error.reason) == (code, reason)
    assert len(server.requests) == 1


def test_dns_rebinding_between_hops_is_caught(server: _Server) -> None:
    # The first lookup of rebind.test answers a public address, the second (the redirect's new connection)
    # loopback: the second connection is refused although the name is the same.
    backend = make_backend(server)
    assert fetch("http://rebind.test/ok", backend)[0] == 200
    error = refused(lambda: fetch("http://rebind.test/ok", make_backend_sharing(backend, server)))
    assert (error.code, error.reason) == ("address_refused", "loopback")


def make_backend_sharing(previous: LoopbackBackend, server: _Server) -> LoopbackBackend:
    """A new backend (a new client, so a new connection) that shares ``previous``'s resolver."""
    return LoopbackBackend(port=server.server_address[1], resolver=previous.resolver)


def test_a_redirect_to_a_rebound_name_is_refused(server: _Server) -> None:
    answers = {"public.test": [["93.184.216.34"]], "rebind.test": [["127.0.0.1"]]}
    backend = make_backend(server, answers)
    error = refused(lambda: fetch("http://public.test/redirect?to=http://rebind.test/", backend))
    assert (error.code, error.reason) == ("address_refused", "loopback")
    assert backend.opened == ["93.184.216.34"]


def test_redirects_are_followed_and_counted(server: _Server) -> None:
    backend = make_backend(server)
    status, body, redirects = fetch("http://public.test/redirect?to=http://other.test/ok", backend)
    assert (status, body) == (200, b"hello")
    assert redirects == ["http://public.test/redirect?to=http://other.test/ok"]
    assert backend.opened == ["93.184.216.34", "93.184.216.35"]
    error = refused(
        lambda: fetch("http://public.test/loop", make_backend(server), EgressPolicy(max_redirects=3))
    )
    assert error.code == "too_many_redirects"
    assert sum(1 for _host, path, _headers in server.requests if path == "/loop") == 4
    error = refused(
        lambda: fetch(
            "http://public.test/redirect?to=/ok", make_backend(server), EgressPolicy(max_redirects=0)
        )
    )
    assert error.code == "too_many_redirects"


def test_the_body_is_cut_off_at_the_size_limit(server: _Server) -> None:
    error = refused(
        lambda: fetch("http://public.test/big", make_backend(server), EgressPolicy(max_bytes=100_000))
    )
    assert error.code == "response_too_large"


def test_compressed_answers_are_refused(server: _Server) -> None:
    error = refused(lambda: fetch("http://public.test/gzip", make_backend(server)))
    assert (error.code, error.details["encoding"]) == ("content_encoding", "gzip")


def test_the_deadline_covers_name_resolution() -> None:
    async def slow(host: str, port: int) -> list[str]:
        await anyio.sleep(10)
        return ["93.184.216.34"]

    started = time.monotonic()
    error = refused(
        lambda: fetch("http://public.test/", GuardedBackend(resolver=slow), EgressPolicy(timeout=0.3))
    )
    assert error.code == "timeout"
    assert time.monotonic() - started < 5


def test_the_deadline_covers_the_body(server: _Server) -> None:
    error = refused(lambda: fetch("http://public.test/slow", make_backend(server), EgressPolicy(timeout=0.5)))
    assert error.code == "timeout"


def test_the_policy_is_checked_before_the_backend(server: _Server) -> None:
    backend = make_backend(server)
    https_only = EgressPolicy(schemes=frozenset({"https"}))
    assert refused(lambda: fetch("http://public.test/", backend, https_only)).code == "scheme_refused"
    listed = EgressPolicy(allow_hosts=frozenset({"other.test"}))
    assert refused(lambda: fetch("http://public.test/", backend, listed)).reason == "not_allowed"
    assert backend.opened == [] and server.requests == []


def test_environment_proxies_are_ignored(server: _Server, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")  # nothing listens there
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    backend = make_backend(server)
    assert fetch("http://public.test/ok", backend)[0] == 200
    assert backend.opened == ["93.184.216.34"]


def test_a_mock_transport_keeps_the_url_checks_and_the_limits() -> None:
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        if request.url.path == "/hop":
            return httpx.Response(302, headers={"Location": "http://10.0.0.1/"})
        return httpx.Response(200, content=b"y" * 200)

    async def go(url: str, policy: EgressPolicy) -> bytes:
        async with guarded_client(policy, mock_transport=httpx.MockTransport(handler)) as client:
            response, _redirects = await send_following_redirects(
                client, client.build_request("GET", url), policy
            )
            return await response.aread()

    assert asyncio.run(go("https://public.test/", EgressPolicy())) == b"y" * 200
    with pytest.raises(EgressError) as caught:
        asyncio.run(go("https://public.test/hop", EgressPolicy()))
    assert caught.value.reason == "private"
    with pytest.raises(EgressError) as caught:
        asyncio.run(go("https://public.test/", EgressPolicy(max_bytes=100)))
    assert caught.value.code == "response_too_large"
    with pytest.raises(EgressError):
        asyncio.run(go("https://127.0.0.1/", EgressPolicy()))
    assert sent == ["https://public.test/", "https://public.test/hop", "https://public.test/"]


def test_tls_uses_the_host_name_not_the_address(tmp_path: Any) -> None:
    """SNI and certificate verification use the URL's name while the socket uses the checked address."""
    cryptography = pytest.importorskip("cryptography")
    assert cryptography
    import ssl

    from tests.net.tls_support import self_signed

    cert, key = self_signed(tmp_path, "good.test")
    httpd = _Server(("127.0.0.1", 0), _Handler)
    httpd.requests = []
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(cert, key)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        client_context = ssl.create_default_context(cafile=str(cert))
        answers = {"good.test": [["93.184.216.34"]], "other.test": [["93.184.216.34"]]}

        async def go(url: str) -> int:
            backend = make_backend(httpd, answers)
            async with guarded_client(EgressPolicy(), backend=backend, verify=client_context) as client:
                response = await client.get(url)
                return response.status_code

        assert asyncio.run(go("https://good.test/ok")) == 200
        # Same address, other name: the certificate does not match (httpx 0.27 and 0.28 raise other types).
        with pytest.raises(Exception) as caught:
            asyncio.run(go("https://other.test/ok"))
        assert not isinstance(caught.value, EgressError)  # the guard let it through; TLS refused it
        assert "other.test" in repr(caught.value) or "certificate" in repr(caught.value).lower()
    finally:
        httpd.shutdown()
        httpd.server_close()
