"""No test under tests/net or tests/web may reach the network: only loopback test servers are allowed."""

from __future__ import annotations

from collections.abc import Iterator
import socket
from typing import Any

import pytest

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """Refuse every name lookup and connection that is not to loopback; a test that tried one fails."""
    attempts: list[Any] = []
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host not in _LOOPBACK and host is not None:
            attempts.append(("getaddrinfo", host))
            raise OSError(f"test network block: lookup of {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    def check(address: Any) -> None:
        if isinstance(address, tuple) and address and address[0] not in _LOOPBACK:
            attempts.append(("connect", address))
            raise OSError(f"test network block: connect to {address!r}")

    def connect(self: socket.socket, address: Any) -> Any:
        check(address)
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> Any:
        check(address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    yield attempts
    assert attempts == [], f"the test tried to reach the network: {attempts}"
