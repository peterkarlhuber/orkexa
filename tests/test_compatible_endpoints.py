"""OpenAI-compatible endpoints never follow a redirect (ORK-003).

A real HTTP server on loopback answers with ``307``; a second one records whether the
request (the conversation) reached it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
import http.server
import json
import threading
from typing import Any

import httpx
import pytest

from orkexa import Agent, AsyncAgent
from orkexa.errors import ProviderError

ANSWER = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "moved"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class _Server:
    def __init__(self, handler: type[http.server.BaseHTTPRequestHandler]) -> None:
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def servers() -> Iterator[tuple[_Server, list[str]]]:
    received: list[str] = []

    class Target(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            received.append(self.rfile.read(length).decode("utf-8"))
            body = json.dumps(ANSWER).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    target = _Server(Target)

    class Redirect(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            self.send_response(307)
            self.send_header("Location", f"{target.url}{self.path}")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    redirect = _Server(Redirect)
    try:
        yield redirect, received
    finally:
        redirect.close()
        target.close()


@pytest.mark.parametrize("provider", ["compatible", "ollama", "vllm"])
def test_redirect_is_an_error_and_never_followed(servers: tuple[_Server, list[str]], provider: str) -> None:
    redirect, received = servers
    agent = Agent(model="m", provider=provider, base_url=f"{redirect.url}/v1", max_retries=0)
    with pytest.raises(ProviderError) as caught:
        agent.ask("the confidential question")
    assert caught.value.status_code == 307
    assert received == []


def test_async_redirect_is_an_error_and_never_followed(servers: tuple[_Server, list[str]]) -> None:
    redirect, received = servers
    agent = AsyncAgent(model="m", provider="compatible", base_url=f"{redirect.url}/v1", max_retries=0)
    with pytest.raises(ProviderError):
        asyncio.run(agent.ask("the confidential question"))
    assert received == []


def test_own_client_may_follow_redirects(servers: tuple[_Server, list[str]]) -> None:
    import openai

    redirect, received = servers
    client = openai.OpenAI(
        api_key="none",
        base_url=f"{redirect.url}/v1",
        max_retries=0,
        http_client=httpx.Client(follow_redirects=True),
    )
    agent = Agent(model="m", provider="compatible", client=client)
    assert agent.ask("hello") == "moved"
    assert len(received) == 1


def _closed_port() -> int:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.mark.parametrize(
    ("provider", "hint"),
    [
        ("compatible", "check that the server runs and that base_url is right"),
        ("ollama", "start Ollama (`ollama serve`"),
        ("vllm", "start the server (`vllm serve <model>`)"),
    ],
)
def test_connection_error_names_the_address_and_what_to_do(provider: str, hint: str) -> None:
    from orkexa.errors import ProviderConnectionError

    port = _closed_port()
    agent = Agent(
        model="m", provider=provider, base_url=f"http://user:secret@127.0.0.1:{port}/v1", max_retries=0
    )
    with pytest.raises(ProviderConnectionError) as caught:
        agent.ask("hello")
    message = str(caught.value)
    assert f"127.0.0.1:{port}/v1 cannot be reached" in message
    assert hint in message
    assert "secret" not in message
    assert caught.value.retryable


def test_redirect_error_says_what_to_do(servers: tuple[_Server, list[str]]) -> None:
    redirect, _ = servers
    agent = Agent(model="m", provider="compatible", base_url=f"{redirect.url}/v1", max_retries=0)
    with pytest.raises(ProviderError, match="set base_url to the address it redirects to"):
        agent.ask("hello")


def test_timeout_error_suggests_a_longer_timeout() -> None:
    import openai

    from orkexa.providers.compatible import OllamaProvider

    provider = OllamaProvider()
    request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    error = provider.map_error(openai.APITimeoutError(request=request))
    assert error is not None
    assert "did not answer in time" in str(error)
    assert "timeout=" in str(error)
