"""Credentials never surface through exceptions, tracebacks, logs or representations.

Checks the exception objects an application actually receives: their messages, the
fully formatted traceback (which follows ``__cause__`` and ``__context__``) and what
``logging`` prints for them, for synchronous, asynchronous and streamed turns, token
counting and batches, and for the real OpenAI SDK behind a mocked HTTP transport.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
import logging
import traceback
from typing import Any

import httpx2
import openai
import pytest

from orkexa import (
    Agent,
    AsyncAgent,
    AuthenticationError,
    ProviderError,
    ProviderServerError,
    register_provider,
)
from orkexa.dynamic import ProviderSettings
from orkexa.providers import ProviderConfig, redact
from orkexa.providers.fake import FakeModel, FakeProvider

KEY = "sk-proj-EXAMPLEPRIVATEKEY1234567890"

# Secrets as they appear in SDK and HTTP error texts.
LEAKS = {
    "configured key": f"invalid key {KEY}",
    "bearer header": "headers={'Authorization': 'Bearer tok_live_ABCDEFGHIJKLMNOP'}",
    "basic header": "Authorization: Basic dXNlcjpwYXNzd29yZDEyMw==",
    "x-api-key header": "x-api-key: zz-EXAMPLE-0123456789",
    "google key": "key AIzaSyEXAMPLE0123456789abcdefghij",
    "url password": "POST https://svc:hunter2-EXAMPLE@gateway.invalid/v1/chat",
    "url query key": "GET https://gateway.invalid/v1?api_key=qk_EXAMPLE_PRIVATE_42&region=eu",
    "json field": '{"client_secret": "cs_EXAMPLE_PRIVATE_42"}',
}
SECRETS = (
    KEY,
    "tok_live_ABCDEFGHIJKLMNOP",
    "dXNlcjpwYXNzd29yZDEyMw==",
    "zz-EXAMPLE-0123456789",
    "AIzaSyEXAMPLE0123456789abcdefghij",
    "hunter2-EXAMPLE",
    "qk_EXAMPLE_PRIVATE_42",
    "cs_EXAMPLE_PRIVATE_42",
)


class LeakyError(Exception):
    """Stands in for an SDK exception whose text and attributes carry credentials."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.headers = {"Authorization": f"Bearer {KEY}"}


class MappingProvider(FakeProvider):
    """A provider whose map_error turns every SDK exception into a ProviderError."""

    name = "mapping_fake"

    def map_error(self, exc: BaseException) -> BaseException | None:
        return self.provider_error(exc, status_code=503, request_id="req_123")


@pytest.fixture(autouse=True)
def _provider() -> Iterator[None]:
    register_provider("mapping_fake", MappingProvider, replace=True)
    yield


def _sdk_failure(text: str, chain: str) -> BaseException:
    """An SDK exception with the secret in its text, its explicit cause or its context."""
    try:
        try:
            raise LeakyError(f"low-level failure: {text}")
        except LeakyError as inner:
            if chain == "explicit":
                raise LeakyError("request failed") from inner
            if chain == "implicit":
                raise LeakyError("request failed")  # noqa: B904 - the implicit context is the point
            raise LeakyError(f"request failed: {text}") from None
    except LeakyError as outer:
        return outer


def _surfaced(exc: BaseException) -> str:
    """Everything an application or a log handler can print for an exception."""
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    record = logging.LogRecord("app", logging.ERROR, __file__, 1, "turn failed", None, None)
    record.exc_info = (type(exc), exc, exc.__traceback__)
    return "\n".join([str(exc), repr(exc), formatted, logging.Formatter().format(record)])


def _assert_clean(exc: BaseException) -> None:
    assert isinstance(exc, ProviderError)
    assert exc.__cause__ is None and exc.__context__ is None
    text = _surfaced(exc)
    for secret in SECRETS:
        assert secret not in text
    # The safe diagnostics stay available.
    assert (exc.provider, exc.status_code, exc.request_id, exc.retryable) == (
        "mapping_fake",
        503,
        "req_123",
        True,
    )
    assert isinstance(exc, ProviderServerError) and "LeakyError" in str(exc)


def _agent(failure: BaseException, **kwargs: Any) -> Agent[str]:
    return Agent(model="mapping_fake:m", client=FakeModel([failure]), api_key=KEY, **kwargs)


@pytest.mark.parametrize("chain", ["explicit", "implicit", "none"])
@pytest.mark.parametrize("leak", sorted(LEAKS))
@pytest.mark.parametrize("stream", [False, True])
def test_sync_turns_surface_no_secret(leak: str, chain: str, stream: bool) -> None:
    agent = _agent(_sdk_failure(LEAKS[leak], chain), stream=stream)
    with pytest.raises(ProviderError) as caught:
        agent.ask("hi")
    _assert_clean(caught.value)


@pytest.mark.parametrize("chain", ["explicit", "implicit"])
@pytest.mark.parametrize("stream", [False, True])
def test_async_turns_surface_no_secret(chain: str, stream: bool) -> None:
    agent = AsyncAgent(
        model="mapping_fake:m",
        client=FakeModel([_sdk_failure(LEAKS["bearer header"], chain)]),
        api_key=KEY,
        stream=stream,
    )
    with pytest.raises(ProviderError) as caught:
        asyncio.run(agent.ask("hi"))
    _assert_clean(caught.value)


def test_token_counting_and_batches_surface_no_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise _sdk_failure(LEAKS["configured key"], "explicit")

    agent = _agent(RuntimeError("unused"))
    monkeypatch.setattr(agent.adapter, "count_tokens", fail)
    monkeypatch.setattr(agent.adapter, "run_batch", fail)
    with pytest.raises(ProviderError) as caught:
        agent.count_tokens("hi")
    _assert_clean(caught.value)
    with pytest.raises(ProviderError) as caught:
        agent._run_batch({"case": "hi"}, poll_interval=0, timeout=None)
    _assert_clean(caught.value)

    async_agent = AsyncAgent(model="mapping_fake:m", client=FakeModel([]), api_key=KEY)

    async def afail(*args: Any, **kwargs: Any) -> Any:
        fail()

    monkeypatch.setattr(async_agent.adapter, "acount_tokens", afail)
    with pytest.raises(ProviderError) as caught:
        asyncio.run(async_agent.count_tokens("hi"))
    _assert_clean(caught.value)


def test_logging_exception_in_an_application_handler(caplog: pytest.LogCaptureFixture) -> None:
    agent = _agent(_sdk_failure(LEAKS["url query key"], "explicit"))
    try:
        agent.ask("hi")
    except ProviderError:
        logging.getLogger("app").exception("model call failed")
    assert "model call failed" in caplog.text and "Traceback" in caplog.text
    for secret in SECRETS:
        assert secret not in caplog.text


def test_real_openai_sdk_error_keeps_no_link_to_the_request() -> None:
    """The SDK exception holds the request with its Authorization header; it must stay unreachable."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["authorization"] == f"Bearer {KEY}"
        body = {"error": {"message": f"Incorrect API key provided: {KEY}", "type": "invalid_request_error"}}
        return httpx2.Response(401, json=body, headers={"x-request-id": "req_abc"})

    client = openai.OpenAI(
        api_key=KEY, max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    agent = Agent(model="openai:gpt-6.1-sol", client=client)
    with pytest.raises(AuthenticationError) as caught:
        agent.ask("hi")
    exc = caught.value
    assert exc.__cause__ is None and exc.__context__ is None
    assert (exc.provider, exc.status_code, exc.retryable) == ("openai", 401, False)
    assert KEY not in _surfaced(exc)


# ---- configuration representations ----------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "secrets", "kept"),
    [
        (
            "https://host.invalid/api?api_key=querytoken_PRIVATE_1234",
            ["querytoken_PRIVATE_1234"],
            "host.invalid/api",
        ),
        (
            "https://gw.invalid/v1?access_token=at_PRIVATE_1&model=x&client_secret=cs_PRIVATE_2",
            ["at_PRIVATE_1", "cs_PRIVATE_2"],
            "model=x",
        ),
        ("https://svc:p%40ss-PRIVATE@gw.invalid/v1", ["p%40ss-PRIVATE", "svc:"], "gw.invalid/v1"),
        ("https://tokPRIVATE99@gw.invalid/v1", ["tokPRIVATE99"], "@gw.invalid/v1"),
        ("https://gw.invalid/v1?X-Amz-Signature=abc%2FPRIVATE&region=eu", ["abc%2FPRIVATE"], "region=eu"),
        ("http://localhost:11434/v1?stream=true", [], "http://localhost:11434/v1?stream=true"),
    ],
)
def test_configuration_representations_hide_url_credentials(url: str, secrets: list[str], kept: str) -> None:
    for text in (
        repr(ProviderConfig(api_key=KEY, base_url=url)),
        repr(ProviderSettings(api_key=KEY, base_url=url)),
    ):
        assert KEY not in text and kept in text
        for secret in secrets:
            assert secret not in text
    assert ProviderConfig(base_url=url).client_kwargs()["base_url"] == url  # runtime value unchanged


def test_redaction_keeps_ordinary_text() -> None:
    for text in (
        "max_tokens=1024",
        "tokens exceeded: 2048 > 1024",
        "https://h.invalid/v1?model=gpt&keyword=x",
    ):
        assert redact(text) == text
