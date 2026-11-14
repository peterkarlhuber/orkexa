"""The HTTP search adapters against handwritten answers, and the DDGS adapter against a fake library (NAL-A2).

The JSON fixtures in tests/fixtures/web are written from the providers' public API documentation; none was
recorded from a live service, and no test contacts one: the HTTP adapters run through the egress guard with an
``httpx.MockTransport`` in place of the network.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
from pathlib import Path
import sys
import types
from typing import Any, ClassVar

import pytest

httpx = pytest.importorskip("httpx")

from orkexa.net.guard import EgressPolicy  # noqa: E402
from orkexa.web import (  # noqa: E402
    DDGS_BACKENDS,
    BraveSearch,
    DDGSSearch,
    MojeekSearch,
    SearchAuthFailed,
    SearchBadRequest,
    SearchRateLimited,
    SearchRequest,
    SearchResponse,
    SearchUnavailable,
    SearXNGSearch,
    TavilySearch,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "web"
KEY = "test-key-0123456789abcdefXYZ"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class Recorder:
    """A mock transport that answers every request with one canned response and records the requests."""

    def __init__(self, status: int = 200, body: Any = None, headers: dict[str, str] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.status = status
        self.body = body
        self.headers = headers or {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.body, bytes):
            return httpx.Response(self.status, content=self.body, headers=self.headers)
        return httpx.Response(
            self.status, json=self.body if self.body is not None else {}, headers=self.headers
        )

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


def search(provider: Any, **request: Any) -> SearchResponse:
    result: SearchResponse = asyncio.run(provider.search(SearchRequest.model_validate(request)))
    return result


# ---- Brave -------------------------------------------------------------------------------------------------


def test_brave_web_search() -> None:
    recorder = Recorder(body=fixture("brave_web.json"))
    brave = BraveSearch(KEY, mock_transport=recorder.transport)
    response = search(
        brave, query="rust async runtime", max_results=5, region="de", language="DE", time_range="week"
    )
    request = recorder.requests[0]
    assert request.url.host == "api.search.brave.com" and request.url.path == "/res/v1/web/search"
    assert dict(request.url.params) == {
        "q": "rust async runtime",
        "count": "5",
        "offset": "0",
        "country": "DE",
        "search_lang": "de",
        "freshness": "pw",
    }
    assert request.headers["X-Subscription-Token"] == KEY
    assert response.provider == "brave" and response.billed_units == 1 and response.notes == []
    first, second = response.hits
    assert first.title == "Asynchronous Programming in Rust"
    assert first.snippet == "A book on async & await in Rust."
    assert (first.published_at, first.source, first.rank) == ("2026-03-03T10:00:00", "rust-lang.github.io", 1)
    assert (second.url, second.published_at, second.rank) == ("https://tokio.rs/", None, 2)


def test_brave_news_and_pages() -> None:
    recorder = Recorder(body=fixture("brave_news.json"))
    response = search(
        BraveSearch(KEY, mock_transport=recorder.transport), query="election", kind="news", page=3
    )
    assert recorder.requests[0].url.path == "/res/v1/news/search"
    assert recorder.requests[0].url.params["offset"] == "2"
    assert response.hits[0].rank == 17  # page 3 of 8 per page
    assert response.hits[0].source == "news.example.org"


# ---- Tavily ------------------------------------------------------------------------------------------------


def test_tavily_search() -> None:
    recorder = Recorder(body=fixture("tavily.json"))
    tavily = TavilySearch(KEY, mock_transport=recorder.transport)
    response = search(tavily, query="who maintains trafilatura", kind="news", time_range="month", region="de")
    request = recorder.requests[0]
    assert (request.method, request.url) == ("POST", httpx.URL("https://api.tavily.com/search"))
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert body["topic"] == "news" and body["time_range"] == "month" and body["max_results"] == 8
    assert body["include_raw_content"] is False and body["include_answer"] is False
    assert [hit.url for hit in response.hits] == [
        "https://github.com/adbar/trafilatura",
        "https://trafilatura.readthedocs.io/",
    ]
    assert response.hits[1].published_at == "2025-10-01"
    assert response.notes == ["tavily does not support region; it was ignored"]
    assert response.billed_units == 1


def test_tavily_has_no_second_page_and_sends_nothing_for_it() -> None:
    recorder = Recorder(body=fixture("tavily.json"))
    response = search(TavilySearch(KEY, mock_transport=recorder.transport), query="x", page=2)
    assert recorder.requests == [] and response.hits == [] and response.billed_units == 1
    assert "tavily does not support pages; the first page was returned" in response.notes


# ---- Mojeek ------------------------------------------------------------------------------------------------


def test_mojeek_search() -> None:
    recorder = Recorder(body=fixture("mojeek.json"))
    mojeek = MojeekSearch(KEY, mock_transport=recorder.transport)
    response = search(mojeek, query="independent search", max_results=2, page=2, language="en", region="gb")
    params = dict(recorder.requests[0].url.params)
    assert params == {
        "q": "independent search",
        "api_key": KEY,
        "fmt": "json",
        "t": "2",
        "s": "3",
        "rb": "GB",
        "lb": "EN",
    }
    assert [hit.rank for hit in response.hits] == [3, 4]
    assert response.hits[0].published_at == "2025-01-01T00:00:00+00:00"
    assert response.hits[1].snippet == "A list & comparison."


def test_mojeek_errors_never_show_the_key() -> None:
    recorder = Recorder(body=fixture("mojeek_error.json"))
    with pytest.raises(SearchAuthFailed) as caught:
        search(MojeekSearch(KEY, mock_transport=recorder.transport), query="x")
    assert KEY not in str(caught.value)
    with pytest.raises(SearchBadRequest):
        search(MojeekSearch(KEY, mock_transport=recorder.transport), query="x", kind="news")

    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot connect to {request.url}")

    with pytest.raises(SearchUnavailable) as unavailable:
        search(MojeekSearch(KEY, mock_transport=httpx.MockTransport(broken)), query="x")
    assert KEY not in str(unavailable.value) and unavailable.value.__cause__ is None


# ---- SearXNG -----------------------------------------------------------------------------------------------


def test_searxng_on_a_private_host() -> None:
    recorder = Recorder(body=fixture("searxng.json"))
    searx = SearXNGSearch("http://192.168.1.20:8080/", mock_transport=recorder.transport)
    assert searx.policy.allow_hosts == frozenset({"192.168.1.20"}) and searx.policy.allow_private
    response = search(
        searx, query="selfhosted search", max_results=2, language="de", region="at", time_range="year"
    )
    params = dict(recorder.requests[0].url.params)
    assert params == {
        "q": "selfhosted search",
        "format": "json",
        "pageno": "1",
        "categories": "general",
        "language": "de-AT",
        "time_range": "year",
    }
    assert recorder.requests[0].url.port == 8080
    assert [hit.source for hit in response.hits] == ["duckduckgo", "github"]
    assert response.hits[1].published_at == "2026-10-30T12:00:00"
    assert response.notes == ["searxng engines without an answer: google, qwant"]
    assert response.billed_units == 0


def test_searxng_region_without_language_is_noted() -> None:
    recorder = Recorder(body={"results": []})
    response = search(
        SearXNGSearch("https://searx.example", mock_transport=recorder.transport),
        query="x",
        region="de",
        kind="news",
    )
    assert "language" not in recorder.requests[0].url.params
    assert recorder.requests[0].url.params["categories"] == "news"
    assert response.notes == [
        "searxng needs a language to apply a region; region was ignored",
        "provider returned no results",
    ]


def test_searxng_stays_on_its_host() -> None:
    def redirect(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/"})

    with pytest.raises(SearchUnavailable):
        search(SearXNGSearch("http://searx.lan", mock_transport=httpx.MockTransport(redirect)), query="x")


# ---- shared HTTP error mapping -----------------------------------------------------------------------------

ADAPTERS: dict[str, Callable[[httpx.MockTransport], Any]] = {
    "brave": lambda transport: BraveSearch(KEY, mock_transport=transport),
    "tavily": lambda transport: TavilySearch(KEY, mock_transport=transport),
    "mojeek": lambda transport: MojeekSearch(KEY, mock_transport=transport),
    "searxng": lambda transport: SearXNGSearch("https://searx.example", mock_transport=transport),
}


@pytest.mark.parametrize("name", sorted(ADAPTERS))
@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, SearchAuthFailed),
        (403, SearchAuthFailed),
        (429, SearchRateLimited),
        (400, SearchBadRequest),
        (422, SearchBadRequest),
        (500, SearchUnavailable),
        (503, SearchUnavailable),
        (301, SearchUnavailable),
    ],
)
def test_http_errors_are_mapped(name: str, status: int, error: type[Exception]) -> None:
    recorder = Recorder(
        status=status, body={"error": "x"}, headers={"Retry-After": "7", "Location": "https://x.test/"}
    )
    with pytest.raises(error) as caught:
        search(ADAPTERS[name](recorder.transport), query="x")
    assert KEY not in str(caught.value)
    assert caught.value.provider == name  # type: ignore[attr-defined]
    if status == 429:
        assert caught.value.retry_after == 7.0  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_empty_and_broken_answers(name: str) -> None:
    empty = {"web": {"results": []}, "results": [], "response": {"status": "OK", "results": []}}
    response = search(ADAPTERS[name](Recorder(body=empty).transport), query="x")
    assert response.hits == [] and response.notes[-1] == "provider returned no results"
    with pytest.raises(SearchUnavailable):
        search(ADAPTERS[name](Recorder(body=b"<html>not json</html>").transport), query="x")
    with pytest.raises(SearchUnavailable):
        search(ADAPTERS[name](Recorder(body=b"[1, 2]").transport), query="x")


def test_tavily_quota_statuses() -> None:
    for status in (432, 433):
        with pytest.raises(SearchRateLimited):
            search(TavilySearch(KEY, mock_transport=Recorder(status=status).transport), query="x")


def test_api_adapters_reach_only_their_api_host() -> None:
    brave = BraveSearch(KEY)
    assert brave.policy.allow_hosts == frozenset({"api.search.brave.com"})
    assert brave.policy.schemes == frozenset({"https"})
    # A policy that does not allow the API host refuses before anything is sent.
    recorder = Recorder(body=fixture("brave_web.json"))
    closed = BraveSearch(
        KEY, policy=EgressPolicy(allow_hosts=frozenset({"other.example"})), mock_transport=recorder.transport
    )
    with pytest.raises(SearchUnavailable):
        search(closed, query="x")
    assert recorder.requests == []


def test_check_makes_one_small_request() -> None:
    recorder = Recorder(body=fixture("brave_web.json"))
    asyncio.run(BraveSearch(KEY, mock_transport=recorder.transport).check())
    assert len(recorder.requests) == 1 and recorder.requests[0].url.params["count"] == "1"


# ---- DDGS --------------------------------------------------------------------------------------------------


class FakeDDGS:
    """Stands in for ``ddgs.DDGS``: records calls; ``books`` and ``extract`` fail the test if called."""

    calls: ClassVar[list[tuple[str, str, dict[str, Any]]]] = []
    rows: ClassVar[list[dict[str, Any]]] = []
    error: ClassVar[Exception | None] = None

    def __init__(self, **kwargs: Any) -> None:
        self.options = kwargs

    def _answer(self, method: str, query: str, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        FakeDDGS.calls.append((method, query, kwargs))
        if FakeDDGS.error is not None:
            raise FakeDDGS.error
        return FakeDDGS.rows

    def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self._answer("text", query, kwargs)

    def news(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self._answer("news", query, kwargs)

    def books(self, query: str, **kwargs: Any) -> Any:
        raise AssertionError("books must never be used")

    def extract(self, url: str, **kwargs: Any) -> Any:
        raise AssertionError("extract must never be used")


class DDGSException(Exception):
    pass


class RatelimitException(DDGSException):
    pass


class TimeoutException(DDGSException):
    pass


@pytest.fixture
def fake_ddgs(monkeypatch: pytest.MonkeyPatch) -> type[FakeDDGS]:
    package = types.ModuleType("ddgs")
    package.DDGS = FakeDDGS  # type: ignore[attr-defined]
    exceptions = types.ModuleType("ddgs.exceptions")
    exceptions.DDGSException = DDGSException  # type: ignore[attr-defined]
    exceptions.RatelimitException = RatelimitException  # type: ignore[attr-defined]
    exceptions.TimeoutException = TimeoutException  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ddgs", package)
    monkeypatch.setitem(sys.modules, "ddgs.exceptions", exceptions)
    FakeDDGS.calls, FakeDDGS.rows, FakeDDGS.error = [], [], None
    return FakeDDGS


def test_ddgs_text_uses_the_fixed_backends(fake_ddgs: type[FakeDDGS]) -> None:
    fake_ddgs.rows = [
        {"title": "One", "href": "https://one.example/", "body": "First <b>hit</b>"},
        {"title": "No link", "body": "dropped"},
        {"title": "Two", "href": "https://two.example/", "body": "Second"},
    ]
    response = search(
        DDGSSearch(), query="q", region="de", language="de", time_range="day", max_results=5, page=2
    )
    method, query, kwargs = fake_ddgs.calls[0]
    assert (method, query) == ("text", "q")
    assert kwargs == {
        "backend": DDGS_BACKENDS,
        "region": "de-de",
        "safesearch": "moderate",
        "timelimit": "d",
        "max_results": 5,
        "page": 2,
    }
    assert DDGS_BACKENDS == "duckduckgo,brave,mojeek,wikipedia"
    assert [(hit.title, hit.rank, hit.source) for hit in response.hits] == [
        ("One", 6, "one.example"),
        ("Two", 7, "two.example"),
    ]
    assert response.hits[0].snippet == "First hit"
    assert response.billed_units == 0 and response.notes == []


def test_ddgs_news(fake_ddgs: type[FakeDDGS]) -> None:
    fake_ddgs.rows = [
        {
            "title": "N",
            "url": "https://n.example/",
            "body": "b",
            "date": "2026-11-13T07:00:00+00:00",
            "source": "Example News",
        }
    ]
    response = search(DDGSSearch(), query="q", kind="news", region="de")
    method, _query, kwargs = fake_ddgs.calls[0]
    assert method == "news" and kwargs["backend"] == "duckduckgo" and kwargs["region"] == "wt-wt"
    assert (
        response.hits[0].source == "Example News"
        and response.hits[0].published_at == "2026-11-13T07:00:00+00:00"
    )
    assert response.notes == ["ddgs needs both region and language; both were ignored"]


def test_ddgs_empty_answers_carry_the_blocking_note(fake_ddgs: type[FakeDDGS]) -> None:
    note = "provider returned no results; it may be blocking requests"
    assert search(DDGSSearch(), query="q").notes == [note]
    fake_ddgs.error = DDGSException("No results found.")
    assert search(DDGSSearch(), query="q").notes == [note]


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (RatelimitException("202 Ratelimit"), SearchRateLimited),
        (TimeoutException("timed out"), SearchUnavailable),
        (OSError("x"), SearchUnavailable),
    ],
)
def test_ddgs_errors(fake_ddgs: type[FakeDDGS], raised: Exception, expected: type[Exception]) -> None:
    fake_ddgs.error = raised
    with pytest.raises(expected):
        search(DDGSSearch(), query="q")


def test_ddgs_runs_in_a_worker_thread(fake_ddgs: type[FakeDDGS], monkeypatch: pytest.MonkeyPatch) -> None:
    used: list[Any] = []
    real = asyncio.to_thread

    async def recording(function: Any, *args: Any, **kwargs: Any) -> Any:
        used.append(function)
        return await real(function, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", recording)
    search(DDGSSearch(), query="q")
    assert len(used) == 1


def test_ddgs_never_uses_books_or_extract() -> None:
    source = (Path(__file__).resolve().parents[2] / "src" / "orkexa" / "web" / "providers.py").read_text(
        encoding="utf-8"
    )
    code = source.split("class DDGSSearch", 1)[1]
    assert ".books(" not in code and ".extract(" not in code


def test_ddgs_without_the_library(monkeypatch: pytest.MonkeyPatch) -> None:
    from orkexa.errors import MissingDependencyError

    monkeypatch.setitem(sys.modules, "ddgs", None)
    with pytest.raises(MissingDependencyError):
        asyncio.run(DDGSSearch().check())
