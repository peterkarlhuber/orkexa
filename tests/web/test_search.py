"""Search models, the provider protocol and the deterministic test provider (NAL-A2)."""

from __future__ import annotations

import asyncio

from pydantic import ValidationError
import pytest

from orkexa.web import (
    BraveSearch,
    DDGSSearch,
    MojeekSearch,
    SearchCapabilities,
    SearchError,
    SearchHit,
    SearchProvider,
    SearchRateLimited,
    SearchRequest,
    SearchResponse,
    SearchUnavailable,
    SearXNGSearch,
    TavilySearch,
    TestSearchProvider,
)
from orkexa.web.search import unsupported_notes


def run(provider: SearchProvider, **request: object) -> SearchResponse:
    return asyncio.run(provider.search(SearchRequest.model_validate(request)))


def test_requests_are_validated() -> None:
    assert SearchRequest(query="x") == SearchRequest(query="x", kind="web", max_results=8, page=1)
    for bad in (
        {"query": ""},
        {"query": "x", "kind": "images"},
        {"query": "x", "max_results": 0},
        {"query": "x", "page": 0},
        {"query": "x", "region": "deu"},
        {"query": "x", "time_range": "hour"},
    ):
        with pytest.raises(ValidationError):
            SearchRequest.model_validate(bad)


def test_every_adapter_satisfies_the_protocol() -> None:
    providers = [
        TestSearchProvider(),
        BraveSearch("key"),
        TavilySearch("key"),
        MojeekSearch("key"),
        SearXNGSearch("http://searx.lan:8080"),
        DDGSSearch(),
    ]
    for provider in providers:
        assert isinstance(provider, SearchProvider)
        assert isinstance(provider.capabilities, SearchCapabilities)
    assert len({provider.name for provider in providers}) == len(providers)


def test_the_error_family() -> None:
    error = SearchRateLimited("slow down", provider="p", status=429, retry_after=2.0)
    assert isinstance(error, SearchError) and isinstance(error, RuntimeError)
    assert (error.provider, error.status, error.retry_after) == ("p", 429, 2.0)
    assert issubclass(SearchUnavailable, SearchError)


def test_the_test_provider_is_deterministic_and_offline() -> None:
    provider = TestSearchProvider()
    first = run(provider, query="Orkexa  agents", max_results=3)
    again = run(TestSearchProvider(), query="orkexa agents", max_results=3)
    assert first == again
    assert [hit.rank for hit in first.hits] == [1, 2, 3]
    assert first.provider == "test" and first.billed_units == 0 and first.notes == []
    assert all(hit.url.startswith("https://example.test/") for hit in first.hits)
    second_page = run(provider, query="orkexa agents", max_results=3, page=2)
    assert [hit.rank for hit in second_page.hits] == [4, 5, 6]
    assert {hit.url for hit in second_page.hits}.isdisjoint({hit.url for hit in first.hits})
    news = run(provider, query="orkexa agents", kind="news", max_results=1)
    assert news.hits[0].published_at is not None and news.hits[0].url != first.hits[0].url
    assert len(provider.requests) == 3


def test_the_test_provider_serves_fixtures() -> None:
    provider = TestSearchProvider(
        {
            "climate report": [
                {"title": "Report", "url": "https://example.test/report", "snippet": "The report."},
                SearchHit(title="Summary", url="https://example.test/summary", rank=99),
            ],
            "news:climate report": [{"title": "Breaking", "url": "https://example.test/news"}],
        },
        name="offline",
        generate=False,
    )
    response = run(provider, query="Climate Report")
    assert [(hit.title, hit.rank) for hit in response.hits] == [("Report", 1), ("Summary", 2)]
    assert response.provider == "offline"
    assert run(provider, query="climate report", kind="news").hits[0].title == "Breaking"
    empty = run(provider, query="unknown")
    assert empty.hits == [] and empty.notes == ["provider returned no results"]


def test_unsupported_parameters_become_notes() -> None:
    capabilities = SearchCapabilities(
        kinds=frozenset({"web"}),
        region=False,
        language=False,
        time_range=False,
        pagination=False,
        content_included=False,
        max_results_cap=5,
    )
    request = SearchRequest(query="x", region="de", language="de", time_range="week", page=2, max_results=10)
    notes = unsupported_notes(request, capabilities, "p")
    assert len(notes) == 5 and all(note.startswith("p ") for note in notes)
    assert unsupported_notes(SearchRequest(query="x", max_results=5), capabilities, "p") == []
    capped = run(TestSearchProvider(), query="x", max_results=50)
    assert len(capped.hits) == 20 and capped.notes == ["test returns at most 20 results per request"]


def test_check_is_free_for_the_offline_providers() -> None:
    asyncio.run(TestSearchProvider().check())


def test_provider_objects_need_their_configuration() -> None:
    with pytest.raises(ValueError):
        BraveSearch("  ")
    with pytest.raises(ValueError):
        DDGSSearch(safesearch="strict")
    with pytest.raises(Exception) as caught:
        SearXNGSearch("http://169.254.169.254/")
    assert getattr(caught.value, "reason", None) == "metadata"
