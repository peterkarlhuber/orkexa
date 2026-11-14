"""Provider-independent web search and safe page fetching (ADR 0017).

* Models and the protocol: :class:`SearchRequest`, :class:`SearchResponse`, :class:`SearchHit`,
  :class:`SearchCapabilities`, :class:`SearchProvider` and the :class:`SearchError` family.
* Providers: :class:`TestSearchProvider` (deterministic, no network), :class:`BraveSearch`,
  :class:`TavilySearch`, :class:`MojeekSearch`, :class:`SearXNGSearch` (``pip install "orkexa[web]"``) and
  :class:`DDGSSearch` (``pip install "orkexa[web-ddgs]"``). None is a default: the caller chooses.
* :func:`fetch_page` fetches one page through the egress guard (:mod:`orkexa.net.guard`) and returns its text.

Importing this package needs only pydantic; the extras are imported when a provider or ``fetch_page`` is used.
"""

from __future__ import annotations

from .fetch import DEFAULT_ACCEPT, FetchedPage, FetchError, fetch_page
from .providers import DDGS_BACKENDS, BraveSearch, DDGSSearch, MojeekSearch, SearXNGSearch, TavilySearch
from .search import (
    SearchAuthFailed,
    SearchBadRequest,
    SearchCapabilities,
    SearchError,
    SearchHit,
    SearchKind,
    SearchProvider,
    SearchRateLimited,
    SearchRequest,
    SearchResponse,
    SearchUnavailable,
    TestSearchProvider,
    TimeRange,
)

__all__ = [
    "DDGS_BACKENDS",
    "DEFAULT_ACCEPT",
    "BraveSearch",
    "DDGSSearch",
    "FetchError",
    "FetchedPage",
    "MojeekSearch",
    "SearXNGSearch",
    "SearchAuthFailed",
    "SearchBadRequest",
    "SearchCapabilities",
    "SearchError",
    "SearchHit",
    "SearchKind",
    "SearchProvider",
    "SearchRateLimited",
    "SearchRequest",
    "SearchResponse",
    "SearchUnavailable",
    "TavilySearch",
    "TestSearchProvider",
    "TimeRange",
    "fetch_page",
]
