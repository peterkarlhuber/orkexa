"""Provider-independent web search: models, errors, the provider protocol and the test provider (ADR 0017).

Only pydantic is needed here. Adapters for real services are in :mod:`orkexa.web.providers`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import OrkexaError

__all__ = [
    "MAX_HIT_URL_CHARS",
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
    "TestSearchProvider",
    "TimeRange",
    "unsupported_notes",
    "usable_hit_url",
]

SearchKind = Literal["web", "news"]
TimeRange = Literal["day", "week", "month", "year"]


class SearchCapabilities(BaseModel):
    """What a provider supports. Parameters it does not support are ignored with a note, never an error."""

    model_config = ConfigDict(frozen=True)

    kinds: frozenset[SearchKind]
    region: bool
    language: bool
    time_range: bool
    pagination: bool
    #: Hits carry page content beyond a snippet (no fetch needed for a first answer).
    content_included: bool
    #: The most hits one request can return; larger ``max_results`` are reduced with a note.
    max_results_cap: int = Field(ge=1)


class SearchRequest(BaseModel):
    """One search. ``region`` is an ISO 3166-1 alpha-2 country code, ``language`` an ISO 639-1 code."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1, max_length=2000)
    kind: SearchKind = "web"
    max_results: int = Field(default=8, ge=1, le=100)
    region: str | None = Field(default=None, pattern=r"^[A-Za-z]{2}$")
    language: str | None = Field(default=None, pattern=r"^[A-Za-z]{2}$")
    time_range: TimeRange | None = None
    page: int = Field(default=1, ge=1, le=100)


class SearchHit(BaseModel):
    """One result. ``rank`` counts from 1 across pages; ``published_at`` is the provider's own text."""

    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None
    #: The site or engine the hit comes from, as the provider names it.
    source: str | None = None
    rank: int = Field(ge=1)


#: The longest hit URL a response keeps.
MAX_HIT_URL_CHARS = 2048


def usable_hit_url(url: str) -> bool:
    """Whether a hit's URL is kept: ``http``/``https``, at most :data:`MAX_HIT_URL_CHARS` characters, no
    whitespace or control characters (a scraped ``href`` could otherwise carry text into a prompt)."""
    if not 0 < len(url) <= MAX_HIT_URL_CHARS or not url.lower().startswith(("http://", "https://")):
        return False
    return not any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url)


class SearchResponse(BaseModel):
    """The hits of one search, the provider's name, notes (ignored parameters, empty answers) and the billed
    units of the request (provider-specific: requests or credits; ``0`` for free providers).

    Hits whose URL fails :func:`usable_hit_url` are dropped, with a note.
    """

    hits: list[SearchHit]
    provider: str
    notes: list[str] = Field(default_factory=list)
    billed_units: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _drop_unusable_hits(self) -> SearchResponse:
        kept = [hit for hit in self.hits if usable_hit_url(hit.url)]
        if len(kept) != len(self.hits):
            self.notes.append(f"{len(self.hits) - len(kept)} hit(s) with an unusable address were left out")
            self.hits = kept
        return self


class SearchError(OrkexaError, RuntimeError):
    """A search failed. Messages never contain API keys."""

    def __init__(self, message: str, *, provider: str, status: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status


class SearchUnavailable(SearchError):
    """The provider cannot be reached, answered with a server error or an unreadable answer."""


class SearchRateLimited(SearchError):
    """The provider refuses because of a rate limit or a quota (HTTP 429, or a detected block)."""

    def __init__(
        self, message: str, *, provider: str, status: int | None = None, retry_after: float | None = None
    ) -> None:
        super().__init__(message, provider=provider, status=status)
        self.retry_after = retry_after


class SearchAuthFailed(SearchError):
    """The provider refuses the credentials (HTTP 401 or 403)."""


class SearchBadRequest(SearchError):
    """The provider refuses the request itself (HTTP 400, 404, 422)."""


@runtime_checkable
class SearchProvider(Protocol):
    """A web search service. Switching providers changes only the object, never the caller."""

    name: str
    capabilities: SearchCapabilities

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Run one search; raises a :class:`SearchError` subclass."""
        ...

    async def check(self) -> None:
        """Check that the provider is usable (configuration and, for remote services, one small request)."""
        ...


def unsupported_notes(request: SearchRequest, capabilities: SearchCapabilities, provider: str) -> list[str]:
    """Notes for the parameters of ``request`` that ``capabilities`` cannot honor."""
    notes = []
    if request.region is not None and not capabilities.region:
        notes.append(f"{provider} does not support region; it was ignored")
    if request.language is not None and not capabilities.language:
        notes.append(f"{provider} does not support language; it was ignored")
    if request.time_range is not None and not capabilities.time_range:
        notes.append(f"{provider} does not support time_range; it was ignored")
    if request.page > 1 and not capabilities.pagination:
        notes.append(f"{provider} does not support pages; the first page was returned")
    if request.max_results > capabilities.max_results_cap:
        notes.append(f"{provider} returns at most {capabilities.max_results_cap} results per request")
    return notes


class TestSearchProvider:
    """A deterministic provider for tests and offline runs: no network, no cost.

    ``fixtures`` maps a query (compared case-insensitively, whitespace collapsed) to its hits in rank order;
    a ``kind`` of ``news`` uses the key ``"news:<query>"`` first. Queries without a fixture get generated hits
    under ``https://example.test/`` that depend only on the query, so runs are reproducible. Every request is
    recorded in :attr:`requests`.
    """

    __test__ = False  # not a pytest test class

    capabilities = SearchCapabilities(
        kinds=frozenset({"web", "news"}),
        region=True,
        language=True,
        time_range=True,
        pagination=True,
        content_included=False,
        max_results_cap=20,
    )

    def __init__(
        self,
        fixtures: Mapping[str, Sequence[SearchHit | Mapping[str, object]]] | None = None,
        *,
        name: str = "test",
        generate: bool = True,
    ) -> None:
        self.name = name
        self.generate = generate
        self.requests: list[SearchRequest] = []
        self._fixtures = {
            self._key(query): [self._hit(hit, rank) for rank, hit in enumerate(hits, start=1)]
            for query, hits in (fixtures or {}).items()
        }

    @staticmethod
    def _key(query: str) -> str:
        return " ".join(query.lower().split())

    @staticmethod
    def _hit(hit: SearchHit | Mapping[str, object], rank: int) -> SearchHit:
        if isinstance(hit, SearchHit):
            return hit.model_copy(update={"rank": rank})
        return SearchHit.model_validate({**hit, "rank": rank})

    def _generated(self, request: SearchRequest, count: int) -> list[SearchHit]:
        query = self._key(request.query)
        digest = hashlib.sha256(f"{request.kind}:{query}".encode()).hexdigest()[:12]
        return [
            SearchHit(
                title=f"Result {number} for {query}",
                url=f"https://example.test/{digest}/{number}",
                snippet=f"Deterministic test result {number} for {query!r}.",
                source="example.test",
                rank=number,
                published_at="2026-01-01T00:00:00Z" if request.kind == "news" else None,
            )
            for number in range(1, count + 1)
        ]

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.requests.append(request)
        cap = self.capabilities.max_results_cap
        per_page = min(request.max_results, cap)
        start = (request.page - 1) * per_page
        key = self._key(request.query)
        hits = self._fixtures.get(f"news:{key}" if request.kind == "news" else key)
        if hits is None and request.kind == "news":
            hits = self._fixtures.get(key)
        if hits is None:
            hits = self._generated(request, start + per_page) if self.generate else []
        page = [
            hit.model_copy(update={"rank": start + index})
            for index, hit in enumerate(hits[start : start + per_page], 1)
        ]
        notes = unsupported_notes(request, self.capabilities, self.name)
        if not page:
            notes.append("provider returned no results")
        return SearchResponse(hits=page, provider=self.name, notes=notes, billed_units=0)

    async def check(self) -> None:
        return None
