"""Search adapters for real services (ADR 0017).

Brave, Tavily, Mojeek and SearXNG are reached only through the egress guard (:mod:`orkexa.net.http`, needs
``pip install "orkexa[web]"``); each adapter's default policy allows only its own API host. DDGS (the ``ddgs``
package, ``pip install "orkexa[web-ddgs]"``) makes its own requests to a fixed list of public search engines.

Nothing here picks a provider: the caller constructs the one its configuration names. No adapter is a default.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
import html
import re
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlsplit

from ..net.guard import EgressError, EgressPolicy, Resolver
from .search import (
    SearchAuthFailed,
    SearchBadRequest,
    SearchCapabilities,
    SearchError,
    SearchHit,
    SearchRateLimited,
    SearchRequest,
    SearchResponse,
    SearchUnavailable,
    unsupported_notes,
)

if TYPE_CHECKING:
    import httpx

__all__ = ["DDGS_BACKENDS", "BraveSearch", "DDGSSearch", "MojeekSearch", "SearXNGSearch", "TavilySearch"]

_TAGS = re.compile(r"<[^>]{0,200}>")


def _plain(value: object) -> str:
    """Provider text without HTML tags and entities (snippets often carry ``<strong>``)."""
    if not isinstance(value, str):
        return ""
    return " ".join(html.unescape(_TAGS.sub("", value)).split())


def _text_or_none(value: object) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _host(url: str) -> str | None:
    try:
        return urlsplit(url).hostname
    except ValueError:
        return None


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after", "").strip()
    try:
        return float(value) if value else None
    except ValueError:
        return None


class _HttpSearch:
    """Shared request and error handling of the HTTP adapters."""

    name: ClassVar[str]
    capabilities: ClassVar[SearchCapabilities]
    #: Billed units of one request (the provider's request or credit count).
    billed_per_request: ClassVar[int] = 1

    def __init__(
        self,
        policy: EgressPolicy,
        *,
        mock_transport: httpx.AsyncBaseTransport | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        self.policy = policy
        self._mock_transport = mock_transport
        self._resolver = resolver

    async def _call(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
        json: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """One request through the guard; returns the decoded JSON answer or raises a :class:`SearchError`."""
        try:
            import httpx

            from ..net.http import deadline, guarded_client
        except ImportError:
            from ..errors import MissingDependencyError

            raise MissingDependencyError(
                f'The {self.name} search needs httpx: pip install "orkexa[web]"'
            ) from None

        provider = self.name
        try:
            async with guarded_client(
                self.policy,
                headers={"Accept": "application/json", **(headers or {})},
                resolver=self._resolver,
                mock_transport=self._mock_transport,
            ) as client:
                with deadline(self.policy):
                    response = await client.request(method, url, params=params, json=json)
        except EgressError as exc:
            raise SearchUnavailable(f"{provider} cannot be reached: {exc}", provider=provider) from exc
        except httpx.HTTPError as exc:
            # The exception text may contain the URL with an API key in its query: it is not repeated.
            raise SearchUnavailable(
                f"{provider} cannot be reached ({type(exc).__name__})", provider=provider
            ) from None
        status = response.status_code
        if status in (401, 403):
            raise SearchAuthFailed(
                f"{provider} refused the credentials (HTTP {status})", provider=provider, status=status
            )
        if status == 429 or status in self._quota_statuses():
            raise SearchRateLimited(
                f"{provider} is rate limited or out of quota (HTTP {status})",
                provider=provider,
                status=status,
                retry_after=_retry_after(response),
            )
        if status in (400, 404, 405, 422):
            raise SearchBadRequest(
                f"{provider} refused the request (HTTP {status})", provider=provider, status=status
            )
        if status >= 300:
            raise SearchUnavailable(f"{provider} answered HTTP {status}", provider=provider, status=status)
        try:
            return response.json()
        except ValueError:
            raise SearchUnavailable(
                f"{provider} sent an answer that is not JSON", provider=provider
            ) from None

    def _quota_statuses(self) -> frozenset[int]:
        return frozenset()

    def _response(self, request: SearchRequest, hits: list[SearchHit], notes: list[str]) -> SearchResponse:
        if not hits:
            notes = [*notes, "provider returned no results"]
        return SearchResponse(
            hits=hits, provider=self.name, notes=notes, billed_units=self.billed_per_request
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        raise NotImplementedError

    async def check(self) -> None:
        """One small search (counts as one billed request at paid providers)."""
        await self.search(SearchRequest(query="orkexa", max_results=1))


def _rank(request: SearchRequest, per_page: int, index: int) -> int:
    return (request.page - 1) * per_page + index


def _api_policy(host: str, policy: EgressPolicy | None) -> EgressPolicy:
    return policy or EgressPolicy(
        schemes=frozenset({"https"}), allow_hosts=frozenset({host}), ports=frozenset({443}), timeout=20.0
    )


def _require_key(api_key: str, name: str) -> str:
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(f"The {name} search needs an API key")
    return api_key.strip()


# ---- Brave -------------------------------------------------------------------------------------------------

_BRAVE_FRESHNESS = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}


class BraveSearch(_HttpSearch):
    """Brave Search API (``X-Subscription-Token``). Brave's terms forbid storing results beyond the task."""

    name = "brave"
    capabilities = SearchCapabilities(
        kinds=frozenset({"web", "news"}),
        region=True,
        language=True,
        time_range=True,
        pagination=True,
        content_included=False,
        max_results_cap=20,
    )
    base_url = "https://api.search.brave.com/res/v1"

    def __init__(
        self,
        api_key: str,
        *,
        policy: EgressPolicy | None = None,
        mock_transport: httpx.AsyncBaseTransport | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        super().__init__(
            _api_policy("api.search.brave.com", policy), mock_transport=mock_transport, resolver=resolver
        )
        self._key = _require_key(api_key, "Brave")

    async def search(self, request: SearchRequest) -> SearchResponse:
        notes = unsupported_notes(request, self.capabilities, self.name)
        count = min(request.max_results, self.capabilities.max_results_cap)
        params: dict[str, str | int] = {"q": request.query, "count": count, "offset": request.page - 1}
        if request.region:
            params["country"] = request.region.upper()
        if request.language:
            params["search_lang"] = request.language.lower()
        if request.time_range:
            params["freshness"] = _BRAVE_FRESHNESS[request.time_range]
        endpoint = "news" if request.kind == "news" else "web"
        data = await self._call(
            "GET",
            f"{self.base_url}/{endpoint}/search",
            params=params,
            headers={"X-Subscription-Token": self._key},
        )
        if not isinstance(data, dict):
            raise SearchUnavailable("brave sent an unexpected answer", provider=self.name)
        web = data.get("web")
        section = (
            data.get("results")
            if request.kind == "news"
            else web.get("results")
            if isinstance(web, dict)
            else None
        )
        hits: list[SearchHit] = []
        for item in section or []:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                continue
            meta = item.get("meta_url")
            hostname = meta.get("hostname") if isinstance(meta, dict) else None
            hits.append(
                SearchHit(
                    title=_plain(item.get("title")),
                    url=item["url"],
                    snippet=_plain(item.get("description")),
                    published_at=_text_or_none(item.get("page_age")) or _text_or_none(item.get("age")),
                    source=_text_or_none(hostname) or _host(item["url"]),
                    rank=_rank(request, count, len(hits) + 1),
                )
            )
        return self._response(request, hits[:count], notes)


# ---- Tavily ------------------------------------------------------------------------------------------------


class TavilySearch(_HttpSearch):
    """Tavily Search API (basic depth: one credit per request; no answer, no raw content)."""

    name = "tavily"
    capabilities = SearchCapabilities(
        kinds=frozenset({"web", "news"}),
        region=False,
        language=False,
        time_range=True,
        pagination=False,
        content_included=False,
        max_results_cap=20,
    )
    base_url = "https://api.tavily.com"

    def __init__(
        self,
        api_key: str,
        *,
        policy: EgressPolicy | None = None,
        mock_transport: httpx.AsyncBaseTransport | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        super().__init__(
            _api_policy("api.tavily.com", policy), mock_transport=mock_transport, resolver=resolver
        )
        self._key = _require_key(api_key, "Tavily")

    def _quota_statuses(self) -> frozenset[int]:
        return frozenset({432, 433})  # plan and pay-as-you-go limits

    async def search(self, request: SearchRequest) -> SearchResponse:
        notes = unsupported_notes(request, self.capabilities, self.name)
        if request.page > 1:
            return self._response(request, [], notes)
        count = min(request.max_results, self.capabilities.max_results_cap)
        body: dict[str, Any] = {
            "query": request.query,
            "topic": "news" if request.kind == "news" else "general",
            "search_depth": "basic",
            "max_results": count,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        if request.time_range:
            body["time_range"] = request.time_range
        data = await self._call(
            "POST", f"{self.base_url}/search", json=body, headers={"Authorization": f"Bearer {self._key}"}
        )
        if not isinstance(data, dict):
            raise SearchUnavailable("tavily sent an unexpected answer", provider=self.name)
        hits: list[SearchHit] = []
        for item in data.get("results") or []:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                continue
            hits.append(
                SearchHit(
                    title=_plain(item.get("title")),
                    url=item["url"],
                    snippet=_plain(item.get("content")),
                    published_at=_text_or_none(item.get("published_date")),
                    source=_host(item["url"]),
                    rank=len(hits) + 1,
                )
            )
        return self._response(request, hits[:count], notes)


# ---- Mojeek ------------------------------------------------------------------------------------------------


class MojeekSearch(_HttpSearch):
    """Mojeek Search API (own index, operated in Europe). The key travels in the query string, so it is
    never repeated in error messages."""

    name = "mojeek"
    capabilities = SearchCapabilities(
        kinds=frozenset({"web"}),
        region=True,
        language=True,
        time_range=False,
        pagination=True,
        content_included=False,
        max_results_cap=20,
    )
    base_url = "https://api.mojeek.com"

    def __init__(
        self,
        api_key: str,
        *,
        policy: EgressPolicy | None = None,
        mock_transport: httpx.AsyncBaseTransport | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        super().__init__(
            _api_policy("api.mojeek.com", policy), mock_transport=mock_transport, resolver=resolver
        )
        self._key = _require_key(api_key, "Mojeek")

    async def search(self, request: SearchRequest) -> SearchResponse:
        notes = unsupported_notes(request, self.capabilities, self.name)
        if request.kind == "news":
            raise SearchBadRequest("mojeek has no news search", provider=self.name)
        count = min(request.max_results, self.capabilities.max_results_cap)
        params: dict[str, str | int] = {
            "q": request.query,
            "api_key": self._key,
            "fmt": "json",
            "t": count,
            "s": (request.page - 1) * count + 1,
        }
        if request.region:
            params["rb"] = request.region.upper()
        if request.language:
            params["lb"] = request.language.upper()
        data = await self._call("GET", f"{self.base_url}/search", params=params)
        body = data.get("response") if isinstance(data, dict) else None
        if not isinstance(body, dict):
            raise SearchUnavailable("mojeek sent an unexpected answer", provider=self.name)
        status = str(body.get("status", "OK"))
        if status.upper() != "OK":
            text = status.lower()
            if "key" in text or "auth" in text:
                raise SearchAuthFailed("mojeek refused the credentials", provider=self.name)
            if "limit" in text or "quota" in text:
                raise SearchRateLimited("mojeek is rate limited or out of quota", provider=self.name)
            raise SearchBadRequest("mojeek refused the request", provider=self.name)
        hits: list[SearchHit] = []
        for item in body.get("results") or []:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                continue
            hits.append(
                SearchHit(
                    title=_plain(item.get("title")),
                    url=item["url"],
                    snippet=_plain(item.get("desc")),
                    published_at=_mojeek_date(item),
                    source=_host(item["url"]),
                    rank=_rank(request, count, len(hits) + 1),
                )
            )
        return self._response(request, hits[:count], notes)


def _mojeek_date(item: Mapping[str, Any]) -> str | None:
    value = item.get("date")
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return datetime.fromtimestamp(value, UTC).isoformat()
    return _text_or_none(value)


# ---- SearXNG -----------------------------------------------------------------------------------------------


class SearXNGSearch(_HttpSearch):
    """A SearXNG instance (``format=json`` must be enabled in its settings). Free; often self-hosted.

    The default policy allows exactly the instance's host and port and, because instances often run in the
    local network, private addresses (``allow_private``); metadata addresses stay refused. ``headers`` go
    with every request (for example a reverse proxy's authorization).
    """

    name = "searxng"
    capabilities = SearchCapabilities(
        kinds=frozenset({"web", "news"}),
        region=True,
        language=True,
        time_range=True,
        pagination=True,
        content_included=False,
        max_results_cap=50,
    )
    billed_per_request = 0

    def __init__(
        self,
        base_url: str,
        *,
        policy: EgressPolicy | None = None,
        headers: Mapping[str, str] | None = None,
        mock_transport: httpx.AsyncBaseTransport | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        origin = EgressPolicy(allow_private=True, ports=None).check_url(base_url)
        if policy is None:
            policy = EgressPolicy(
                schemes=frozenset({origin.scheme}),
                allow_private=True,
                allow_hosts=frozenset({origin.host}),
                ports=frozenset({origin.port}),
                timeout=20.0,
            )
        super().__init__(policy, mock_transport=mock_transport, resolver=resolver)
        self.base_url = base_url.rstrip("/")
        self._headers = dict(headers or {})

    async def search(self, request: SearchRequest) -> SearchResponse:
        notes = unsupported_notes(request, self.capabilities, self.name)
        count = min(request.max_results, self.capabilities.max_results_cap)
        params: dict[str, str | int] = {
            "q": request.query,
            "format": "json",
            "pageno": request.page,
            "categories": "news" if request.kind == "news" else "general",
        }
        if request.language and request.region:
            params["language"] = f"{request.language.lower()}-{request.region.upper()}"
        elif request.language:
            params["language"] = request.language.lower()
        elif request.region:
            notes.append("searxng needs a language to apply a region; region was ignored")
        if request.time_range:
            params["time_range"] = request.time_range
        data = await self._call("GET", f"{self.base_url}/search", params=params, headers=self._headers)
        if not isinstance(data, dict):
            raise SearchUnavailable("searxng sent an unexpected answer", provider=self.name)
        unresponsive = [
            str(entry[0])
            for entry in data.get("unresponsive_engines") or []
            if isinstance(entry, list) and entry
        ]
        if unresponsive:
            notes.append(f"searxng engines without an answer: {', '.join(sorted(set(unresponsive)))}")
        hits: list[SearchHit] = []
        for item in data.get("results") or []:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                continue
            hits.append(
                SearchHit(
                    title=_plain(item.get("title")),
                    url=item["url"],
                    snippet=_plain(item.get("content")),
                    published_at=_text_or_none(item.get("publishedDate")),
                    source=_text_or_none(item.get("engine")) or _host(item["url"]),
                    rank=_rank(request, count, len(hits) + 1),
                )
            )
            if len(hits) == count:
                break
        return self._response(request, hits, notes)


# ---- DDGS --------------------------------------------------------------------------------------------------

#: The fixed text backends of :class:`DDGSSearch`; news uses DDGS's only news backend, ``duckduckgo``.
DDGS_BACKENDS = "duckduckgo,brave,mojeek,wikipedia"
_DDGS_TIMELIMIT = {"day": "d", "week": "w", "month": "m", "year": "y"}
DDGS_EMPTY_NOTE = "provider returned no results; it may be blocking requests"


class DDGSSearch:
    """The ``ddgs`` metasearch library: free, without a key and without guarantees.

    It queries public search engines' web pages (backends fixed to :data:`DDGS_BACKENDS`), which may block it;
    DDGS cannot tell a block from an empty answer, so an empty result carries the note
    ``"provider returned no results; it may be blocking requests"``. Its requests do not go through the egress
    guard (they go to fixed public hosts, never to caller-chosen URLs). Only ``text`` and ``news`` are used:
    never ``books`` and never ``extract``. Runs in a worker thread.
    """

    name = "ddgs"
    capabilities = SearchCapabilities(
        kinds=frozenset({"web", "news"}),
        region=True,
        language=True,
        time_range=True,
        pagination=True,
        content_included=False,
        max_results_cap=20,
    )

    def __init__(self, *, timeout: int = 10, safesearch: str = "moderate") -> None:
        if safesearch not in ("on", "moderate", "off"):
            raise ValueError("safesearch must be 'on', 'moderate' or 'off'")
        self.timeout = timeout
        self.safesearch = safesearch

    @staticmethod
    def _load() -> Any:
        try:
            import ddgs
        except ImportError:
            from ..errors import MissingDependencyError

            raise MissingDependencyError(
                'The DDGS search needs ddgs: pip install "orkexa[web-ddgs]"'
            ) from None
        return ddgs

    async def check(self) -> None:
        self._load()

    async def search(self, request: SearchRequest) -> SearchResponse:
        notes = unsupported_notes(request, self.capabilities, self.name)
        if request.region and request.language:
            region = f"{request.region.lower()}-{request.language.lower()}"
        else:
            region = "wt-wt"
            if request.region or request.language:
                notes.append("ddgs needs both region and language; both were ignored")
        rows = await asyncio.to_thread(self._run, request, region)
        count = min(request.max_results, self.capabilities.max_results_cap)
        hits: list[SearchHit] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            url = row.get("url") if request.kind == "news" else row.get("href")
            if not isinstance(url, str) or not url:
                continue
            hits.append(
                SearchHit(
                    title=_plain(row.get("title")),
                    url=url,
                    snippet=_plain(row.get("body")),
                    published_at=_text_or_none(row.get("date")),
                    source=_text_or_none(row.get("source")) or _host(url),
                    rank=_rank(request, count, len(hits) + 1),
                )
            )
            if len(hits) == count:
                break
        if not hits:
            notes.append(DDGS_EMPTY_NOTE)
        return SearchResponse(hits=hits, provider=self.name, notes=notes, billed_units=0)

    def _run(self, request: SearchRequest, region: str) -> list[Any]:
        ddgs = self._load()
        from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

        client = ddgs.DDGS(timeout=self.timeout)
        kwargs: dict[str, Any] = {
            "region": region,
            "safesearch": self.safesearch,
            "timelimit": _DDGS_TIMELIMIT[request.time_range] if request.time_range else None,
            "max_results": min(request.max_results, self.capabilities.max_results_cap),
            "page": request.page,
        }
        try:
            if request.kind == "news":
                rows = client.news(request.query, backend="duckduckgo", **kwargs)
            else:
                rows = client.text(request.query, backend=DDGS_BACKENDS, **kwargs)
        except RatelimitException:
            raise SearchRateLimited(
                "ddgs is rate limited by the search engines", provider=self.name
            ) from None
        except TimeoutException:
            raise SearchUnavailable("ddgs timed out", provider=self.name) from None
        except DDGSException:
            # "No results found." and engine errors look alike: DDGS cannot tell a block from an empty answer.
            return []
        except SearchError:
            raise
        except Exception as exc:  # the library's HTTP stack raises its own exception types
            raise SearchUnavailable(f"ddgs failed ({type(exc).__name__})", provider=self.name) from None
        return list(rows or [])
