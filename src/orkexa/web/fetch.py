"""Fetch one web page through the egress guard and turn it into text (ADR 0017).

Needs ``pip install "orkexa[web]"`` (httpx, trafilatura, pypdf).
"""

from __future__ import annotations

import asyncio
import codecs
from collections.abc import Iterable
from datetime import UTC, datetime
import hashlib
from html.parser import HTMLParser
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..errors import OrkexaError
from ..net.guard import DEFAULT_MAX_BYTES, EgressError, EgressPolicy, Resolver, redact_url

if TYPE_CHECKING:
    import httpx

__all__ = ["DEFAULT_ACCEPT", "FetchError", "FetchedPage", "fetch_page"]

#: The media types :func:`fetch_page` accepts by default.
DEFAULT_ACCEPT = frozenset({"text/html", "application/xhtml+xml", "text/plain", "application/pdf"})
USER_AGENT = "Mozilla/5.0 (compatible; orkexa-fetch)"
_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})


class FetchedPage(BaseModel):
    """A fetched page as text: Markdown for HTML (plain text if extraction finds no main content), the text
    layer of a PDF, or the text itself.

    ``truncated``: the body was longer than the byte limit and only its beginning was used. ``size`` and
    ``sha256`` describe the bytes read. ``redirects`` lists the URLs that redirected, in order; ``final_url``
    is the URL of the answer.
    """

    final_url: str
    status: int
    content_type: str
    charset: str | None = None
    title: str | None = None
    text: str
    truncated: bool = False
    size: int = Field(ge=0)
    sha256: str
    fetched_at: datetime
    redirects: list[str] = Field(default_factory=list)


class FetchError(OrkexaError, RuntimeError):
    """A page could not be fetched or read. ``code``:

    * ``fetch.status``: the server answered with an error status (``status`` is set);
    * ``fetch.unsupported_type``: a media type outside ``accept``;
    * ``fetch.too_large``: a PDF larger than the byte limit, or with more pages than allowed;
    * ``fetch.unreadable``: the content is not what its type says (a "PDF" without the PDF header) or cannot
      be parsed;
    * ``fetch.network``: the connection failed.

    Refusals and limits of the guard itself raise :class:`~orkexa.net.guard.EgressError`.
    """

    def __init__(self, code: str, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def _accept_header(accept: Iterable[str]) -> str:
    return ", ".join(sorted(accept))


def _media_type(header: str) -> tuple[str, str | None]:
    """``(media type, charset)`` of a ``Content-Type`` header."""
    parts = header.split(";")
    media = parts[0].strip().lower()
    charset = None
    for part in parts[1:]:
        name, _, value = part.partition("=")
        if name.strip().lower() == "charset":
            charset = value.strip().strip("\"'").lower() or None
    return media, charset


def _sniff(data: bytes) -> str:
    head = data[:1024].lstrip().lower()
    if b"%PDF-" in data[:1024]:
        return "application/pdf"
    if head.startswith((b"<!doctype html", b"<html", b"<head", b"<body")):
        return "text/html"
    return ""


# ---- character sets ----------------------------------------------------------------------------------------

_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.I)
_LATIN1_LABELS = frozenset({"iso-8859-1", "iso8859-1", "latin-1", "latin1", "l1", "us-ascii", "ascii"})


def _codec(label: str | None) -> str | None:
    """A Python codec for a charset label (WHATWG: Latin-1 and ASCII labels mean windows-1252)."""
    if not label:
        return None
    label = label.strip().lower()
    if label in _LATIN1_LABELS:
        return "cp1252"
    try:
        return codecs.lookup(label).name
    except LookupError:
        return None


def _decode_with(data: bytes, codec: str, final: bool) -> str | None:
    try:
        return codecs.getincrementaldecoder(codec)("strict").decode(data, final=final)
    except (UnicodeDecodeError, LookupError):
        return None


def decode_text(data: bytes, declared: str | None, *, html: bool, truncated: bool = False) -> tuple[str, str]:
    """``(text, codec)``: byte-order mark, then valid UTF-8, then the declared charset (header, ``<meta>``),
    then detection (charset-normalizer), then UTF-8 with replacement characters.

    Valid UTF-8 with non-ASCII bytes wins over a declared single-byte charset: such bytes hardly ever occur
    in real Latin-1 text, and a wrong header is the more common error."""
    final = not truncated
    for bom, codec in (
        (codecs.BOM_UTF8, "utf-8"),
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
    ):
        if data.startswith(bom):
            text = _decode_with(data[len(bom) :], codec, final)
            if text is not None:
                return text, codec
    if data.isascii():
        return data.decode("ascii"), _codec(declared) or "utf-8"
    utf8 = _decode_with(data, "utf-8", final)
    if utf8 is not None:
        return utf8, "utf-8"
    candidates = [_codec(declared)]
    if html:
        match = _META_CHARSET.search(data[:4096])
        if match:
            candidates.append(_codec(match.group(1).decode("ascii", "ignore")))
    for candidate in candidates:
        if candidate and candidate not in ("utf-8", "utf-16", "utf-32"):
            text = _decode_with(data, candidate, final)
            if text is not None:
                return text, candidate
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(data[:200_000]).best()
        if best is not None and best.encoding:
            text = _decode_with(data, best.encoding, final)
            if text is not None:
                return text, codecs.lookup(best.encoding).name
    except ImportError:  # pragma: no cover - charset-normalizer comes with orkexa[web]
        pass
    return data.decode("utf-8", "replace"), "utf-8"


# ---- HTML --------------------------------------------------------------------------------------------------

_SKIP = frozenset({"script", "style", "noscript", "template", "svg", "head", "iframe", "object"})
_BLOCK = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "ul",
        "ol",
        "tr",
        "table",
        "section",
        "article",
        "header",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "pre",
        "blockquote",
        "hr",
        "dd",
        "dt",
        "main",
        "nav",
        "aside",
        "form",
    }
)


class _TextParser(HTMLParser):
    """Plain text and ``<title>`` of an HTML page (the fallback when trafilatura finds no main content)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag == "title":
            self._in_title = True
        elif tag in _SKIP:
            self._skip += 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        elif not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self.parts).splitlines())
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    def title(self) -> str | None:
        return " ".join("".join(self.title_parts).split()) or None


def html_to_text(markup: str, url: str | None = None) -> tuple[str | None, str]:
    """``(title, text)`` of an HTML page: trafilatura's Markdown of the main content, or plain text."""
    parser = _TextParser()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:
        pass
    title = parser.title()
    text = ""
    try:
        import trafilatura

        text = (
            trafilatura.extract(
                markup,
                url=url,
                output_format="markdown",
                include_comments=False,
                include_tables=True,
                include_links=False,
                include_images=False,
            )
            or ""
        )
        metadata = trafilatura.extract_metadata(markup)
        if metadata is not None and metadata.title:
            title = metadata.title
    except ImportError:  # pragma: no cover - trafilatura comes with orkexa[web]
        pass
    return title, text.strip() or parser.text()


# ---- fetching ----------------------------------------------------------------------------------------------


async def fetch_page(
    url: str,
    policy: EgressPolicy | None = None,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    accept: Iterable[str] = DEFAULT_ACCEPT,
    max_pdf_pages: int = 100,
    user_agent: str = USER_AGENT,
    resolver: Resolver | None = None,
    mock_transport: httpx.AsyncBaseTransport | None = None,
) -> FetchedPage:
    """Fetch ``url`` within ``policy`` (default :class:`~orkexa.net.guard.EgressPolicy`), extract its text.

    * Redirects are followed by hand, each hop checked again; ``policy.timeout`` bounds the whole exchange.
    * At most ``min(max_bytes, policy.max_bytes)`` bytes are used: a longer HTML or text body is cut there
      (``truncated=True``), a longer PDF is refused (``fetch.too_large``).
    * The media type must be in ``accept``; a body without a type is sniffed (PDF header, HTML start tag).
      PDFs must start with the PDF header and are read in a separate, time- and memory-limited process
      (:func:`orkexa.retrieval.pdf_pages_isolated`) with at most ``max_pdf_pages`` pages.
    * HTML becomes Markdown of the main content (trafilatura), or plain text when none is found.

    Raises :class:`~orkexa.net.guard.EgressError` (refused address, redirect limit, timeout, size of the
    whole answer) or :class:`FetchError`.
    """
    try:
        from ..net.http import deadline, guarded_client, send_following_redirects
    except ImportError:
        from ..errors import MissingDependencyError

        raise MissingDependencyError('fetch_page needs httpx: pip install "orkexa[web]"') from None
    import httpx

    policy = policy or EgressPolicy()
    accepted = frozenset(media.lower() for media in accept)
    limit = min(max_bytes, policy.max_bytes)
    if limit < 1 or max_pdf_pages < 1:
        raise ValueError("max_bytes and max_pdf_pages must be positive")
    headers = {"User-Agent": user_agent, "Accept": _accept_header(accepted)}
    chunks: list[bytes] = []
    truncated = False
    try:
        async with guarded_client(
            policy, headers=headers, resolver=resolver, mock_transport=mock_transport
        ) as client:
            with deadline(policy):
                request = client.build_request("GET", url)
                response, redirects = await send_following_redirects(client, request, policy)
                try:
                    status = response.status_code
                    if status >= 300:
                        raise FetchError("fetch.status", f"The server answered HTTP {status}", status=status)
                    media, declared_charset = _media_type(response.headers.get("content-type", ""))
                    if media and media not in accepted:
                        raise FetchError("fetch.unsupported_type", f"The media type {media} is not accepted")
                    declared = response.headers.get("content-length", "").strip()
                    if media == "application/pdf" and declared.isdigit() and int(declared) > limit:
                        raise FetchError("fetch.too_large", f"The PDF is larger than {limit} bytes")
                    total = 0
                    try:
                        async for chunk in response.aiter_raw():
                            if total + len(chunk) > limit:
                                chunks.append(chunk[: limit - total])
                                truncated = True
                                break
                            chunks.append(chunk)
                            total += len(chunk)
                    except EgressError as exc:
                        if exc.code != "response_too_large":
                            raise
                        truncated = True  # the answer exceeded the policy's limit, which equals ours
                finally:
                    await response.aclose()
    except httpx.HTTPError as exc:
        raise FetchError(
            "fetch.network", f"{redact_url(url)} cannot be fetched ({type(exc).__name__})"
        ) from None
    data = b"".join(chunks)
    final_url = str(response.url)
    if not media:
        media = _sniff(data)
        if media not in accepted:
            raise FetchError("fetch.unsupported_type", "The answer has no media type and is not HTML or PDF")
    title: str | None = None
    charset: str | None = None
    if media == "application/pdf":
        if truncated:
            raise FetchError("fetch.too_large", f"The PDF is larger than {limit} bytes")
        text = await _pdf_text(data, max_pdf_pages)
    else:
        html = media in _HTML_TYPES
        decoded, codec = decode_text(data, declared_charset, html=html, truncated=truncated)
        charset = codec
        if html:
            title, text = await asyncio.to_thread(html_to_text, decoded, final_url)
        else:
            text = decoded.strip()
    return FetchedPage(
        final_url=final_url,
        status=status,
        content_type=media,
        charset=charset,
        title=title,
        text=text,
        truncated=truncated,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        fetched_at=datetime.now(UTC),
        redirects=redirects,
    )


async def _pdf_text(data: bytes, max_pages: int) -> str:
    if b"%PDF-" not in data[:1024]:
        raise FetchError("fetch.unreadable", "The answer says it is a PDF but has no PDF header")
    from ..retrieval import RetrievalError, pdf_pages_isolated

    try:
        pages = await asyncio.to_thread(pdf_pages_isolated, data, max_pages=max_pages)
    except RetrievalError as exc:
        if exc.reason == "pages":
            raise FetchError("fetch.too_large", f"The PDF has more than {max_pages} pages") from None
        raise FetchError("fetch.unreadable", f"The PDF cannot be read ({exc.reason})") from None
    except ValueError:
        raise FetchError("fetch.unreadable", "The PDF cannot be read") from None
    return "\n\n".join(page.strip() for page in pages if page.strip())
