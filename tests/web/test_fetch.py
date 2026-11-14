"""``fetch_page``: extraction, character sets, PDF, limits and the guard (NAL-A2).

Pages come from tests/fixtures/web (``article_latin1.html`` is stored in ISO-8859-1 on purpose) and
tests/fixtures/pdf, served by an ``httpx.MockTransport`` behind the egress guard; nothing is resolved or
connected.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import hashlib
from pathlib import Path
from typing import Any

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("trafilatura")

from orkexa.net.guard import EgressError, EgressPolicy  # noqa: E402
from orkexa.web import FetchedPage, FetchError, fetch_page  # noqa: E402
from orkexa.web.fetch import decode_text  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
UTF8 = (FIXTURES / "web" / "article_utf8.html").read_bytes()
LATIN1 = (FIXTURES / "web" / "article_latin1.html").read_bytes()
PDF = (FIXTURES / "pdf" / "normal_10p.pdf").read_bytes()


def serve(routes: dict[str, tuple[int, dict[str, str], bytes]]) -> tuple[httpx.MockTransport, list[str]]:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        status, headers, body = routes.get(request.url.path, (404, {}, b"missing"))
        return httpx.Response(status, headers=headers, content=body)

    return httpx.MockTransport(handler), seen


def fetch(
    routes: dict[str, tuple[int, dict[str, str], bytes]], path: str = "/page", **options: Any
) -> FetchedPage:
    transport, _seen = serve(routes)
    page: FetchedPage = asyncio.run(
        fetch_page(f"https://site.example{path}", mock_transport=transport, **options)
    )
    return page


def html(
    body: bytes, content_type: str = "text/html; charset=utf-8"
) -> dict[str, tuple[int, dict[str, str], bytes]]:
    return {"/page": (200, {"Content-Type": content_type}, body)}


def failure(call: Callable[[], Any]) -> FetchError:
    with pytest.raises(FetchError) as caught:
        call()
    return caught.value


def test_utf8_article_becomes_markdown() -> None:
    page = fetch(html(UTF8))
    assert page.title == "Brücken in Österreich"
    assert page.charset == "utf-8" and page.content_type == "text/html" and page.status == 200
    assert "Die Größe der Brücke" in page.text and "Grüße aus Österreich" in page.text
    assert "tracking" not in page.text and "color: red" not in page.text
    assert page.final_url == "https://site.example/page" and page.redirects == []
    assert page.size == len(UTF8) and page.sha256 == hashlib.sha256(UTF8).hexdigest()
    assert not page.truncated and page.fetched_at.tzinfo is not None


def test_latin1_article() -> None:
    page = fetch(html(LATIN1, "text/html; charset=ISO-8859-1"))
    assert page.charset == "cp1252"  # WHATWG: the Latin-1 label means windows-1252
    assert page.title == "Brücken in Österreich" and "Die Größe der Brücke" in page.text


def test_latin1_article_found_through_the_meta_tag() -> None:
    page = fetch(html(LATIN1, "text/html"))
    assert page.charset == "cp1252" and "Größe" in page.text


def test_a_wrong_header_does_not_garble_the_text() -> None:
    # The header says Latin-1, the bytes are UTF-8.
    page = fetch(html(UTF8, "text/html; charset=iso-8859-1"))
    assert page.charset == "utf-8" and "Größe" in page.text and "GrÃ¶ÃŸe" not in page.text
    # The header says UTF-8, the bytes are Latin-1: the <meta> declaration decides.
    page = fetch(html(LATIN1, "text/html; charset=utf-8"))
    assert page.charset == "cp1252" and "Größe" in page.text


def test_an_empty_page() -> None:
    page = fetch(html((FIXTURES / "web" / "empty.html").read_bytes()))
    assert (page.title, page.text, page.truncated) == (None, "", False)
    blank = fetch(html(b""))
    assert blank.text == "" and blank.size == 0


def test_a_paywall_stub_keeps_its_little_text() -> None:
    page = fetch(html((FIXTURES / "web" / "paywall.html").read_bytes()))
    assert page.title == "Premium story" and "Subscribe to continue reading." in page.text


def test_plain_text() -> None:
    page = fetch(html("Grüße\nzwei Zeilen\n".encode(), "text/plain; charset=utf-8"))
    assert page.text == "Grüße\nzwei Zeilen" and page.title is None


def test_pdf_text() -> None:
    page = fetch({"/doc.pdf": (200, {"Content-Type": "application/pdf"}, PDF)}, "/doc.pdf")
    assert page.content_type == "application/pdf" and page.charset is None
    assert page.text.startswith("Page 1 of the normal fixture")
    assert page.text.count("Page ") >= 10


def test_pdf_limits_and_magic_bytes() -> None:
    routes = {"/doc.pdf": (200, {"Content-Type": "application/pdf"}, PDF)}
    assert failure(lambda: fetch(routes, "/doc.pdf", max_pdf_pages=5)).code == "fetch.too_large"
    assert failure(lambda: fetch(routes, "/doc.pdf", max_bytes=1000)).code == "fetch.too_large"
    fake = {"/doc.pdf": (200, {"Content-Type": "application/pdf"}, b"<html>not a pdf</html>")}
    assert failure(lambda: fetch(fake, "/doc.pdf")).code == "fetch.unreadable"
    broken = {"/doc.pdf": (200, {"Content-Type": "application/pdf"}, b"%PDF-1.7\n garbage")}
    assert failure(lambda: fetch(broken, "/doc.pdf")).code == "fetch.unreadable"


def test_a_long_html_page_is_truncated() -> None:
    body = b"<html><body><article>" + b"<p>" + b"word " * 40_000 + b"</p></article></body></html>"
    page = fetch(html(body), max_bytes=10_000)
    assert page.truncated and page.size == 10_000
    assert page.sha256 == hashlib.sha256(body[:10_000]).hexdigest()
    assert page.text.startswith("word word")
    # Reaching the policy's own limit truncates in the same way.
    page = fetch(html(body), policy=EgressPolicy(max_bytes=20_000))
    assert page.truncated and page.size <= 20_000


def test_truncation_inside_a_multibyte_character_keeps_utf8() -> None:
    body = "<html><body><p>" + "ä" * 10_000 + "</p></body></html>"
    page = fetch(html(body.encode("utf-8"), "text/html"), max_bytes=1_001)
    assert page.truncated and page.charset == "utf-8" and "ää" in page.text


def test_types_outside_accept_are_refused_before_reading() -> None:
    routes = {"/page": (200, {"Content-Type": "application/zip"}, b"PK\x03\x04")}
    assert failure(lambda: fetch(routes)).code == "fetch.unsupported_type"
    assert failure(lambda: fetch(html(UTF8), accept={"application/pdf"})).code == "fetch.unsupported_type"
    routes = {"/page": (200, {"Content-Type": "image/svg+xml"}, b"<svg/>")}
    assert failure(lambda: fetch(routes)).code == "fetch.unsupported_type"


def test_untyped_answers_are_sniffed() -> None:
    assert fetch({"/page": (200, {}, UTF8)}).content_type == "text/html"
    assert fetch({"/page": (200, {}, PDF)}).content_type == "application/pdf"
    assert failure(lambda: fetch({"/page": (200, {}, b"\x00\x01binary")})).code == "fetch.unsupported_type"


def test_error_statuses() -> None:
    error = failure(lambda: fetch({"/page": (503, {"Content-Type": "text/html"}, b"down")}))
    assert (error.code, error.status) == ("fetch.status", 503)
    assert failure(lambda: fetch({}, "/missing")).status == 404


def test_redirects_are_followed_and_reported() -> None:
    routes = {
        "/old": (301, {"Location": "https://site.example/new"}, b""),
        "/new": (302, {"Location": "/page"}, b""),
        "/page": (200, {"Content-Type": "text/html; charset=utf-8"}, UTF8),
    }
    page = fetch(routes, "/old")
    assert page.final_url == "https://site.example/page"
    assert page.redirects == ["https://site.example/old", "https://site.example/new"]
    with pytest.raises(EgressError) as caught:
        fetch({"/page": (302, {}, b"")})
    assert caught.value.code == "redirect_invalid"


def test_the_guard_applies_to_every_hop() -> None:
    routes = {"/page": (302, {"Location": "http://127.0.0.1/admin"}, b"")}
    transport, seen = serve(routes)
    with pytest.raises(EgressError) as caught:
        asyncio.run(fetch_page("https://site.example/page", mock_transport=transport))
    assert caught.value.reason == "loopback" and seen == ["https://site.example/page"]
    with pytest.raises(EgressError) as caught:
        asyncio.run(fetch_page("http://169.254.169.254/latest/meta-data/", mock_transport=transport))
    assert caught.value.reason == "metadata"
    policy = EgressPolicy(deny_hosts=frozenset({"*.example"}))
    with pytest.raises(EgressError) as caught:
        asyncio.run(fetch_page("https://site.example/", policy, mock_transport=transport))
    assert caught.value.reason == "denied"
    assert seen == ["https://site.example/page"]


def test_network_failures() -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(FetchError) as caught:
        asyncio.run(
            fetch_page("https://site.example/a?token=SECRET", mock_transport=httpx.MockTransport(broken))
        )
    assert caught.value.code == "fetch.network" and "SECRET" not in str(caught.value)


def test_compressed_answers_are_refused() -> None:
    with pytest.raises(EgressError) as caught:
        fetch({"/page": (200, {"Content-Type": "text/html", "Content-Encoding": "br"}, b"x")})
    assert caught.value.code == "content_encoding"


def test_the_request_asks_for_accepted_types_only() -> None:
    headers: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers.append(request.headers)
        return httpx.Response(200, headers={"Content-Type": "text/plain"}, content=b"ok")

    asyncio.run(fetch_page("https://site.example/", mock_transport=httpx.MockTransport(handler)))
    assert headers[0]["Accept"] == "application/pdf, application/xhtml+xml, text/html, text/plain"
    assert headers[0]["Accept-Encoding"] == "identity" and "orkexa" in headers[0]["User-Agent"]


@pytest.mark.parametrize(
    ("data", "declared", "expected"),
    [
        (b"plain ascii", None, ("plain ascii", "utf-8")),
        (b"plain ascii", "iso-8859-1", ("plain ascii", "cp1252")),
        ("﻿mit BOM: ä".encode(), None, ("mit BOM: ä", "utf-8")),
        ("ä".encode("utf-16-le").join([b"\xff\xfe", b""]), None, ("ä", "utf-16-le")),
        ("Größe".encode("cp1252"), "windows-1252", ("Größe", "cp1252")),
        ("Größe".encode("cp1252"), "no-such-charset", None),  # detection decides; any decoding without errors
    ],
)
def test_decode_text(data: bytes, declared: str | None, expected: tuple[str, str] | None) -> None:
    text, codec = decode_text(data, declared, html=False)
    if expected is not None:
        assert (text, codec) == expected
    else:
        assert text and codec
