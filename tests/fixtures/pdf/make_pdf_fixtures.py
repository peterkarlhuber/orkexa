"""Write the PDF fixtures of the PDF limit tests (MUP-ONA A1, ``tests/test_pdf_limits.py``).

Usage (from the repository root)::

    python tests/fixtures/pdf/make_pdf_fixtures.py [--out tests/fixtures/pdf]

Every file is built byte for byte by this script (no PDF library, no external files), so the output is
deterministic: the small files are committed and ``tests/test_pdf_limits.py`` checks that the script
reproduces them; ``pages_100k.pdf`` (12 MB) and ``flate_bomb.pdf`` (0.5 MB) are only generated at test time.

* ``normal_10p.pdf``: ten pages with one line of text each (the expected text is :data:`NORMAL_TEXTS`).
* ``pages_100k.pdf``: 100,000 pages (one shared content stream); far over the page budget.
* ``nested_objects.pdf``: one page whose resources hold arrays nested 5,000 levels deep.
* ``flate_bomb.pdf``: eight pages, each with its own 64 MiB FlateDecode content stream (below pypdf's
  per-stream decompression limits; together far over a 256 MiB memory limit); 0.5 MB, generated at test time.
* ``slow_content_stream.pdf``: five pages whose content streams hold 150,000 text operators each (below
  pypdf's per-stream limits), so that text extraction runs for many seconds (the pattern of the pypdf
  advisories about long runtimes for large streams).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path
import sys
import zlib

NORMAL_TEXTS = [
    f"Page {number} of the normal fixture: warranty, shipping and refunds." for number in range(1, 11)
]
COMMITTED = ("normal_10p.pdf", "nested_objects.pdf", "slow_content_stream.pdf")

_FONT = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
_CATALOG = b"<< /Type /Catalog /Pages 2 0 R >>"


def _pdf(objects: Sequence[bytes]) -> bytes:
    """A PDF whose object N (1-based) is ``objects[N - 1]``; object 1 must be the catalog."""
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def _stream(data: bytes, *, compress: bool = False) -> bytes:
    if compress:
        data = zlib.compress(data, 9)
        return b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(data) + data + b"\nendstream"
    return b"<< /Length %d >>\nstream\n" % len(data) + data + b"\nendstream"


def _text(line: str) -> bytes:
    escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("latin-1")
    return b"BT /F1 12 Tf 72 720 Td (" + escaped + b") Tj ET"


def _page(contents: bytes, resources: bytes = b"<< /Font << /F1 3 0 R >> >>") -> bytes:
    return (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources "
        + resources
        + b" /Contents "
        + contents
        + b" >>"
    )


def normal_10p() -> bytes:
    pages = len(NORMAL_TEXTS)
    kids = b" ".join(b"%d 0 R" % (4 + 2 * index) for index in range(pages))
    objects = [_CATALOG, b"<< /Type /Pages /Kids [" + kids + b"] /Count %d >>" % pages, _FONT]
    for index, line in enumerate(NORMAL_TEXTS):
        objects.append(_page(b"%d 0 R" % (5 + 2 * index)))
        objects.append(_stream(_text(line)))
    return _pdf(objects)


def pages_100k(count: int = 100_000) -> bytes:
    kids = b" ".join(b"%d 0 R" % (5 + index) for index in range(count))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [" + kids + b"] /Count %d >>" % count,
        _FONT,
        _stream(_text("One of very many pages.")),
    ]
    objects += [b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>"] * count
    return _pdf(objects)


def nested_objects(depth: int = 5000) -> bytes:
    nested = b"[" * depth + b"1" + b"]" * depth
    resources = b"<< /Font << /F1 3 0 R >> /Properties << /Deep " + nested + b" >> >>"
    return _pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
            _FONT,
            _page(b"5 0 R", resources),
            _stream(_text("Text next to deeply nested objects.")),
        ]
    )


def _zeros(size: int) -> Iterator[bytes]:
    block = b" " * (1 << 20)  # spaces: valid (empty) content stream syntax
    for _ in range(size >> 20):
        yield block


def _compressed_stream(chunks: Iterator[bytes]) -> bytes:
    compressor = zlib.compressobj(9)
    data = b"".join(compressor.compress(chunk) for chunk in chunks) + compressor.flush()
    return b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(data) + data + b"\nendstream"


def _pages_with_streams(streams: Sequence[bytes]) -> bytes:
    """One page per stream; every page has its own content stream object."""
    count = len(streams)
    kids = b" ".join(b"%d 0 R" % (4 + 2 * index) for index in range(count))
    objects = [_CATALOG, b"<< /Type /Pages /Kids [" + kids + b"] /Count %d >>" % count, _FONT]
    for index, stream in enumerate(streams):
        objects += [_page(b"%d 0 R" % (5 + 2 * index)), stream]
    return _pdf(objects)


def flate_bomb(pages: int = 8, size: int = 64 << 20) -> bytes:
    # Each page decompresses its own 64 MiB stream (below pypdf's per-stream limits); pypdf keeps the decoded
    # streams, so the pages together need far more than a 256 MiB memory limit.
    return _pages_with_streams([_compressed_stream(_zeros(size)) for _ in range(pages)])


def slow_content_stream(pages: int = 5, operators: int = 150_000) -> bytes:
    line = b"1 0 0 1 72 720 Tm (x) Tj\n"
    return _pages_with_streams(
        [
            _stream(b"BT /F1 12 Tf\n%d 0 Td\n" % page + line * operators + b"ET", compress=True)
            for page in range(pages)
        ]
    )


BUILDERS = {
    "normal_10p.pdf": normal_10p,
    "pages_100k.pdf": pages_100k,
    "nested_objects.pdf": nested_objects,
    "flate_bomb.pdf": flate_bomb,
    "slow_content_stream.pdf": slow_content_stream,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--all", action="store_true", help="also write the files that are not committed")
    args = parser.parse_args(argv)
    for name in BUILDERS if args.all else COMMITTED:
        data = BUILDERS[name]()
        (args.out / name).write_bytes(data)
        print(f"{name}: {len(data)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
