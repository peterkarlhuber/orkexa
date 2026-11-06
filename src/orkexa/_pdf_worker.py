"""Child process of :func:`orkexa.retrieval.pdf_pages_isolated`: extracts the text of one PDF.

Started only by :mod:`orkexa._pdf_launch` as ``python -I -m orkexa._pdf_worker <pypdf directory>`` with a
minimal environment, an empty private working directory and no inherited file descriptors besides the three
pipes. Protocol:

* stdin: one JSON line with the limits (``max_pages``, ``max_chars``, ``max_memory_mb``, ``cpu_seconds``),
  then the PDF bytes until end of file.
* stdout: one JSON object, ``{"ok": true, "pages": [...]}`` or
  ``{"ok": false, "reason": ..., "message": ...}`` with the reasons ``pages``, ``chars``, ``memory``,
  ``limit`` (a pypdf limit) and ``invalid`` (not a readable PDF). The exit code is 0 for a result and 3 for a
  refusal.

Before pypdf is imported, the worker lowers its own resource limits (Linux: address space, CPU time, open
files, file size 0, no core files; macOS: the same without the address space, which macOS does not enforce;
Windows: none). It opens no files or sockets and reads no configuration. It imports only the standard library
and pypdf (``python -m`` imports the ``orkexa`` package first, before the limits apply).
"""

from __future__ import annotations

import io
import json
import sys
from typing import Any

EXIT_REFUSED = 3
OPEN_FILES = 16


def _limit(resource: Any, name: str, value: int) -> None:
    kind = getattr(resource, name, None)
    if kind is None:
        return
    _soft, hard = resource.getrlimit(kind)
    if hard != resource.RLIM_INFINITY:
        value = min(value, hard)
    resource.setrlimit(kind, (value, value))  # the hard limit too: the worker cannot raise it again


def apply_limits(limits: dict[str, Any]) -> None:
    """Lower this process's resource limits (POSIX only); the first thing the worker does after its header."""
    if sys.platform == "win32":
        return
    import resource

    cpu = int(limits["cpu_seconds"])
    kind = resource.RLIMIT_CPU
    _soft, hard = resource.getrlimit(kind)
    ceiling = cpu + 1 if hard == resource.RLIM_INFINITY else min(cpu + 1, hard)
    resource.setrlimit(
        kind, (min(cpu, ceiling), ceiling)
    )  # SIGXCPU at the soft limit, SIGKILL at the hard one
    _limit(resource, "RLIMIT_NOFILE", OPEN_FILES)
    _limit(resource, "RLIMIT_FSIZE", 0)  # writing a file fails; stdout and stderr are pipes
    _limit(resource, "RLIMIT_CORE", 0)
    if sys.platform.startswith("linux"):
        _limit(resource, "RLIMIT_AS", int(limits["max_memory_mb"]) << 20)


def _refuse(reason: str, message: str) -> int:
    _write({"ok": False, "reason": reason, "message": message[:500]})
    return EXIT_REFUSED


def _write(result: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(result, ensure_ascii=True))
    sys.stdout.flush()


def extract(data: bytes, max_pages: int, max_chars: int) -> int:
    from pypdf import PdfReader
    from pypdf.errors import LimitReachedError, PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        trailer: Any = reader.trailer
        declared = trailer["/Root"]["/Pages"].get("/Count", 0)  # checked before the page tree is read
        if isinstance(declared, int) and declared > max_pages:
            return _refuse("pages", f"The PDF declares {declared} pages; at most {max_pages} are allowed")
        pages: list[str] = []
        total = 0
        for number, page in enumerate(reader.pages, start=1):
            if number > max_pages:
                return _refuse("pages", f"The PDF has more than {max_pages} pages")
            text = page.extract_text() or ""
            total += len(text)
            if total > max_chars:
                return _refuse("chars", f"The PDF has more than {max_chars} characters of text")
            pages.append(text)
    except MemoryError:
        return _refuse("memory", "The PDF needs more memory than allowed")
    except LimitReachedError as exc:
        return _refuse("limit", f"pypdf refused the PDF: {exc}")
    except (PyPdfError, ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        return _refuse("invalid", f"The PDF cannot be read: {type(exc).__name__}: {exc}")
    _write({"ok": True, "pages": pages})
    return 0


def main(argv: list[str]) -> int:
    header = json.loads(sys.stdin.buffer.readline())
    apply_limits(header)
    if argv:
        sys.path.append(argv[0])  # where the parent found pypdf (-I leaves out PYTHONPATH and user sites)
    data = sys.stdin.buffer.read()
    return extract(data, int(header["max_pages"]), int(header["max_chars"]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
