"""Child process of :func:`orkexa.web.fetch.html_to_text_isolated`: turns one HTML page into text.

Started only through :func:`orkexa._pdf_launch.run_worker` as
``python -I -m orkexa.web._html_worker <package directory>`` with a minimal environment, an empty private
working directory and no inherited file descriptors besides the three pipes. Protocol:

* stdin: one JSON line with the limits (``max_chars``, ``max_memory_mb``, ``cpu_seconds``) and the page's
  ``url`` (or ``null``), then the markup as UTF-8 until end of file.
* stdout: one JSON object, ``{"ok": true, "title": ..., "text": ...}`` or
  ``{"ok": false, "reason": ..., "message": ...}`` with the reasons ``memory`` and ``invalid``. The exit
  code is 0 for a result and 3 for a refusal.

The parsers are imported first, then the worker lowers its own resource limits (the same limits as the PDF
worker: CPU time, open files, file size 0, no core files and, on Linux, the address space) before it reads the
markup. Parsing a hostile page can therefore cost at most the CPU time and memory the parent allows.
"""

from __future__ import annotations

import contextlib
import json
import sys

EXIT_REFUSED = 3


def _write(result: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(result, ensure_ascii=True))
    sys.stdout.flush()


def main(argv: list[str]) -> int:
    if argv:
        sys.path.append(
            argv[0]
        )  # where the parent found trafilatura (-I leaves out PYTHONPATH and user sites)
    from orkexa._pdf_worker import apply_limits
    from orkexa.web.fetch import html_to_text

    with contextlib.suppress(ImportError):  # import the parser before the limits apply
        import trafilatura  # noqa: F401
    header = json.loads(sys.stdin.buffer.readline())
    apply_limits(header)
    try:
        markup = sys.stdin.buffer.read().decode("utf-8", errors="replace")
        url = header.get("url")
        title, text = html_to_text(markup, url if isinstance(url, str) else None)
    except MemoryError:
        _write({"ok": False, "reason": "memory", "message": "The page needs more memory than allowed"})
        return EXIT_REFUSED
    except Exception as exc:  # a parser failure on hostile markup
        _write(
            {"ok": False, "reason": "invalid", "message": f"The page cannot be parsed ({type(exc).__name__})"}
        )
        return EXIT_REFUSED
    max_chars = int(header["max_chars"])
    if len(text) > max_chars:
        text = text[:max_chars]
    _write({"ok": True, "title": title, "text": text})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
