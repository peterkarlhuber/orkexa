"""The web layer imports without its extras and opens no connection on import (NAL-A2)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]

_CODE = """
import socket, sys
BLOCKED = ("httpx", "httpcore", "anyio", "trafilatura", "charset_normalizer", "ddgs", "primp", "pypdf")
for name in BLOCKED:
    sys.modules[name] = None
def _no_network(*args, **kwargs):
    raise AssertionError("a network connection was opened on import")
socket.socket.connect = _no_network
socket.create_connection = _no_network
socket.getaddrinfo = _no_network

import orkexa.net.guard as guard
import orkexa.web as web
from orkexa.errors import MissingDependencyError

assert guard.EgressPolicy().check_url("https://example.com/").host == "example.com"
assert web.TestSearchProvider().name == "test"
loaded = sorted(name for name in sys.modules if name.split(".")[0] in BLOCKED)
assert all(sys.modules[name] is None for name in loaded), loaded
try:
    import orkexa.net.http
except MissingDependencyError as exc:
    assert 'orkexa[web]' in str(exc)
else:
    raise AssertionError("orkexa.net.http imported without httpx")
import asyncio
for call in (lambda: web.fetch_page("https://example.com/"), lambda: web.BraveSearch("k").check()):
    try:
        asyncio.run(call())
    except MissingDependencyError as exc:
        assert 'orkexa[web]' in str(exc)
    else:
        raise AssertionError("no MissingDependencyError")
print("ok")
"""


def test_the_web_layer_imports_without_its_extras() -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(_CODE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stdout + result.stderr
