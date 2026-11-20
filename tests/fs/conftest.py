"""ORKEXA_TEST_FS_FALLBACKS=1 runs tests/fs on the POSIX fallbacks (CI job "fs fallbacks", review L1).

The fallbacks are what an older kernel or a seccomp filter leaves: no ``openat2`` (components are opened with
``openat`` and ``O_NOFOLLOW`` and checked by device and mount id) and no ``renameat2`` or ``renameatx_np`` (a
file is renamed by a hard link and an unlink; a folder cannot be renamed).
"""

from __future__ import annotations

from collections.abc import Iterator
import os
import sys

import pytest

FALLBACKS = os.environ.get("ORKEXA_TEST_FS_FALLBACKS") == "1" and os.name == "posix"


@pytest.fixture(autouse=True)
def _posix_fallbacks(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if FALLBACKS:
        from orkexa.fs import _posix

        if sys.platform.startswith("linux"):
            monkeypatch.setitem(_posix._openat2_state, "usable", False)
        monkeypatch.setattr(_posix, "_rename_call", lambda *args: None)
    yield
