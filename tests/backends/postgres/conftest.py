"""Fixtures of the PostgreSQL backend tests (MUP-ONA B5); the server setup is in :mod:`.support`."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from . import support


@pytest.fixture
def db() -> Iterator[support.Database]:
    """A fresh migrated test schema (marker ``postgres`` on the test)."""
    with support.database() as opened:
        yield opened
