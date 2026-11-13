"""Fixtures of the PostgreSQL backend tests (MUP-ONA B5); the server setup is in :mod:`.support`."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from . import support

# TABRION MIG-ORK-1: the server of every "postgres" test comes from tabrion.testing (support.py).
from .support import orkexa_postgres_gate, tabrion_postgres_url  # noqa: F401  # fixtures


@pytest.fixture
def db() -> Iterator[support.Database]:
    """A fresh migrated test schema (marker ``postgres`` on the test)."""
    with support.database() as opened:
        yield opened
