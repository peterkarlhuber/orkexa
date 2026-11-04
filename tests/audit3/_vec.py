"""Skip marker for tests that need the sqlite-vec extension (not loadable on every CI platform)."""

import importlib.util
import sqlite3

import pytest

HAS_VEC = importlib.util.find_spec("sqlite_vec") is not None and hasattr(
    sqlite3.connect(":memory:"), "enable_load_extension"
)
needs_vec = pytest.mark.skipif(not HAS_VEC, reason="sqlite-vec cannot be loaded here")
