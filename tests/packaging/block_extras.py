"""A pytest plugin for the core suite without extras (MUP-ONA T-PKG-1): blocks psycopg and pglast.

``pytest -p tests.packaging.block_extras ...`` makes ``import psycopg`` and ``import pglast`` fail as if the
extras ``postgres`` and ``sqlcheck`` were not installed (``find_spec`` returns ``None``), before any test
module is imported.
"""

from __future__ import annotations

import sys

BLOCKED = ("psycopg", "psycopg_binary", "psycopg_c", "psycopg_pool", "pglast")

for _name in BLOCKED:
    sys.modules[_name] = None  # type: ignore[assignment]
