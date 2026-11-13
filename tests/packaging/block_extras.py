"""A pytest plugin for the core suite without extras (MUP-ONA T-PKG-1): blocks psycopg, pglast and tabrion.

``pytest -p tests.packaging.block_extras ...`` makes ``import psycopg``, ``import pglast`` and
``import tabrion`` fail as if the extras ``postgres`` and ``sqlcheck`` and the ``dev`` extra's Tabrion
(TABRION MIG-ORK-1) were not installed (``find_spec`` returns ``None``), before any test module is imported.
"""

from __future__ import annotations

import sys

BLOCKED = ("psycopg", "psycopg_binary", "psycopg_c", "psycopg_pool", "pglast", "tabrion")

for _name in BLOCKED:
    sys.modules[_name] = None  # type: ignore[assignment]
