"""The SQLite storage backends: :class:`SQLiteIndex`, :class:`SQLiteFileStore` and :class:`SQLiteMemory`.

They need nothing beyond Python's ``sqlite3`` (vector search additionally ``sqlite-vec``). This is their
canonical location; ``orkexa`` and the contract modules :mod:`orkexa.retrieval`, :mod:`orkexa.files` and
:mod:`orkexa.memory` re-export the same classes.
"""

from __future__ import annotations

from .files import SQLiteFileStore
from .index import SQLiteIndex
from .memory import SQLiteMemory

__all__ = ["SQLiteFileStore", "SQLiteIndex", "SQLiteMemory"]
