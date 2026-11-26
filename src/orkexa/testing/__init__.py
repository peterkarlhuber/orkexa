"""Helpers for testing code that uses Orkexa, and for testing Orkexa backends (MUP-ONA B1, B5).

* :mod:`orkexa.testing.fakes`: :class:`~orkexa.testing.fakes.ScriptedClient`, a scripted stand-in for a
  provider SDK client (no network).
* :mod:`orkexa.testing.conformance`: the public conformance suite of the storage contracts (``Index``,
  ``ReindexableIndex``, ``FileStore``, ``MemoryBackend``), runnable against any backend, and of read-only
  folder grants (``FolderAccess``, Nodestro N-63).
* :mod:`orkexa.testing.pathguard`: the attack catalog of the path guard :mod:`orkexa.fs` (Nodestro N-10).
* :mod:`orkexa.testing.sqlcheck`: a rule engine that checks that PostgreSQL statements on tenant-partitioned
  tables are bound to one tenant (MUP-ONA B5; needs ``pip install "pglast>=7,<9"``, a development dependency).

Nothing here needs pytest, and no Orkexa module outside this package imports it. Import the submodules
directly; this package itself exports nothing.
"""
