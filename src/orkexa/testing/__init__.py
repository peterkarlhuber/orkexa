"""Helpers for testing code that uses Orkexa, and for testing Orkexa backends (MUP-ONA B1).

* :mod:`orkexa.testing.fakes`: :class:`~orkexa.testing.fakes.ScriptedClient`, a scripted stand-in for a
  provider SDK client (no network).
* :mod:`orkexa.testing.conformance`: the public conformance suite of the storage contracts (``Index``,
  ``ReindexableIndex``, ``FileStore``, ``MemoryBackend``), runnable against any backend.

Nothing here needs pytest, and no Orkexa module outside this package imports it. Import the submodules
directly; this package itself exports nothing.
"""
