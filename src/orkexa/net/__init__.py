"""Outbound network building blocks (ADR 0017).

* :mod:`orkexa.net.guard`: the egress guard core (address rules, :class:`~orkexa.net.guard.EgressPolicy`,
  :class:`~orkexa.net.guard.GuardedBackend`). Standard library only at import time.
* :mod:`orkexa.net.http`: the ``httpx`` client built on the guard (``pip install "orkexa[web]"``).

The package itself exports nothing: import from the submodules.
"""

from __future__ import annotations

__all__: list[str] = []
