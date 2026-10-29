"""Root test configuration.

The unchanged ``claude-agent`` 0.8.0 suite in ``tests/legacy`` imports its helpers as
``tests.conftest`` (its original location); they are re-exported here.
"""

from __future__ import annotations

import re

import pytest

from .legacy.conftest import (  # noqa: F401
    AsyncFakeClient,
    AsyncFakeMessages,
    AsyncFakeStream,
    FakeClient,
    FakeMessages,
    FakeStream,
    ParsedResponse,
    make_beta_message,
    make_message,
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Live tests call paid APIs: they run only when selected with ``-m live``.

    An API key in the environment is not enough, so a plain ``pytest`` never makes a
    paid request.
    """
    expression = config.getoption("-m") or ""
    if re.search(r"(?<!not )\blive\b", expression):
        return
    skip = pytest.mark.skip(reason="live API test; run with: pytest -m live (needs API keys, costs money)")
    for item in items:
        if item.get_closest_marker("live") is not None:
            item.add_marker(skip)
