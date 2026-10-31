"""Root test configuration."""

from __future__ import annotations

import re

import pytest


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
