"""Packaging tests build wheels and create virtual environments; they only run on request.

Run them with ``pytest -m packaging`` (needs ``uv`` or ``pip`` with access to a package
index for pydantic and the provider SDKs).
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if "packaging" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="packaging test; run with: pytest -m packaging")
    for item in items:
        if item.get_closest_marker("packaging") is not None:
            item.add_marker(skip)
