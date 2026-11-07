"""The conformance suite against the local backends without partitions (MUP-ONA B1: T-CONF-5, T-CONF-6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from orkexa.testing.conformance import ALL_FILE_STORE_CASES, ALL_MEMORY_CASES, Case

from .adapter import run_and_check
from .factories import FACTORIES


# MUP-ONA T-CONF-5
# MUP-ONA T-CONF-6
@pytest.mark.parametrize(
    ("target", "cases"),
    [
        ("local-file-store", ALL_FILE_STORE_CASES),
        ("file-memory", ALL_MEMORY_CASES),
        ("ephemeral-memory", ALL_MEMORY_CASES),
    ],
)
def test_the_local_backends_conform(target: str, cases: tuple[Case, ...], tmp_path: Path) -> None:
    factory, expect = FACTORIES[target]
    report = run_and_check(target, factory, cases, expect, tmp_path)
    assert report.passed
    assert report.skipped  # no partitions, no concurrent writers: the expectation file names those cases
