"""The thin pytest adapter of the conformance suite (MUP-ONA B1, item k): run, save the report, compare skips.

``ORKEXA_CONFORMANCE_JOB`` names the CI job whose entry of ``expected_skips.json`` applies (default ``test``);
``ORKEXA_CONFORMANCE_REPORTS`` names a directory that receives every report as JSON (a CI artifact).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
import os
from pathlib import Path
from typing import Any

from orkexa.testing.conformance import Case, Report, load_expected_skips, run_cases

EXPECTED_SKIPS = Path(__file__).resolve().parent / "expected_skips.json"


def job() -> str:
    return os.environ.get("ORKEXA_CONFORMANCE_JOB", "test")


def save(report: Report) -> None:
    folder = os.environ.get("ORKEXA_CONFORMANCE_REPORTS")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = Path(folder) / f"{report.target}.json"
        path.write_text(json.dumps(report.to_dict(), indent=1), encoding="utf-8")


def run_and_check(
    target: str, factory: Callable[..., Any], cases: Sequence[Case], expect: frozenset[str], tmp_path: Path
) -> Report:
    """Run ``cases``, save the report; fail unless every case passed and the skips match the expectation."""
    report = run_cases(factory, cases, expect=expect, target=target, workdir=tmp_path)
    save(report)
    report.check(load_expected_skips(EXPECTED_SKIPS, job(), target))
    return report
