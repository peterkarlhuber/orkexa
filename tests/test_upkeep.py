"""Snapshots that age: built-in prices, the Bedrock capability table, the sandbox base image.

Each one records when it was last checked against its source. In the default run an old
snapshot is a warning (so that the age of a snapshot never turns an unrelated change red);
with ``ORKEXA_CHECK_FRESHNESS=1`` (the release workflow and the release checklist,
``docs/development/releasing.md``) it fails.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import os
from pathlib import Path
import re
import warnings

import pytest

from orkexa.pricing._builtin import BUILTIN_CHECKED_AT, BUILTIN_VERSION
from orkexa.providers.bedrock import BEDROCK_CAPABILITIES_CHECKED_AT

ROOT = Path(__file__).resolve().parents[1]
#: The most a snapshot may age before a release.
MAX_AGE = timedelta(days=90)
STRICT = os.environ.get("ORKEXA_CHECK_FRESHNESS") == "1"


def _sandbox_base_checked_at() -> date:
    text = (ROOT / "docker" / "sandbox.Dockerfile").read_text(encoding="utf-8")
    found = re.search(r"verified[\s#]+against[\s#]+\S+[\s#]+on[\s#]+(\d{4}-\d{2}-\d{2})", text)
    assert found, "docker/sandbox.Dockerfile must say when its base image digest was verified"
    return date.fromisoformat(found.group(1))


SNAPSHOTS = {
    f"built-in prices ({BUILTIN_VERSION}, src/orkexa/pricing/_builtin.py)": BUILTIN_CHECKED_AT.date(),
    "Bedrock capability table (bedrock_capabilities, BEDROCK_CAPABILITIES_CHECKED_AT)": (
        BEDROCK_CAPABILITIES_CHECKED_AT
    ),
    "sandbox base image digest and hashed requirements (docker/)": _sandbox_base_checked_at(),
}


def stale(snapshots: dict[str, date], today: date, max_age: timedelta = MAX_AGE) -> list[str]:
    """The snapshots older than ``max_age`` on ``today``, with their age."""
    return [
        f"{name}: checked {checked.isoformat()}, {(today - checked).days} days ago"
        for name, checked in snapshots.items()
        if today - checked > max_age
    ]


def test_snapshots_are_fresh() -> None:
    old = stale(SNAPSHOTS, datetime.now(UTC).date())
    if not old:
        return
    message = (
        f"Snapshots older than {MAX_AGE.days} days; review them against their sources "
        "(docs/development/releasing.md, 'Upkeep'): " + "; ".join(old)
    )
    if STRICT:
        pytest.fail(message)
    warnings.warn(message, stacklevel=1)


def test_no_snapshot_is_dated_in_the_future() -> None:
    tomorrow = datetime.now(UTC).date() + timedelta(days=1)
    assert [name for name, checked in SNAPSHOTS.items() if checked > tomorrow] == []


def test_stale_flags_only_old_snapshots() -> None:
    snapshots = {"old": date(2026, 1, 1), "new": date(2026, 3, 1)}
    assert stale(snapshots, date(2026, 4, 15)) == ["old: checked 2026-01-01, 104 days ago"]
    assert stale(snapshots, date(2026, 3, 31)) == []


def test_old_snapshot_warns_by_default_and_fails_when_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "SNAPSHOTS", {"ancient": date(2000, 1, 1)})
    monkeypatch.setattr(module, "STRICT", False)
    with pytest.warns(UserWarning, match="ancient: checked 2000-01-01"):
        test_snapshots_are_fresh()
    monkeypatch.setattr(module, "STRICT", True)
    with pytest.raises(pytest.fail.Exception, match="ancient"):
        test_snapshots_are_fresh()
