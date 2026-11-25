"""The folder grant conformance suite (T-CONF-9, Nodestro N-63) against the reference implementation on
SafeRoot, on every operating system the CI runs (the offline jobs run it on macOS and Windows)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from orkexa.testing.conformance import FOLDER_CASES, Case, Request, run_cases

from .adapter import run_and_check
from .factories import FACTORIES
from .folder_access import SafeRootFolders, _search


# MUP-ONA T-CONF-9
def test_the_safe_root_folders_conform(tmp_path: Path) -> None:
    factory, expect = FACTORIES["safe-root-folders"]
    report = run_and_check("safe-root-folders", factory, FOLDER_CASES, expect, tmp_path)
    assert len(report.passed) == len(FOLDER_CASES)


class _Leaky(SafeRootFolders):
    """Ignores exclude patterns and revocations: the suite must catch both."""

    def grant(self, path: str, **kwargs: object) -> str:  # type: ignore[override]
        kwargs.pop("exclude", None)
        return super().grant(path, **kwargs)  # type: ignore[arg-type]

    def revoke(self, grant: str) -> None:
        pass


def test_the_suite_catches_a_leaky_implementation(tmp_path: Path) -> None:
    report = run_cases(
        lambda request: _Leaky(request.workdir / "data"), FOLDER_CASES, expect=(), workdir=tmp_path
    )
    assert {"folders.exclude_patterns", "folders.revoked"} <= set(report.failed)
    assert "folders.list_read_search" in report.passed


class _DenyOnRead(SafeRootFolders):
    """Checks a deny only when a file is read, as Nodestro's folder tools did before the N-10 review (H1)."""

    def list(self, grant: str, path: str = "") -> list[str]:
        return self._logged(grant, "list", lambda root: [entry.path for entry in root.list(path)])  # type: ignore[return-value]

    def search(self, grant: str, text: str) -> list[str]:
        return self._logged(grant, "search", lambda root: _search(root, text.casefold()))  # type: ignore[return-value]


class _Literal(SafeRootFolders):
    """Compares a deny with the path as spelled (no canonical form, with case)."""

    def deny(self, path: str) -> None:
        self.denies.append(tuple(path.split(os.sep)))

    def _denied(self, target: str) -> bool:
        parts = tuple(target.split(os.sep))
        return any(parts[: len(deny)] == deny for deny in self.denies)


class _Everyone(SafeRootFolders):
    """Ignores the grant's agents and mode."""

    def grant(self, path: str, **kwargs: object) -> str:  # type: ignore[override]
        kwargs.pop("agents", None)
        kwargs["unattended"] = True
        return super().grant(path, **kwargs)  # type: ignore[arg-type]


def test_the_suite_catches_denies_scopes_and_modes_that_do_not_hold(tmp_path: Path) -> None:
    cases = [
        case
        for case in FOLDER_CASES
        if case.name in {"folders.denies", "folders.deny_spellings", "folders.scope_and_mode"}
    ]

    def run(kind: type[SafeRootFolders], name: str) -> set[str]:
        (tmp_path / name).mkdir()
        report = run_cases(
            lambda request: kind(request.workdir / "data"), cases, expect=(), workdir=tmp_path / name
        )
        return set(report.failed)

    assert {"folders.denies", "folders.deny_spellings"} <= run(_DenyOnRead, "on-read")
    # On Windows (8.3 names) and macOS (/var is /private/var) the temporary folder's spelling differs from
    # its canonical path, so a literal store fails folders.denies there as well.
    literal = run(_Literal, "literal")
    assert "folders.deny_spellings" in literal and literal <= {"folders.denies", "folders.deny_spellings"}
    assert run(_Everyone, "everyone") == {"folders.scope_and_mode"}


def test_a_backend_that_is_no_folder_access_is_reported(tmp_path: Path) -> None:
    report = run_cases(lambda request: object(), FOLDER_CASES[:1], expect=(), workdir=tmp_path)
    assert any("does not implement the contract of the folders cases" in error for error in report.errors)


@pytest.mark.parametrize("case", FOLDER_CASES, ids=[case.name for case in FOLDER_CASES])
def test_every_case_is_documented_and_marked(case: Case) -> None:
    assert case.contract == "folders"
    assert case.description
    assert not case.needs


def test_a_store_that_protects_nothing_fails(tmp_path: Path) -> None:
    """A grant store that protects another folder than the data folder fails the data folder cases."""

    def unprotected(request: Request) -> SafeRootFolders:
        found = SafeRootFolders(request.workdir / "data")
        found.data = request.workdir / "elsewhere"
        return found

    report = run_cases(unprotected, FOLDER_CASES, expect=(), workdir=tmp_path)
    assert {"folders.data_folder_not_grantable", "folders.data_folder_inside_a_grant"} <= set(report.failed)
