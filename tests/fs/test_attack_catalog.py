"""The path-guard attack catalog (orkexa.testing.pathguard) against SafeRoot, on every operating system.

Every name attack of the catalog runs on Linux, macOS and Windows: the name rules are the same everywhere. The
link attacks run where the layout could create the link (``Layout.created``); on the CI runners that is every
link of the platform (symbolic links on all three, junctions on Windows).
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from orkexa.fs import FsError, SafeRoot
from orkexa.testing import pathguard
from orkexa.testing.pathguard import CANARY, CATALOG, NAME_ATTACKS, build_layout, check_reader, link_attacks

HERE = Path(__file__).resolve().parent


def _code(error: BaseException) -> str | None:
    return error.code if isinstance(error, FsError) else None


@pytest.fixture
def layout(tmp_path: Path) -> pathguard.Layout:
    return build_layout(tmp_path)


def test_the_catalog_has_every_p0_2_case_and_every_spike_gap() -> None:
    ids = [case.id for case in CATALOG]
    assert ids == [f"N10-{number:02d}" for number in range(1, 24)] + [
        f"W06-{number}" for number in range(1, 6)
    ]
    for case in CATALOG:
        assert case.platforms <= {"linux", "darwin", "win32"} and case.platforms
        assert case.tests, case.id


def test_every_test_the_catalog_names_exists() -> None:
    for case in CATALOG:
        for test in case.tests:
            module, name = test.split("::")
            source = (HERE / module).read_text(encoding="utf-8")
            assert re.search(rf"^def {re.escape(name)}\b", source, re.MULTILINE), (case.id, test)


def test_every_name_attack_belongs_to_a_case_and_expects_its_codes() -> None:
    cases = {case.id: case for case in CATALOG}
    for case, _, codes in NAME_ATTACKS:
        assert codes <= cases[case].codes, case


@pytest.mark.parametrize(
    ("case", "path", "codes"),
    NAME_ATTACKS,
    ids=[f"{case}-{index}" for index, (case, _, _) in enumerate(NAME_ATTACKS)],
)
def test_name_attacks_are_refused(
    layout: pathguard.Layout, case: str, path: str | bytes, codes: frozenset[str]
) -> None:
    root = SafeRoot(layout.root)
    with pytest.raises(FsError) as raised:
        root.read_bytes(path)
    assert raised.value.code in codes, (case, path, raised.value.code)
    assert str(layout.root) not in str(raised.value)  # messages never carry an absolute path


def test_link_attacks_are_refused(layout: pathguard.Layout) -> None:
    attacks = link_attacks(layout)
    expected = {"link_file", "link_dir", "link_rel", "chain", "hard"} | (
        {"junction"} if pathguard.sys.platform == "win32" else set()
    )
    assert {
        name for name, made in layout.created.items() if made
    } >= expected  # the runners can make them all
    root = SafeRoot(layout.root)
    for case, path, codes in attacks:
        with pytest.raises(FsError) as raised:
            root.read_bytes(path)
        assert raised.value.code in codes, (case, path, raised.value.code)


def test_check_reader_passes_safe_root(layout: pathguard.Layout) -> None:
    root = SafeRoot(layout.root)
    assert check_reader(root.read_bytes, layout, code_of=_code) == []


def test_check_reader_catches_a_naive_reader(layout: pathguard.Layout) -> None:
    """The helper is only useful if it fails a reader that follows links and joins paths."""

    def naive(path: str | bytes) -> bytes:
        text = path.decode("utf-8", "replace") if isinstance(path, bytes) else path
        return (layout.root / text).read_bytes()

    findings = check_reader(naive, layout)
    problems = {finding.problem for finding in findings}
    assert "read the canary outside the root" in problems
    assert "read a blocked file" in problems
    assert any(finding.case == "N10-03" for finding in findings)


def test_swap_race_catches_a_naive_reader(layout: pathguard.Layout) -> None:
    def naive(path: str) -> bytes:
        return (layout.root / path).read_bytes()

    assert layout.created["link_file"]
    findings = pathguard.swap_race(naive, layout, iterations=10_000)
    assert [finding.case for finding in findings] == ["N10-04"]


def test_the_canary_is_never_in_a_listing(layout: pathguard.Layout) -> None:
    root = SafeRoot(layout.root)
    names = {entry.name: entry.kind for entry in root.list("")}
    assert ".ssh" not in names and "_netrc" not in names  # blocked: not listed
    assert [entry.name for entry in root.list("Library")] == []  # Library/Keychains is blocked
    for link in ("link_file", "link_dir", "link_rel", "chain1", "junction"):
        if link in names:
            assert names[link] == "link"
    for entry in root.list("sub"):
        assert entry.name != ".ssh"
    assert CANARY not in b"".join(
        root.read_bytes(entry.path)
        for entry in root.list("")
        if entry.kind == "file" and entry.name != "hard"
    )
