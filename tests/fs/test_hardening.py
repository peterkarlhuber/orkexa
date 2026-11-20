"""Regression tests for the independent review of orkexa #109 (findings H1-H2, M1-M5, L1-L7).

Each test fails on the reviewed head (7107840) and passes with the fix; the finding id is in its name or
docstring. Linux-, POSIX- or Windows-only tests say why in their skip reason.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import errno
import os
from pathlib import Path
import sys
import time

import pytest

from orkexa import fs
from orkexa.fs import MAX_DEPTH, FsError, SafeRoot, TextBudget
from orkexa.fs._names import Pattern, fold, home_parents
from orkexa.testing import pathguard
from orkexa.testing.pathguard import CANARY, build_layout, check_reader, swap_race

LINUX = sys.platform.startswith("linux")
WINDOWS = sys.platform == "win32"
POSIX = os.name == "posix"
FALLBACKS = os.environ.get("ORKEXA_TEST_FS_FALLBACKS") == "1" and POSIX


def _code(call: Callable[[], object]) -> str:
    with pytest.raises(FsError) as raised:
        call()
    return raised.value.code


@pytest.fixture
def home(tmp_path: Path) -> Path:
    folder = tmp_path / "home" / "ada"
    folder.mkdir(parents=True)
    return folder


# ---- H1 the checks are linear ------------------------------------------------------------------------------


def test_h1_pattern_matching_is_linear() -> None:
    """Several ``**`` used to backtrack exponentially (2,048 components took 68 s with the GIL held)."""
    started = time.monotonic()
    assert not Pattern("**/a/**/b/**/c/**/d").covers(("a",) * MAX_DEPTH)
    assert Pattern("**/a/**/b/**/c").covers(("x", "a", "y", "b", "c", "z"))
    assert not Pattern("**/a/**/b/**/c").matches(("a", "b"))
    assert time.monotonic() - started < 1
    with pytest.raises(ValueError):
        Pattern("/".join(["a"] * (MAX_DEPTH + 1)))


def test_h1_paths_have_a_depth_limit(tmp_path: Path) -> None:
    root = SafeRoot(tmp_path, exclude=["**/x/**/y/**/z"])
    started = time.monotonic()
    assert len(root.check("/".join(["a"] * MAX_DEPTH))) == MAX_DEPTH
    assert time.monotonic() - started < 1
    deep = "/".join(["a"] * (MAX_DEPTH + 1))
    assert _code(lambda: root.check(deep)) == "fs.invalid_path"
    assert _code(lambda: root.read_bytes(deep)) == "fs.invalid_path"


# ---- H2 the blocklist and the home folder apply to the root itself -----------------------------------------


def test_h2_a_blocked_folder_cannot_be_the_root(home: Path) -> None:
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_ed25519").write_bytes(CANARY)
    (home / "Library" / "Keychains" / "inner").mkdir(parents=True)
    for folder in (home / ".ssh", home / "Library" / "Keychains", home / "Library" / "Keychains" / "inner"):
        assert _code(lambda folder=folder: SafeRoot(folder, home=home)) == "fs.protected_root", folder
    assert SafeRoot(home / ".ssh", home=home, blocked=[]).read_bytes("id_ed25519") == CANARY  # replaceable


def test_h2_the_blocklist_continues_from_the_root(home: Path) -> None:
    """A root inside a folder that a blocked pattern starts in (``~/.config`` for ``**/.config/gh``)."""
    (home / ".config" / "gh").mkdir(parents=True)
    (home / ".config" / "gh" / "hosts.yml").write_bytes(CANARY)
    (home / ".config" / "app.toml").write_bytes(b"ok")
    (home / "Library" / "Keychains").mkdir(parents=True)
    (home / "Library" / "Keychains" / "login.keychain-db").write_bytes(CANARY)
    config = SafeRoot(home / ".config", home=home)
    assert _code(lambda: config.read_bytes("gh/hosts.yml")) == "fs.blocked"
    assert [entry.name for entry in config.list("")] == ["app.toml"]
    library = SafeRoot(home / "Library", home=home)
    assert _code(lambda: library.read_bytes("Keychains/login.keychain-db")) == "fs.blocked"
    assert library.list("") == []


def test_h2_home_and_the_folders_above_it(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (home / "projects").mkdir()
    monkeypatch.setenv("HOME", str(home.parent.parent))  # the process HOME is not the user's home here
    monkeypatch.setenv("USERPROFILE", str(home.parent.parent))
    for folder in (home, home.parent, home.parent.parent):
        assert _code(lambda folder=folder: SafeRoot(folder, home=home)) == "fs.protected_root", folder
    assert SafeRoot(home / "projects", home=home).list("") == []
    for parent in home_parents():
        if os.path.isdir(parent):
            assert _code(lambda parent=parent: SafeRoot(parent, home=home)) == "fs.protected_root", parent


# ---- M1 protected folders are looked up on every call ------------------------------------------------------


def test_m1_a_protected_folder_created_after_the_grant(home: Path) -> None:
    project = home / "project"
    project.mkdir()
    data = project / "nodestro-data"
    root = SafeRoot(project, protected=[data], home=home)
    assert _code(lambda: root.list("nodestro-data")) == "fs.blocked"  # missing: refused, not skipped
    data.mkdir()
    (data / "vault.db").write_bytes(CANARY)
    assert _code(lambda: root.read_bytes("nodestro-data/vault.db")) == "fs.blocked"
    assert _code(lambda: root.read_bytes("NODESTRO-DATA/vault.db")) == "fs.blocked"
    assert [entry.name for entry in root.list("")] == []


def test_m1_a_protected_folder_recreated_or_moved(home: Path) -> None:
    project = home / "project"
    data = project / "data"
    data.mkdir(parents=True)
    root = SafeRoot(project, protected=[data], home=home)
    os.rename(data, project / "old")
    data.mkdir()
    (data / "vault.db").write_bytes(CANARY)
    assert _code(lambda: root.read_bytes("data/vault.db")) == "fs.blocked"
    os.rename(data, project / "renamed")  # the recreated folder under another name: refused by identity
    assert _code(lambda: root.read_bytes("renamed/vault.db")) == "fs.blocked"
    assert _code(lambda: root.list("old")) == "fs.blocked"  # and so is the first one


# ---- M2 renaming never changes what may be read ------------------------------------------------------------


def test_m2_rename_keeps_include_and_exclude(home: Path) -> None:
    project = home / "project"
    (project / "public").mkdir(parents=True)
    (project / "drafts").mkdir()
    (project / "salary.txt").write_bytes(CANARY)
    (project / "drafts" / "secret.md").write_bytes(CANARY)
    (project / "drafts" / "other").mkdir()
    included = SafeRoot(project, include=["public/**"], writable=True, home=home)
    assert _code(lambda: included.rename("salary.txt", "public/salary.txt")) == "fs.excluded"
    assert (project / "salary.txt").exists()
    excluded = SafeRoot(project, exclude=["drafts/*.md"], writable=True, home=home)
    assert _code(lambda: excluded.rename("drafts", "notes")) == "fs.excluded"
    assert _code(lambda: excluded.rename("drafts/secret.md", "notes.md")) == "fs.excluded"
    assert (project / "drafts" / "secret.md").exists()
    if not FALLBACKS:  # a rename that changes nothing the patterns decide still works
        excluded.rename("drafts/other", "drafts/more")
        assert (project / "drafts" / "more").is_dir()
    blocked = SafeRoot(project, writable=True, home=home)
    (project / "keys").mkdir()
    assert _code(lambda: blocked.rename("keys", ".ssh")) == "fs.blocked"


def test_m2_rename_keeps_protected_folders_in_place(home: Path) -> None:
    project = home / "project"
    data = project / "app" / "data"
    data.mkdir(parents=True)
    root = SafeRoot(project, protected=[data], writable=True, home=home)
    assert _code(lambda: root.rename("app", "app2")) == "fs.blocked"
    assert _code(lambda: root.rename("app/data", "data2")) == "fs.blocked"
    assert data.is_dir()


# ---- M3 folding and the name the file system answered with -------------------------------------------------


def test_m3_fold_drops_ignorables_and_dotless_i(tmp_path: Path) -> None:
    assert fold(".s\u200bsh") == fold(".SSH") == ".ssh"
    assert fold("\u00adpr\u0131vate\ufe0f") == "private"
    folder = tmp_path / "root"
    folder.mkdir()
    root = SafeRoot(folder, blocked=["**/private", "**/.ssh"])
    for path in (".s\u200bsh/id", ".ss\u2060h/id", "pr\u0131vate/x", "PRIVATE\u200d/x"):
        assert _code(lambda path=path: root.read_bytes(path)) == "fs.blocked", ascii(path)


@pytest.mark.skipif(not POSIX, reason="Windows compares the final path of each handle (test_windows.py)")
def test_m3_the_stored_name_must_fold_to_the_name_asked_for(
    layout: pathguard.Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orkexa.fs import _posix

    root = SafeRoot(layout.root)
    monkeypatch.setattr(_posix, "_stored_name", lambda *args: "something-else")
    assert _code(lambda: root.read_bytes("sub/deep.txt")) == "fs.ambiguous_name"
    monkeypatch.setattr(_posix, "_stored_name", lambda dir_fd, fd, name, *rest: name.upper())
    assert root.read_bytes("sub/deep.txt") == b"deep"  # only the case differs: the same name


@pytest.mark.skipif(not LINUX, reason="the directory scan is how Linux checks case-insensitive file systems")
def test_m3_case_insensitive_file_systems_are_checked_by_scan(
    layout: pathguard.Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orkexa.fs import _posix

    monkeypatch.setattr(_posix, "_caseless_file_system", lambda fd: True)
    seen: list[str] = []
    real = _posix._scan_name

    def scan(dir_fd: int, name: str, info: object, shown: str) -> str | None:
        seen.append(name)
        return real(dir_fd, name, info, shown)  # type: ignore[arg-type]

    monkeypatch.setattr(_posix, "_scan_name", scan)
    assert SafeRoot(layout.root).read_bytes("sub/deep.txt") == b"deep"
    assert seen == ["sub", "deep.txt"]


@pytest.fixture
def layout(tmp_path: Path) -> pathguard.Layout:
    return build_layout(tmp_path)


# ---- M4 descriptors, depth and listing budgets -------------------------------------------------------------


@pytest.mark.skipif(not POSIX, reason="the descriptor limit (RLIMIT_NOFILE) is a POSIX limit")
def test_m4_a_deep_path_holds_two_descriptors(tmp_path: Path) -> None:
    import resource

    folder = tmp_path / "root"
    deep = folder.joinpath(*(["d"] * (MAX_DEPTH - 1)))
    deep.mkdir(parents=True)
    (deep / "f.txt").write_bytes(b"deep")
    root = SafeRoot(folder)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    in_use = len(os.listdir("/dev/fd" if os.path.isdir("/dev/fd") else "/proc/self/fd"))
    resource.setrlimit(resource.RLIMIT_NOFILE, (in_use + 16, hard))
    try:
        assert root.read_bytes("/".join(["d"] * (MAX_DEPTH - 1) + ["f.txt"])) == b"deep"
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_m4_a_listing_has_a_hard_entry_budget(tmp_path: Path) -> None:
    folder = tmp_path / "root"
    folder.mkdir()
    for number in range(300):
        (folder / f"x{number}.key").write_bytes(b"")
    root = SafeRoot(folder, exclude=["**/*.key"], max_entries=100)
    with pytest.raises(FsError) as raised:
        root.list("")
    assert (raised.value.code, raised.value.reason) == ("fs.too_many_entries", "entries")


def test_m4_a_listing_has_a_time_budget(tmp_path: Path) -> None:
    folder = tmp_path / "root"
    folder.mkdir()
    for number in range(50):
        (folder / f"x{number}.txt").write_bytes(b"")
    root = SafeRoot(folder, max_list_seconds=1e-9)
    with pytest.raises(FsError) as raised:
        root.list("")
    assert (raised.value.code, raised.value.reason) == ("fs.too_many_entries", "timeout")


def test_m4_the_name_fallback_scan_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "root"
    folder.mkdir()
    for number in range(30):
        (folder / f"x{number}.txt").write_bytes(b"")
    monkeypatch.setattr(fs._backend, "MAX_SCAN", 10)
    with pytest.raises(FsError) as raised:
        SafeRoot(folder).read_bytes("\u00fcber.txt")  # not ASCII and not found: the NFD fallback scans
    assert raised.value.code == "fs.too_many_entries"


# ---- M5 the PDF memory budget ------------------------------------------------------------------------------


def test_m5_no_pdf_without_a_memory_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "root"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(fs, "_memory_budget_enforced", lambda: False)
    error = _code(lambda: SafeRoot(folder).extract_text("a.pdf", budget=TextBudget(max_memory_mb=64)))
    assert error == "fs.unsupported_type"
    monkeypatch.undo()  # the real check: Linux, macOS and Windows enforce the budget
    assert fs._memory_budget_enforced() is (LINUX or sys.platform in ("darwin", "win32"))


# ---- L1 openat2 is probed once -----------------------------------------------------------------------------


@pytest.mark.skipif(not LINUX, reason="openat2 is a Linux system call")
def test_l1_an_openat2_error_never_switches_to_the_fallback(
    layout: pathguard.Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orkexa.fs import _posix

    if not _posix.openat2_available():
        pytest.skip("this kernel has no openat2")
    root = SafeRoot(layout.root)
    monkeypatch.setattr(_posix, "_call_openat2", lambda *args: -errno.EPERM)
    assert _code(lambda: root.read_bytes("inside.txt")) == "fs.permission"
    assert _posix._openat2_state["usable"] is True


@pytest.mark.skipif(not LINUX, reason="mount ids (/proc/self/fdinfo) are a Linux feature")
def test_l1_the_fallback_refuses_when_mounts_cannot_be_checked(
    layout: pathguard.Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orkexa.fs import _posix

    monkeypatch.setitem(_posix._openat2_state, "usable", False)
    assert SafeRoot(layout.root).read_bytes("inside.txt") == b"inside"
    monkeypatch.setattr(_posix, "_mount_id", lambda fd: None)
    assert _code(lambda: SafeRoot(layout.root).read_bytes("inside.txt")) == "fs.cross_device"


# ---- L2 the rename fallback leaves nothing behind ----------------------------------------------------------


@pytest.mark.skipif(not POSIX, reason="the link-and-unlink fallback is the POSIX rename without renameat2")
def test_l2_the_rename_fallback_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from orkexa.fs import _posix

    folder = tmp_path / "root"
    folder.mkdir()
    (folder / "a.txt").write_bytes(b"a")
    root = SafeRoot(folder, writable=True)
    monkeypatch.setattr(_posix, "_rename_call", lambda *args: None)
    real_unlink = os.unlink

    def unlink(name: str, *, dir_fd: int | None = None) -> None:
        if name == "a.txt":
            raise PermissionError(errno.EACCES, "denied", name)
        real_unlink(name, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", unlink)
    assert _code(lambda: root.rename("a.txt", "b.txt")) == "fs.permission"
    monkeypatch.setattr(os, "unlink", real_unlink)
    assert (folder / "a.txt").exists() and not (folder / "b.txt").exists()
    assert os.stat(folder / "a.txt").st_nlink == 1
    root.rename("a.txt", "b.txt")
    assert sorted(os.listdir(folder)) == ["b.txt"]


# ---- L3 special files are never opened ---------------------------------------------------------------------


@pytest.mark.skipif(not POSIX, reason="FIFOs are POSIX special files")
def test_l3_a_fifo_is_refused_before_it_is_opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from orkexa.fs import _posix

    folder = tmp_path / "root"
    folder.mkdir()
    os.mkfifo(folder / "pipe")
    opened: list[str] = []
    real = _posix._child

    def child(dir_fd: int, name: str, flags: int, shown: str) -> int:
        opened.append(name)
        return real(dir_fd, name, flags, shown)

    monkeypatch.setattr(_posix, "_child", child)
    root = SafeRoot(folder)
    assert _code(lambda: root.read_bytes("pipe")) == "fs.not_regular_file"
    assert _code(lambda: root.stat("pipe")) == "fs.not_regular_file"
    assert "pipe" not in opened


# ---- L4 the size of a file with several links --------------------------------------------------------------


def test_l4_stat_and_list_keep_the_hard_link_rule(layout: pathguard.Layout) -> None:
    assert layout.created["hard"]
    root = SafeRoot(layout.root)
    assert _code(lambda: root.stat("hard")) == "fs.hard_link"
    sizes = {entry.name: entry.size for entry in root.list("")}
    assert sizes["hard"] == 0 and sizes["inside.txt"] == len(b"inside")
    opted_in = SafeRoot(layout.root, allow_hard_links=True)
    assert opted_in.stat("hard").size == len(CANARY)
    assert {entry.name: entry.size for entry in opted_in.list("")}["hard"] == len(CANARY)


# ---- L5 revocation stops a listing and an extraction -------------------------------------------------------


def test_l5_revoke_stops_a_listing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "root"
    folder.mkdir()
    for number in range(20):
        (folder / f"{number}.txt").write_bytes(b"")
    root = SafeRoot(folder)
    real = fs._backend.listing

    def listing(*args: object) -> Iterator[tuple[str, str, int, int]]:
        for count, item in enumerate(real(*args)):  # type: ignore[arg-type]
            if count == 5:
                root.revoke()
            yield item

    monkeypatch.setattr(fs._backend, "listing", listing)
    assert _code(lambda: root.list("")) == "fs.revoked"


def test_l5_revoke_stops_an_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "root"
    folder.mkdir()
    (folder / "notes.md").write_bytes(b"text")
    root = SafeRoot(folder)
    real = fs._decode

    def decode(data: bytes, shown: str) -> str:
        root.revoke()  # revoked while the text is being extracted
        return real(data, shown)

    monkeypatch.setattr(fs, "_decode", decode)
    assert _code(lambda: root.extract_text("notes.md")) == "fs.revoked"


# ---- L6 the representation hides the path ------------------------------------------------------------------


def test_l6_repr_does_not_show_the_path(home: Path) -> None:
    (home / "Private Project").mkdir()
    root = SafeRoot(home / "Private Project", home=home)
    assert "Private" not in repr(root) and str(home) not in repr(root)
    assert root.path.endswith("Private Project")


# ---- L7 the catalog helpers fail weak readers --------------------------------------------------------------


def test_l7_check_reader_needs_a_positive_control_and_real_refusals(layout: pathguard.Layout) -> None:
    def refuses_everything(path: str | bytes) -> bytes:
        raise FsError("fs.blocked", "no")

    assert {finding.case for finding in check_reader(refuses_everything, layout)} >= {"control"}

    root = SafeRoot(layout.root)

    def crashes(path: str | bytes) -> bytes:
        if path in ("inside.txt", "sub/deep.txt"):
            return root.read_bytes(path)
        raise ValueError("boom")

    problems = {finding.problem for finding in check_reader(crashes, layout)}
    assert "failed with ValueError, not a refusal" in problems
    assert check_reader(root.read_bytes, layout) == []


def test_l7_swap_race_reports_a_race_that_did_not_run(layout: pathguard.Layout) -> None:
    root = SafeRoot(layout.root)
    findings = swap_race(root.read_bytes, layout, iterations=50, min_swaps=10**9, timeout=0.5)
    assert [finding.problem for finding in findings] == [findings[0].problem]
    assert "swaps happened" in findings[0].problem
