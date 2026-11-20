"""Regression tests for the re-review of orkexa #109 (findings N1-N7, merged as 6fdea4e).

Each test of a fix fails on 6fdea4e and passes with it; the finding id is in its name. N4 is documented, not
changed: its test pins the behavior the guide describes. Linux-only tests say why in their skip reason.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

from orkexa import _pdf_launch, fs
from orkexa.fs import FsError, SafeRoot, TextBudget, root_problem
from orkexa.retrieval import RetrievalError, pdf_pages_isolated

LINUX = sys.platform.startswith("linux")
WINDOWS = sys.platform == "win32"
#: An empty folder with the casefold attribute on ext4 (CI job fs-fallbacks mounts one under sudo).
CASEFOLD_DIR = os.environ.get("ORKEXA_TEST_CASEFOLD_DIR", "")


def _code(call: Callable[[], object]) -> str:
    with pytest.raises(FsError) as raised:
        call()
    return raised.value.code


def _link(target: Path, link: Path) -> None:
    """A folder link: a junction on Windows (no privilege needed), a symbolic link elsewhere."""
    if WINDOWS:
        import _winapi

        _winapi.CreateJunction(str(target), str(link))  # type: ignore[attr-defined,unused-ignore]
    else:
        os.symlink(target, link, target_is_directory=True)


# ---- N1 every user's home folder ---------------------------------------------------------------------------


def test_n1_every_users_home_folder_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    homes = tmp_path / "homes"
    ada, bob = homes / "ada", homes / "bob"
    (bob / "Documents").mkdir(parents=True)
    ada.mkdir()
    monkeypatch.setattr(fs, "home_parents", lambda: [str(homes)])
    assert _code(lambda: SafeRoot(bob, home=ada)) == "fs.protected_root"  # not the process user's home
    assert _code(lambda: SafeRoot(ada, home=tmp_path / "service")) == "fs.protected_root"
    assert _code(lambda: SafeRoot(homes, home=ada)) == "fs.protected_root"
    assert SafeRoot(bob / "Documents", home=ada).path.endswith("Documents")  # inside a home: fine
    assert root_problem(str(bob), home=ada) is not None


# ---- N2 blocked names of the root as given -----------------------------------------------------------------


def test_n2_a_blocked_folder_given_through_its_link_is_refused(tmp_path: Path) -> None:
    home = tmp_path / "home" / "ada"
    keys = tmp_path / "vault" / "keys"
    (keys / "sub").mkdir(parents=True)
    home.mkdir(parents=True)
    (keys / "id_ed25519").write_bytes(b"KEY")
    _link(keys, home / ".ssh")  # ~/.ssh kept elsewhere (dotfile managers, encrypted volumes)
    _link(home / ".ssh", home / "keys")  # a harmless name that leads through ~/.ssh
    for given in (home / ".ssh", home / ".ssh" / "sub", home / "keys"):
        assert _code(lambda given=given: SafeRoot(given, home=home)) == "fs.protected_root", given
    assert root_problem(os.path.realpath(keys), home=home, given=str(home / ".ssh")) is not None


# ---- N3 renaming onto a protected folder that does not exist yet ------------------------------------------


def test_n3_nothing_is_renamed_to_where_a_protected_folder_will_be(tmp_path: Path) -> None:
    folder = tmp_path / "root"
    (folder / "mine" / "data").mkdir(parents=True)
    (folder / "mine" / "data" / "config.json").write_bytes(b'{"evil": true}')
    (folder / "a.txt").write_bytes(b"a")
    root = SafeRoot(folder, protected=[folder / "app" / "data"], writable=True)  # not created yet
    assert _code(lambda: root.rename("mine", "app")) == "fs.blocked"
    assert _code(lambda: root.rename("mine/data", "app")) == "fs.blocked"
    assert _code(lambda: root.rename("a.txt", "app")) == "fs.blocked"
    assert not (folder / "app").exists()
    root.rename("a.txt", "b.txt")  # a rename elsewhere still works
    assert (folder / "b.txt").read_bytes() == b"a"


# ---- N4 protected folders known at the grant ---------------------------------------------------------------


def test_n4_a_protected_folder_that_exists_at_the_grant_stays_protected_when_moved(tmp_path: Path) -> None:
    """Documented, not changed: identities are learned at the grant and on every call, and never forgotten.

    A protected folder created after the grant and moved within the root before any call has seen it is the
    remaining case (docs/guides/path-guard.md, "Remaining risks"); this test pins the part that holds.
    """
    folder = tmp_path / "root"
    data = folder / "data"
    data.mkdir(parents=True)
    (data / "v").write_bytes(b"vault")
    root = SafeRoot(folder, protected=[data])
    os.rename(data, folder / "elsewhere")  # before any call
    assert _code(lambda: root.read_bytes("elsewhere/v")) == "fs.blocked"


# ---- N5 the macOS memory meter -----------------------------------------------------------------------------


def test_n5_the_macos_meter_is_the_physical_footprint() -> None:
    meter = _pdf_launch._memory_meter()
    if sys.platform == "darwin":
        assert meter is _pdf_launch._footprint_darwin
        assert meter(os.getpid()) > 1 << 20
    else:
        assert meter is None  # an operating-system limit enforces the budget (RLIMIT_AS, job object)


def test_n5_no_pdf_when_memory_cannot_be_measured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_pdf_launch, "_memory_meter", lambda: lambda pid: None)
    with pytest.raises(RetrievalError) as raised:
        pdf_pages_isolated(b"%PDF-1.4\n", max_memory_mb=64)
    assert raised.value.reason == "unenforced"
    folder = tmp_path / "root"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(b"%PDF-1.4\n")
    error = _code(lambda: SafeRoot(folder).extract_text("a.pdf", budget=TextBudget(max_memory_mb=64)))
    assert error == "fs.unsupported_type"


def test_n5_a_worker_whose_memory_cannot_be_measured_is_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    me = os.getpid()
    monkeypatch.setattr(_pdf_launch, "_memory_meter", lambda: lambda pid: 1 if pid == me else None)
    with pytest.raises(RetrievalError) as raised:
        pdf_pages_isolated(b"%PDF-1.4\n", max_memory_mb=64, timeout=20)
    assert raised.value.reason == "unenforced"


def test_n5_the_watcher_stops_a_worker_over_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    me = os.getpid()
    monkeypatch.setattr(_pdf_launch, "_memory_meter", lambda: lambda pid: 1 if pid == me else 1 << 40)
    with pytest.raises(RetrievalError) as raised:
        pdf_pages_isolated(b"%PDF-1.4\n", max_memory_mb=64, timeout=20)
    assert raised.value.reason == "memory"


# ---- N6 the Windows job object -----------------------------------------------------------------------------


class _FailingJob:
    def __init__(self, limit: int) -> None:
        raise OSError(5, "Access is denied")


class _UnassignableJob:
    def __init__(self, limit: int) -> None:
        self.closed = False

    def assign(self, handle: int) -> None:
        raise OSError(5, "Access is denied")

    def peak(self) -> int:
        return 0

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("job", [_FailingJob, _UnassignableJob])
def test_n6_a_job_object_that_cannot_be_used_is_a_retrieval_error(
    job: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[object] = []
    real_popen = subprocess.Popen

    def popen(*args: object, **kwargs: object) -> object:
        process = real_popen(*args, **kwargs)  # type: ignore[call-overload]
        started.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(_pdf_launch, "_WindowsJob", job)
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RetrievalError) as raised:
        pdf_pages_isolated(b"%PDF-1.4\n", max_memory_mb=64)
    monkeypatch.undo()
    assert raised.value.reason == "unenforced"
    assert "job object" in str(raised.value)
    assert all(process.poll() is not None for process in started)  # type: ignore[attr-defined]


def test_n6_html_without_a_memory_budget_is_a_fetch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from orkexa.web.fetch import FetchError, html_to_text_isolated

    monkeypatch.setattr(_pdf_launch, "_memory_meter", lambda: lambda pid: None)
    with pytest.raises(FetchError) as raised:
        html_to_text_isolated("<p>hello</p>")
    assert "memory budget" in str(raised.value)


# ---- N7 special files, unknown file systems, a real casefold folder ----------------------------------------


@pytest.mark.skipif(not LINUX, reason="O_PATH and /proc/self/fd are Linux features")
def test_n7_a_special_file_swapped_in_after_the_check_is_never_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orkexa.fs import _posix

    folder = tmp_path / "root"
    folder.mkdir()
    os.mkfifo(folder / "pipe")
    (folder / "plain.txt").write_bytes(b"plain")
    regular = os.stat(folder / "plain.txt")
    monkeypatch.setitem(getattr(_posix, "_reopen_state", {}), "usable", True)  # also in fs-fallbacks
    # The swap: the check looked at a regular file, the name is a FIFO when it is opened.
    monkeypatch.setattr(_posix, "_precheck", lambda dir_fd, name, shown: regular)
    opened: list[str] = []
    real = _posix._child

    def child(dir_fd: int, name: str, flags: int, shown: str) -> int:
        if not flags & os.O_PATH:  # an O_PATH descriptor opens nothing
            opened.append(name)
        return real(dir_fd, name, flags, shown)

    monkeypatch.setattr(_posix, "_child", child)
    root = SafeRoot(folder)
    assert _code(lambda: root.read_bytes("pipe")) == "fs.not_regular_file"
    assert _code(lambda: root.stat("pipe")) == "fs.not_regular_file"
    assert "pipe" not in opened
    assert root.read_bytes("plain.txt") == b"plain"


def test_n7_file_systems_not_known_to_be_case_sensitive_are_checked() -> None:
    from orkexa.fs._names import caseless_file_system_type

    for f_type in (0x01021997, 0x9660, 0x65735546, 0x2FC12FC1, 0x6969, 0x4D44, 0x12345678):
        assert caseless_file_system_type(f_type), hex(f_type)  # 9p, ISO 9660, FUSE/virtiofs, ZFS, NFS, FAT
    for f_type in (0xEF53, 0x9123683E, 0x58465342, 0x01021994):
        assert not caseless_file_system_type(f_type), hex(f_type)  # ext4, Btrfs, XFS, tmpfs


@pytest.mark.skipif(
    not CASEFOLD_DIR, reason="needs an ext4 casefold folder (ORKEXA_TEST_CASEFOLD_DIR, CI job fs-fallbacks)"
)
def test_m3_n7_a_real_casefold_folder_is_checked_by_its_stored_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from orkexa.fs import _posix

    folder = Path(CASEFOLD_DIR) / f"t{uuid.uuid4().hex}"
    folder.mkdir()  # inherits the casefold attribute
    fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert _posix._casefold_folder(fd), "ORKEXA_TEST_CASEFOLD_DIR is not a casefold folder"
    finally:
        os.close(fd)
    (folder / "Report.TXT").write_bytes(b"report")
    (folder / "Sub").mkdir()
    (folder / "Sub" / "Deep.md").write_bytes(b"deep")
    answered: list[str | None] = []
    real = _posix._scan_name

    def scan(dir_fd: int, name: str, info: object, shown: str) -> str | None:
        answered.append(real(dir_fd, name, info, shown))  # type: ignore[arg-type]
        return answered[-1]

    monkeypatch.setattr(_posix, "_scan_name", scan)
    root = SafeRoot(folder)
    assert root.read_bytes("report.txt") == b"report"  # the kernel matched without case ...
    assert answered == ["Report.TXT"]  # ... and the stored name was read back from the disk
    assert root.read_bytes("SUB/deep.MD") == b"deep"
    assert answered[1:] == ["Sub", "Deep.md"]
    # A fold that disagrees with the file system's is caught: the name stored is not the name asked for.
    monkeypatch.setattr(_posix, "fold", lambda text: text)
    assert _code(lambda: root.read_bytes("report.txt")) == "fs.ambiguous_name"
