"""The folder browser (:func:`orkexa.fs.browse_folder`, Nodestro plan N-11): folder names only, no system
folders, no other user's home, no blocked or protected folder, and links only to folders that pass the same
checks.

A test that only makes sense on one operating system says why in its skip reason.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from orkexa import fs
from orkexa.fs import FsError, browse_folder, browse_problem, system_folders
from orkexa.testing.pathguard import make_junction

WINDOWS = sys.platform == "win32"
POSIX = os.name == "posix"


def _code(call: object) -> str:
    assert callable(call)
    with pytest.raises(FsError) as raised:
        call()
    return raised.value.code


def _link(link: Path, target: Path) -> None:
    """A symbolic link to a folder, or on Windows a junction when symbolic links need a privilege."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if not make_junction(link, target):
            pytest.skip("neither a symbolic link nor a junction can be made here")


def _names(path: Path | str, **options: object) -> list[str]:
    return [folder.name for folder in browse_folder(path, **options).folders]  # type: ignore[arg-type]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "tree"
    for name in ("Beta", "alpha", "gamma/inner", ".hidden", "with space"):
        (root / name).mkdir(parents=True)
    (root / "notes.txt").write_text("canary-file-content", encoding="utf-8")
    (root / "alpha" / "secret.pdf").write_bytes(b"%PDF-1.4")
    return root


def test_only_folder_names_are_answered_never_files(tree: Path) -> None:
    listing = browse_folder(tree)
    assert listing.path == os.path.realpath(tree) or Path(listing.path).samefile(tree)
    assert [folder.name for folder in listing.folders] == ["alpha", "Beta", "gamma", "with space"]
    assert all(Path(folder.path).is_dir() and not folder.link for folder in listing.folders)
    assert Path(listing.folders[0].path).samefile(tree / "alpha")
    assert listing.parent is not None and Path(listing.parent).samefile(tree.parent)
    assert not listing.truncated
    shown = repr(listing)
    assert "notes.txt" not in shown and "secret.pdf" not in shown and "canary" not in shown
    assert _names(tree / "alpha") == []


def test_a_file_a_missing_folder_and_bad_paths_are_refused(tree: Path) -> None:
    assert _code(lambda: browse_folder(tree / "notes.txt")) == "fs.not_a_directory"
    assert _code(lambda: browse_folder(tree / "missing")) == "fs.not_found"
    assert _code(lambda: browse_folder("relative/folder")) == "fs.invalid_path"
    assert _code(lambda: browse_folder(f"{tree}\x00")) == "fs.invalid_path"
    assert _code(lambda: browse_folder(f" {tree}")) == "fs.invalid_path"
    assert _code(lambda: browse_folder("\\\\server\\share\\folder")) == "fs.network_path"
    assert _code(lambda: browse_folder("//server/share/folder")) == "fs.network_path"
    with pytest.raises(ValueError):
        browse_folder(tree, max_folders=0)


def test_the_root_of_the_file_system_is_browsable_but_shows_no_system_folder(tmp_path: Path) -> None:
    root = os.path.abspath(os.sep)
    # A home folder elsewhere: the administrator's own /root would be shown to the administrator.
    listing = browse_folder(root, home=tmp_path)
    assert listing.parent is None
    shown = {fs.fold(folder.path) for folder in listing.folders}
    for system in system_folders():
        if Path(system).parent == Path(root):
            assert fs.fold(system) not in shown, system
    if POSIX:
        names = {folder.name for folder in listing.folders}
        assert not names & {"proc", "sys", "dev", "etc", "usr", "bin", "sbin", "boot", "System", "Library"}


@pytest.mark.parametrize("name", ["proc", "sys", "dev", "etc", "usr", "boot"])
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the Linux system folders")
def test_linux_system_folders_cannot_be_browsed(name: str) -> None:
    folder = Path("/") / name
    if not folder.is_dir():
        pytest.skip(f"/{name} does not exist here")
    assert _code(lambda: browse_folder(folder)) == "fs.not_browsable"
    inner = next((item for item in folder.iterdir() if item.is_dir() and not item.is_symlink()), None)
    if inner is not None:
        assert _code(lambda: browse_folder(inner)) == "fs.not_browsable"


@pytest.mark.skipif(sys.platform != "darwin", reason="the macOS system folders")
def test_macos_system_folders_cannot_be_browsed() -> None:
    for folder in ("/System", "/Library", "/private/etc", "/etc", "/usr"):
        assert _code(lambda: browse_folder(folder)) == "fs.not_browsable", folder  # noqa: B023
    # /tmp is a link to /private/tmp, which is allowed; /etc and /var are links into /private, which is not.
    names = {folder.name: folder for folder in browse_folder("/").folders}
    assert "etc" not in names and "var" not in names
    if "tmp" in names:
        assert names["tmp"].link and fs.fold(names["tmp"].path) == fs.fold("/private/tmp")


@pytest.mark.skipif(not WINDOWS, reason="the Windows system folders")
def test_windows_system_folders_cannot_be_browsed() -> None:
    windows = os.environ.get("SYSTEMROOT", "C:\\Windows")
    assert _code(lambda: browse_folder(windows)) == "fs.not_browsable"
    assert _code(lambda: browse_folder(os.path.join(windows, "System32"))) == "fs.not_browsable"
    program_files = os.environ.get("PROGRAMFILES", "C:\\Program Files")
    assert _code(lambda: browse_folder(program_files)) == "fs.not_browsable"
    drive = os.path.splitdrive(windows)[0] + "\\"
    names = {fs.fold(folder.name) for folder in browse_folder(drive).folders}
    assert not names & {"windows", "program files", "program files (x86)", "programdata"}
    assert not any(name.startswith("$") for name in names)


def test_parent_references_are_resolved_before_the_check(tree: Path) -> None:
    """``..`` is resolved first: it cannot walk from a browsable folder into a system folder."""
    assert Path(browse_folder(f"{tree}{os.sep}gamma{os.sep}..").path).samefile(tree)
    system = system_folders()[0]
    if not Path(system).is_dir():
        pytest.skip("the first system folder does not exist here")
    climb = os.sep.join([str(tree), *([".."] * len(tree.parts)), *Path(system).parts[1:]])
    assert _code(lambda: browse_folder(climb)) == "fs.not_browsable"


def test_a_link_to_a_browsable_folder_is_shown_as_its_target(tree: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _link(tree / "shortcut", elsewhere)
    listing = browse_folder(tree)
    [shortcut] = [folder for folder in listing.folders if folder.name == "shortcut"]
    assert shortcut.link and Path(shortcut.path).samefile(elsewhere)
    assert fs.fold(shortcut.path) != fs.fold(str(tree / "shortcut"))  # the target, never the link


def test_a_link_into_a_system_folder_is_not_shown_and_not_followed(tree: Path) -> None:
    system = next((folder for folder in system_folders() if Path(folder).is_dir()), None)
    if system is None:
        pytest.skip("no system folder exists here")
    _link(tree / "escape", Path(system))
    assert "escape" not in _names(tree)
    assert _code(lambda: browse_folder(tree / "escape")) == "fs.not_browsable"


def test_a_link_to_a_file_or_a_dangling_link_is_not_shown(tree: Path) -> None:
    try:
        os.symlink(tree / "notes.txt", tree / "file-link")
        os.symlink(tree / "missing", tree / "dangling", target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links cannot be made here")
    assert {"file-link", "dangling"}.isdisjoint(_names(tree))


def test_blocked_folders_are_neither_listed_nor_browsable(tree: Path, tmp_path: Path) -> None:
    ssh = tree / ".ssh"
    ssh.mkdir()
    keys = tree / "Keys"
    _link(keys, ssh)
    assert "Keys" not in _names(tree)  # a link to a blocked folder
    assert _code(lambda: browse_folder(ssh)) == "fs.not_browsable"
    assert _code(lambda: browse_folder(keys)) == "fs.not_browsable"
    (ssh / "inner").mkdir()
    assert _code(lambda: browse_folder(ssh / "inner")) == "fs.not_browsable"
    # A blocked name that is itself a link elsewhere: every spelling counts (as for a root).
    vault = tmp_path / "vault"
    vault.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    _link(other / ".aws", vault)
    assert _code(lambda: browse_folder(other / ".aws")) == "fs.not_browsable"
    assert browse_problem(str(vault), given=str(other / ".aws")) is not None
    assert browse_problem(str(vault)) is None


def test_protected_folders_are_neither_listed_nor_browsable(tree: Path) -> None:
    data = tree / "gamma"
    _link(tree / "data-link", data)
    assert {"gamma", "data-link"}.isdisjoint(_names(tree, protected=[data]))
    assert _code(lambda: browse_folder(data, protected=[data])) == "fs.not_browsable"
    assert _code(lambda: browse_folder(data / "inner", protected=[data])) == "fs.not_browsable"
    assert _code(lambda: browse_folder(tree / "data-link", protected=[data])) == "fs.not_browsable"
    assert "gamma" in _names(tree)


def test_other_users_home_folders_are_not_shown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    users = tmp_path / "users"
    for name in ("ada", "bob", "carol/Documents"):
        (users / name).mkdir(parents=True)
    (users / "ada" / "Documents").mkdir()
    monkeypatch.setattr(fs, "home_parents", lambda: [str(users)])
    home = users / "ada"
    assert _names(users, home=home) == ["ada"]
    assert _names(home, home=home) == ["Documents"]
    assert _code(lambda: browse_folder(users / "bob", home=home)) == "fs.not_browsable"
    assert _code(lambda: browse_folder(users / "carol" / "Documents", home=home)) == "fs.not_browsable"
    # A link to another user's folder is refused by the identity of the folder it leads to.
    _link(home / "Documents" / "bobs", users / "bob")
    assert "bobs" not in _names(home / "Documents", home=home)
    assert _code(lambda: browse_folder(home / "Documents" / "bobs", home=home)) == "fs.not_browsable"


@pytest.mark.skipif(not POSIX, reason="the administrator's home folder /root is a POSIX system folder")
def test_the_own_home_folder_is_browsable_even_inside_a_system_folder(tmp_path: Path) -> None:
    """A home folder in a system folder (/root, a service account in /var/lib) is the user's own; one that is
    a file-system root or holds a system folder opens nothing more."""
    assert fs._home_open("/root")
    assert fs._home_open("/var/lib/service")
    assert not fs._home_open("/")
    assert not fs._home_open("/var")
    assert not fs._home_open("/usr")
    assert not fs._home_open(None)
    # An account whose home lies in a folder of programs, libraries, configuration or the kernel (Debian's
    # daemon in /usr/sbin, bin in /bin) gets no browsable home (N-11 review #2).
    closed_homes = ("/usr/sbin", "/bin", "/sbin", "/usr/lib/svc", "/lib/x", "/etc/svc", "/dev/x", "/proc/1")
    for closed in closed_homes:
        assert not fs._home_open(closed), closed
    for name in ("sbin", "bin"):
        for prefix in ("/usr/", "/"):
            folder = Path(prefix + name)
            if folder.is_dir():
                assert _code(lambda folder=folder: browse_folder(folder, home=folder)) == "fs.not_browsable"
    etc = Path("/etc")
    if etc.is_dir():
        assert _code(lambda: browse_folder(etc, home="/")) == "fs.not_browsable"


def test_hidden_and_unusable_names_are_left_out(tree: Path) -> None:
    assert ".hidden" not in _names(tree)
    if POSIX:
        (tree / "bad\x01name").mkdir()
        (tree / "$dollar").mkdir()
        names = _names(tree)
        assert "bad\x01name" not in names
        assert "$dollar" in names  # only Windows hides $ names ($Recycle.Bin, $WinREAgent)


@pytest.mark.skipif(not WINDOWS, reason="Windows' hidden and system attributes")
def test_windows_hidden_folders_and_dollar_names_are_left_out(tree: Path) -> None:
    import subprocess

    (tree / "$Recycle").mkdir()
    (tree / "Secretive").mkdir()
    subprocess.run(["attrib", "+h", str(tree / "Secretive")], check=True, capture_output=True)
    assert {"$Recycle", "Secretive"}.isdisjoint(_names(tree))


def test_limits_truncate_the_answer(tree: Path) -> None:
    listing = browse_folder(tree, max_folders=2)
    assert [folder.name for folder in listing.folders] == ["alpha", "Beta"] and listing.truncated
    assert browse_folder(tree, max_entries=1).truncated


def test_a_folder_swapped_after_the_check_is_not_listed(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The folder is opened once after the checks and must still be the folder that was checked (TOCTOU)."""
    real = fs._backend.canonical

    def swapped(path: str) -> tuple[str, object]:
        canonical, info = real(path)
        return canonical, type(info)(info.kind, info.dev, info.ino + 1, info.size, info.nlink, info.mtime_ns)

    monkeypatch.setattr(fs._backend, "canonical", swapped)
    assert _code(lambda: browse_folder(tree)) == "fs.root_changed"


def test_a_hidden_folder_this_user_cannot_open_is_reported_as_not_shown(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another user's /root cannot even be opened; it is still ``fs.not_browsable``, as a root would be
    ``fs.protected_root``. A folder the browser would show stays ``fs.permission``."""
    system = next((folder for folder in system_folders() if Path(folder).is_dir()), None)
    if system is None:
        pytest.skip("no system folder exists here")
    real = fs._backend.canonical

    closed = {fs.fold(os.path.realpath(system)), fs.fold(str(tree.resolve()))}

    def denied(path: str) -> tuple[str, object]:
        if fs.fold(os.path.realpath(path)) in closed:
            raise FsError("fs.permission", "The root folder cannot be opened (Permission denied)")
        return real(path)

    monkeypatch.setattr(fs._backend, "canonical", denied)
    assert _code(lambda: browse_folder(system, home=tree)) == "fs.not_browsable"
    assert _code(lambda: browse_folder(tree, home=tree)) == "fs.permission"


def test_no_answer_tells_whether_something_exists_in_a_refused_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N-11 review #1: a file, a missing name or a folder below a refused folder (another user's home, a
    blocked folder, a protected folder, a system folder) all answer ``fs.not_browsable``, so the answers
    enumerate neither files nor users. Outside refused folders the codes stay as they are."""
    users = tmp_path / "users"
    (users / "bob" / "proj").mkdir(parents=True)
    (users / "bob" / "notes.txt").write_text("x", encoding="utf-8")
    home = users / "ada"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_ed25519").write_text("key", encoding="utf-8")
    (home / "data").mkdir()
    (home / "data" / "file.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(fs, "home_parents", lambda: [str(users)])
    refused = [
        users / "bob" / "notes.txt",
        users / "bob" / "proj",
        users / "bob" / "nope",
        users / "carol",
        users / "carol" / "nope",
        home / ".ssh" / "id_ed25519",
        home / ".ssh" / "missing",
        home / "data" / "file.txt",
        home / "data" / "missing",
    ]
    for path in refused:
        assert _code(lambda path=path: browse_folder(path, home=home, protected=[home / "data"])) == (
            "fs.not_browsable"
        ), path
    # A link in the own home that leads into a refused folder answers alike, whatever its target.
    _link(home / "to-bob", users / "bob")
    for path in (home / "to-bob" / "notes.txt", home / "to-bob" / "nope"):
        assert _code(lambda path=path: browse_folder(path, home=home)) == "fs.not_browsable", path
    # The own home still tells a file and a missing folder apart.
    (home / "own.txt").write_text("x", encoding="utf-8")
    assert _code(lambda: browse_folder(home / "own.txt", home=home)) == "fs.not_a_directory"
    assert _code(lambda: browse_folder(home / "missing", home=home)) == "fs.not_found"


@pytest.mark.skipif(not POSIX, reason="/etc/shadow and /etc are POSIX system files")
def test_a_file_or_a_missing_name_in_a_system_folder_is_not_shown(tmp_path: Path) -> None:
    for path in ("/etc/shadow", "/etc/passwd", "/etc/no-such-file-n11", "/usr/no-such-folder-n11"):
        assert _code(lambda path=path: browse_folder(path, home=tmp_path)) == "fs.not_browsable", path


def test_the_error_code_is_documented() -> None:
    assert "fs.not_browsable" in fs.FS_ERROR_CODES
