"""SafeRoot beyond the name attacks: races, roots, links, special files, limits, Unicode, writing.

The catalog ids (N10-xx, W06-x) of orkexa.testing.pathguard.CATALOG are named at each test. A test that only
makes sense on one operating system says why in its skip reason (a physical difference, never a failure).
"""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import unicodedata

import pytest

from orkexa import fs
from orkexa.fs import FsError, SafeRoot, display_name, root_problem, system_folders
from orkexa.fs._names import is_cloud_tag, reparse_problem
from orkexa.testing import pathguard
from orkexa.testing.pathguard import CANARY, build_layout, swap_race

LINUX = sys.platform.startswith("linux")
DARWIN = sys.platform == "darwin"
WINDOWS = sys.platform == "win32"
POSIX = os.name == "posix"
#: tests/fs/conftest.py: the POSIX fallbacks (no openat2, no renameat2) are forced in this run.
FALLBACKS = os.environ.get("ORKEXA_TEST_FS_FALLBACKS") == "1" and POSIX


def _code(call: object) -> str:
    assert callable(call)
    with pytest.raises(FsError) as raised:
        call()
    return raised.value.code


@pytest.fixture
def layout(tmp_path: Path) -> pathguard.Layout:
    return build_layout(tmp_path)


@pytest.fixture
def root(layout: pathguard.Layout) -> SafeRoot:
    return SafeRoot(layout.root)


# ---- N10-02 absolute paths, N10-03 internal links ----------------------------------------------------------


def test_absolute_outside(layout: pathguard.Layout, root: SafeRoot) -> None:
    for path in (str(layout.secret), str(layout.secret).replace("\\", "/"), str(layout.root / "inside.txt")):
        assert _code(lambda path=path: root.read_bytes(path)) == "fs.absolute_path"


def test_internal_links(layout: pathguard.Layout, root: SafeRoot) -> None:
    """A link that stays inside the root is refused as well: links are never followed (documented)."""
    assert layout.created["link_inside"]
    assert _code(lambda: root.read_bytes("link_inside")) == "fs.link"
    assert root.read_bytes("sub/deep.txt") == b"deep"


# ---- N10-04 swap races -------------------------------------------------------------------------------------


def test_swap_race_file(layout: pathguard.Layout, root: SafeRoot) -> None:
    assert layout.created["link_file"]
    assert swap_race(root.read_bytes, layout, iterations=10_000) == []


def test_swap_race_folder(layout: pathguard.Layout, root: SafeRoot) -> None:
    assert swap_race(root.read_bytes, layout, iterations=10_000, folder=True) == []
    assert os.path.lexists(layout.root / "race")  # the race really ran with a link to swap in


# ---- N10-05 the root replaced ------------------------------------------------------------------------------


def _replace_with_link(path: Path, target: Path) -> None:
    if WINDOWS:
        assert pathguard.make_junction(path, target)
    else:
        os.symlink(target, path, target_is_directory=True)


def test_root_replaced(tmp_path: Path) -> None:
    granted = tmp_path / "granted"
    granted.mkdir()
    (granted / "a.txt").write_bytes(b"granted")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.txt").write_bytes(CANARY)
    root = SafeRoot(granted)
    assert root.read_bytes("a.txt") == b"granted"
    os.rename(granted, tmp_path / "moved")
    _replace_with_link(granted, outside)
    assert _code(lambda: root.read_bytes("a.txt")) == "fs.root_changed"
    assert _code(lambda: root.list("")) == "fs.root_changed"
    os.rmdir(granted) if WINDOWS else os.unlink(granted)
    os.rename(outside, granted)  # another real folder at the same path
    assert _code(lambda: root.read_bytes("a.txt")) == "fs.root_changed"


def test_expect_identity(tmp_path: Path) -> None:
    folder = tmp_path / "granted"
    folder.mkdir()
    identity = SafeRoot(folder).identity
    assert SafeRoot(folder, expect_identity=identity).identity == identity
    os.rename(folder, tmp_path / "kept")  # kept, so its inode is not reused
    folder.mkdir()
    assert _code(lambda: SafeRoot(folder, expect_identity=identity)) == "fs.root_changed"


# ---- N10-06, W06-1 hard links ------------------------------------------------------------------------------


def test_hard_links(layout: pathguard.Layout) -> None:
    assert layout.created["hard"]
    assert _code(lambda: SafeRoot(layout.root).read_bytes("hard")) == "fs.hard_link"
    opted_in = SafeRoot(layout.root, allow_hard_links=True)
    assert (
        opted_in.read_bytes("hard") == CANARY
    )  # the operator chose this, knowing the file may live elsewhere
    (layout.root / "sub" / "twin").write_bytes(b"x")
    os.link(layout.root / "sub" / "twin", layout.root / "sub" / "twin2")
    assert (
        _code(lambda: SafeRoot(layout.root).read_bytes("sub/twin")) == "fs.hard_link"
    )  # also inside the root


# ---- N10-07 mounts -----------------------------------------------------------------------------------------


def _privileged(command: list[str]) -> list[str] | None:
    if POSIX and os.geteuid() == 0:
        return command
    if shutil.which("sudo") and subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0:
        return ["sudo", "-n", *command]
    return None


@pytest.mark.skipif(not LINUX, reason="bind mounts and tmpfs are Linux mounts")
@pytest.mark.parametrize("openat2", [True, False], ids=["openat2", "fallback"])
def test_linux_mounts_below_the_root(
    layout: pathguard.Layout, openat2: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N10-07, review L1: refused by openat2 (RESOLVE_NO_XDEV) and by the fallback's mount-id check alike.

    The bind-mounted folder has its own canary with a single link, so ``fs.hard_link`` cannot be what refuses
    it (the layout's ``outside/secret.txt`` has a second link, ``root/hard``).
    """
    from orkexa.fs import _posix

    mount = _privileged(["mount"])
    if mount is None:
        pytest.skip("mounting needs root or passwordless sudo")
    umount = _privileged(["umount"])
    assert umount is not None
    if openat2 and not _posix.openat2_available():
        pytest.skip("this kernel has no openat2")
    if not openat2:
        monkeypatch.setitem(_posix._openat2_state, "usable", False)
    elsewhere = layout.base / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "secret.txt").write_bytes(CANARY)
    assert os.stat(elsewhere / "secret.txt").st_nlink == 1
    bind, tmpfs = layout.root / "bind", layout.root / "tmpfs"
    bind.mkdir()
    tmpfs.mkdir()
    subprocess.run([*mount, "--bind", str(elsewhere), str(bind)], check=True)
    try:
        subprocess.run([*mount, "-t", "tmpfs", "-o", "size=1m", "tmpfs", str(tmpfs)], check=True)
        try:
            root = SafeRoot(layout.root)
            assert (layout.root / "bind" / "secret.txt").read_bytes() == CANARY  # the mount is there
            assert _code(lambda: root.read_bytes("bind/secret.txt")) == "fs.cross_device"
            assert _code(lambda: root.list("bind")) == "fs.cross_device"
            assert _code(lambda: root.list("tmpfs")) == "fs.cross_device"
            assert root.read_bytes("inside.txt") == b"inside"
        finally:
            subprocess.run([*umount, str(tmpfs)], check=True)
    finally:
        subprocess.run([*umount, str(bind)], check=True)


@pytest.mark.skipif(not DARWIN, reason="disk images (hdiutil) are a macOS feature")
def test_macos_disk_image_below_the_root(layout: pathguard.Layout) -> None:
    image = layout.base / "disk.dmg"
    subprocess.run(
        ["hdiutil", "create", "-size", "2m", "-fs", "HFS+", "-volname", "orkexa", str(image)],
        check=True,
        capture_output=True,
    )
    mounted = layout.root / "volume"
    mounted.mkdir()
    subprocess.run(
        ["hdiutil", "attach", "-nobrowse", "-mountpoint", str(mounted), str(image)],
        check=True,
        capture_output=True,
    )
    try:
        (mounted / "secret.txt").write_bytes(CANARY)
        root = SafeRoot(layout.root)
        assert _code(lambda: root.read_bytes("volume/secret.txt")) == "fs.cross_device"
    finally:
        subprocess.run(["hdiutil", "detach", str(mounted)], check=True, capture_output=True)


# ---- N10-08 magic links ------------------------------------------------------------------------------------


@pytest.mark.skipif(not LINUX, reason="/proc magic links exist on Linux only")
def test_linux_magic_links(layout: pathguard.Layout) -> None:
    os.symlink("/proc/self/root/etc/passwd", layout.root / "magic_root")
    with open(layout.secret, "rb") as handle:
        os.symlink(f"/proc/self/fd/{handle.fileno()}", layout.root / "magic_fd")
        root = SafeRoot(layout.root)
        assert _code(lambda: root.read_bytes("magic_root")) == "fs.link"
        assert _code(lambda: root.read_bytes("magic_fd")) == "fs.link"
    assert _code(lambda: root.read_bytes("/proc/self/root/etc/passwd")) == "fs.absolute_path"
    assert _code(lambda: SafeRoot("/proc")) == "fs.protected_root"
    assert _code(lambda: SafeRoot("/proc/self/root/proc")) == "fs.protected_root"


# ---- N10-09 special files ----------------------------------------------------------------------------------


def test_special_files(layout: pathguard.Layout) -> None:
    root = SafeRoot(layout.root)
    for device in ("NUL", "CON", "aux.txt"):
        assert _code(lambda device=device: root.read_bytes(device)) == "fs.invalid_path"
    if WINDOWS:
        return  # no FIFOs, sockets or device nodes in a Windows folder; the device names above are the case
    os.mkfifo(layout.root / "fifo")
    os.symlink("/dev/zero", layout.root / "zero")
    os.symlink("/dev/random", layout.root / "random")
    short = Path(os.environ.get("TMPDIR", "/tmp")) / f"ok{os.getpid()}"  # AF_UNIX paths are short
    short.mkdir(exist_ok=True)
    server = socket.socket(socket.AF_UNIX)
    try:
        server.bind(str(short / "s"))
        sock_root = SafeRoot(short)
        started = time.monotonic()
        assert _code(lambda: root.read_bytes("fifo")) == "fs.not_regular_file"  # O_NONBLOCK: no hang
        assert _code(lambda: sock_root.read_bytes("s")) == "fs.not_regular_file"
        assert _code(lambda: root.read_bytes("zero")) == "fs.link"
        assert _code(lambda: root.read_bytes("random")) == "fs.link"
        assert time.monotonic() - started < 5
    finally:
        server.close()
        shutil.rmtree(short, ignore_errors=True)
    assert _code(lambda: SafeRoot("/dev")) == "fs.protected_root"
    kinds = {entry.name: entry.kind for entry in root.list("")}
    assert kinds["fifo"] == "other" and kinds["zero"] == "link"


# ---- N10-10 size limits ------------------------------------------------------------------------------------


def _sparse(path: Path, size: int) -> None:
    path.write_bytes(b"")
    if WINDOWS:
        subprocess.run(["fsutil", "sparse", "setflag", str(path)], check=True, capture_output=True)
    with open(path, "r+b") as handle:
        # Not truncate(): on Windows the C runtime extends a file by writing zeros, which fills the disk.
        handle.seek(size - 1)
        handle.write(b"\0")


def test_large_and_sparse_files(tmp_path: Path) -> None:
    folder = tmp_path / "root"
    folder.mkdir()
    _sparse(folder / "huge.bin", 100 * 1024**3)
    (folder / "big.txt").write_bytes(b"x" * 2048)
    root = SafeRoot(folder, max_file_bytes=1024)
    started = time.monotonic()
    assert _code(lambda: SafeRoot(folder).read_bytes("huge.bin")) == "fs.too_large"
    assert time.monotonic() - started < 5  # refused from the size, nothing read
    with pytest.raises(FsError) as raised:
        root.read_bytes("big.txt")
    assert raised.value.code == "fs.too_large"
    assert "2048 bytes" in str(raised.value) and "1024" in str(raised.value)  # an understandable message
    assert _code(lambda: root.read_bytes("big.txt", max_bytes=10)) == "fs.too_large"


# ---- N10-11, N10-18 blocked names and protected folders ----------------------------------------------------


def test_blocked_spellings(tmp_path: Path) -> None:
    folder = tmp_path / "root"
    nfc = unicodedata.normalize("NFC", "Privé")
    (folder / nfc).mkdir(parents=True)
    (folder / nfc / "x.txt").write_bytes(b"private")
    (folder / "Straße").mkdir()
    (folder / "Straße" / "x.txt").write_bytes(b"street")
    root = SafeRoot(folder, blocked=["**/Privé", "**/STRASSE", "**/İnfo"])
    for path in (
        f"{nfc}/x.txt",
        unicodedata.normalize("NFD", f"{nfc}/x.txt"),
        f"{nfc.upper()}/x.txt",
        "Straße/x.txt",  # ß case-folds to ss
        "strasse/x.txt",
        "i\u0307nfo/a.txt",  # İ case-folds to i + combining dot above
    ):
        assert _code(lambda path=path: root.read_bytes(path)) == "fs.blocked", path
    assert root.list("") == []
    assert SafeRoot(folder, blocked=[]).read_bytes("Straße/x.txt") == b"street"


def test_protected_folder_any_spelling(tmp_path: Path) -> None:
    home = tmp_path / "home"
    data = home / "AppData"
    data.mkdir(parents=True)
    (data / "vault.db").write_bytes(CANARY)
    root = SafeRoot(home, protected=[data])
    assert _code(lambda: root.read_bytes("AppData/vault.db")) == "fs.blocked"
    # Protected folders are compared by name too (folded), so every spelling is refused on every system.
    assert _code(lambda: root.read_bytes("APPDATA/vault.db")) == "fs.blocked"
    assert _code(lambda: root.list("appdata")) == "fs.blocked"
    assert [entry.name for entry in root.list("")] == []  # not listed either
    if not WINDOWS:
        os.symlink(data, home / "alias")
        assert _code(lambda: root.read_bytes("alias/vault.db")) == "fs.link"


def test_patterns_fold_case(tmp_path: Path) -> None:
    folder = tmp_path / "root"
    (folder / "docs").mkdir(parents=True)
    (folder / "docs" / "A.MD").write_bytes(b"a")
    (folder / "docs" / "key.PEM").write_bytes(b"k")
    (folder / "notes.txt").write_bytes(b"n")
    root = SafeRoot(folder, include=["docs/**"], exclude=["**/*.pem"])
    assert root.read_bytes("docs/A.MD") == b"a"
    assert _code(lambda: root.read_bytes("docs/key.PEM")) == "fs.excluded"
    assert _code(lambda: root.read_bytes("notes.txt")) == "fs.excluded"
    assert [entry.name for entry in root.list("docs")] == ["A.MD"]
    assert [entry.name for entry in root.list("")] == ["docs"]
    excluded_folder = SafeRoot(folder, exclude=["DOCS"])
    assert _code(lambda: excluded_folder.read_bytes("docs/A.MD")) == "fs.excluded"
    with pytest.raises(ValueError):
        SafeRoot(folder, exclude=["../x"])
    with pytest.raises(TypeError):
        SafeRoot(folder, exclude="*.pem")


def test_protected_inside_a_wide_root(tmp_path: Path) -> None:
    """N10-22: the product's data folder and credential folders inside a root that is too wide."""
    home = tmp_path / "home"
    data = home / ".locria"
    (data / "runs" / "r1").mkdir(parents=True)
    (data / "vault.json").write_bytes(CANARY)
    (data / "runs" / "r1" / "out.txt").write_bytes(CANARY)
    for credential in (".aws/credentials", ".config/gcloud/key.json", "Library/Keychains/k.db", ".netrc"):
        target = home / credential
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(CANARY)
    root = SafeRoot(home, protected=[data])
    blocked = (".locria/vault.json", ".locria/runs/r1/out.txt", ".aws/credentials", ".config/gcloud/key.json")
    for path in (*blocked, "Library/Keychains/k.db", ".netrc"):
        assert _code(lambda path=path: root.read_bytes(path)) == "fs.blocked", path


# ---- N10-12 macOS ------------------------------------------------------------------------------------------


@pytest.mark.skipif(not DARWIN, reason="firmlinks and /System/Volumes/Data exist on macOS only")
def test_macos_firmlink_root(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    real = os.path.realpath(data)
    detour = "/System/Volumes/Data" + real
    assert os.path.isdir(detour)
    assert _code(lambda: SafeRoot(detour, protected=[data])) == "fs.protected_root"
    assert _code(lambda: SafeRoot("/System/Volumes/Data")) == "fs.protected_root"
    assert _code(lambda: SafeRoot("/system/library")) == "fs.protected_root"  # case-insensitive APFS


def test_finder_alias_is_a_file(tmp_path: Path) -> None:
    """N10-12: a Finder alias is a regular file with bookmark data, never followed like a link."""
    folder = tmp_path / "root"
    folder.mkdir()
    alias = b"book\x00\x00\x00\x00mark\x00\x00\x00\x00" + str(tmp_path / "outside").encode()
    (folder / "Secret alias").write_bytes(alias)
    assert SafeRoot(folder).read_bytes("Secret alias") == alias


# ---- N10-16, W06-4 roots that are device or network paths --------------------------------------------------


def test_device_and_unc_roots() -> None:
    host = socket.gethostname()
    for path, code in (
        ("\\\\.\\PhysicalDrive0", "fs.invalid_path"),
        ("\\\\?\\GLOBALROOT\\Device\\HarddiskVolume1", "fs.invalid_path"),
        ("\\\\?\\Volume{00000000-0000-0000-0000-000000000000}\\", "fs.invalid_path"),
        ("\\??\\C:\\Users", "fs.invalid_path"),
        ("\\\\?\\UNC\\server\\share", "fs.network_path"),
        ("\\\\server\\share\\folder", "fs.network_path"),
        ("\\\\localhost\\c$\\Users", "fs.network_path"),
        ("\\\\127.0.0.1\\c$\\Users", "fs.network_path"),
        (f"\\\\{host}\\c$\\Users", "fs.network_path"),
        (f"//{host}/c$/Users", "fs.network_path"),
        ("//server/share", "fs.network_path"),
        ("relative/folder", "fs.invalid_path"),
        (" /tmp", "fs.invalid_path"),
        ("/tmp\x00x", "fs.invalid_path"),
    ):
        assert _code(lambda path=path: SafeRoot(path)) == code, path


def test_device_prefix_of_a_local_drive_is_the_drive(tmp_path: Path) -> None:
    folder = tmp_path / "root"
    folder.mkdir()
    if WINDOWS:
        long_form = "\\\\?\\" + str(folder)
        assert SafeRoot(long_form).path.casefold() == SafeRoot(folder).path.casefold()
    else:
        assert _code(lambda: SafeRoot("\\\\?\\C:\\Users")) == "fs.invalid_path"  # no drive here: not absolute


def test_cloud_placeholders() -> None:
    """W06-4: OneDrive placeholders are refused unless the root allows them; links are always refused."""
    cloud, recall = 0x9000001A, 0x00400000
    assert is_cloud_tag(cloud) and is_cloud_tag(0x9000F01A) and not is_cloud_tag(0xA000000C)
    assert reparse_problem(0x400, cloud, allow_cloud=False) == "fs.cloud_file"
    assert reparse_problem(recall, 0, allow_cloud=False) == "fs.cloud_file"
    assert reparse_problem(0x400, cloud, allow_cloud=True) is None
    assert reparse_problem(0x400, 0xA000000C, allow_cloud=True) == "fs.link"  # symbolic link
    assert reparse_problem(0x400, 0xA0000003, allow_cloud=True) == "fs.link"  # junction, mount point
    assert reparse_problem(0x400, 0x8000001B, allow_cloud=True) == "fs.link"  # app execution alias
    assert reparse_problem(0x400, 0x80000013, allow_cloud=False) is None  # deduplication keeps the data
    assert reparse_problem(0x20, 0, allow_cloud=False) is None


# ---- N10-19 Unicode ----------------------------------------------------------------------------------------


def test_bidi_names(tmp_path: Path) -> None:
    folder = tmp_path / "root"
    folder.mkdir()
    name = "invoice\u202efdp.exe"
    (folder / name).write_bytes(b"bidi")
    root = SafeRoot(folder)
    assert root.read_bytes(name) == b"bidi"  # readable (P0-2 Unicode decision) ...
    (entry,) = root.list("")
    assert entry.display == "invoice\\u202efdp.exe"  # ... but shown escaped
    assert display_name("a\u200bb") == "a\\u200bb"


# ---- N10-21 revocation -------------------------------------------------------------------------------------


def _ignore(*args: object) -> None:
    del args


def test_revoke(tmp_path: Path) -> None:
    folder = tmp_path / "root"
    folder.mkdir()
    (folder / "big.bin").write_bytes(b"\x00" * (8 << 20))
    (folder / "small.txt").write_bytes(b"s")
    root = SafeRoot(folder)
    chunks = {"read": 0}

    def revoked_after_the_first_chunk() -> bool:
        chunks["read"] += 1
        if chunks["read"] == 2:
            root.revoke()  # revoked while the 8 MiB read is in its second chunk
        return root.revoked

    backend = fs._backend
    with (
        pytest.raises(FsError) as raised,
        backend.walk(
            root.path, root.identity, ("big.bin",), final="file", on_node=_ignore, shown="big.bin"
        ) as (
            node,
            _,
        ),
    ):
        backend.read(node, 16 << 20, revoked_after_the_first_chunk, "big.bin")
    assert raised.value.code == "fs.revoked" and chunks["read"] == 2
    assert root.revoked
    for call in (lambda: root.read_bytes("small.txt"), lambda: root.list(""), lambda: root.stat("small.txt")):
        assert _code(call) == "fs.revoked"


def test_revoke_stops_a_read_in_progress(tmp_path: Path) -> None:
    folder = tmp_path / "root"
    folder.mkdir()
    (folder / "big.bin").write_bytes(b"\x01" * (32 << 20))
    root = SafeRoot(folder)
    errors: list[str] = []

    def reader() -> None:
        try:
            for _ in range(200):
                root.read_bytes("big.bin")
        except FsError as exc:
            errors.append(exc.code)

    thread = threading.Thread(target=reader)
    thread.start()
    time.sleep(0.05)
    root.revoke()
    thread.join(timeout=60)
    assert errors == ["fs.revoked"]


# ---- N10-23 roots that cannot be granted -------------------------------------------------------------------


def test_roots_that_cannot_be_granted(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "inner").mkdir(parents=True)
    candidates = [os.path.abspath(os.sep), str(Path.home()), *system_folders()]
    if WINDOWS:
        candidates += [
            os.environ.get("SYSTEMDRIVE", "C:") + "\\",
            os.environ.get("WINDIR", "C:\\Windows") + "\\System32",
        ]
    else:
        candidates += ["/etc", "/usr/lib", "/var/lib", "/root", "/dev", "/System/Library", "/LIBRARY"]
    for path in candidates:
        if not os.path.isdir(path):
            continue
        assert _code(lambda path=path: SafeRoot(path)) == "fs.protected_root", path
    assert _code(lambda: SafeRoot(data, protected=[data])) == "fs.protected_root"
    assert _code(lambda: SafeRoot(data / "inner", protected=[data])) == "fs.protected_root"
    if not WINDOWS:
        os.symlink(data / "inner", tmp_path / "alias")
        assert _code(lambda: SafeRoot(tmp_path / "alias", protected=[data])) == "fs.protected_root"
    SafeRoot(tmp_path, protected=[data])  # a root that contains the data folder is fine (blocked inside)
    assert root_problem(str(tmp_path)) is None


# ---- W06-2 NFC and NFD -------------------------------------------------------------------------------------


def test_nfc_and_nfd_names(tmp_path: Path) -> None:
    folder = tmp_path / "root"
    folder.mkdir()
    nfc, nfd = unicodedata.normalize("NFC", "café.txt"), unicodedata.normalize("NFD", "café.txt")
    (folder / nfd).write_bytes(b"stored as NFD")
    root = SafeRoot(folder)
    assert root.read_bytes(nfc) == b"stored as NFD"  # input is NFC; the stored NFD name is found
    assert root.read_bytes(nfd) == b"stored as NFD"
    (entry,) = root.list("")
    assert entry.path == nfc
    try:
        (folder / nfc).write_bytes(b"stored as NFC")
    except OSError:
        return
    if len(os.listdir(folder)) == 1:
        return  # a normalization-insensitive file system (APFS) has one name for both
    assert root.read_bytes(nfc) == b"stored as NFC"  # the exact name wins
    assert all(entry.ambiguous for entry in root.list(""))


def test_ambiguous_names_are_not_guessed(tmp_path: Path) -> None:
    folder = tmp_path / "root"
    folder.mkdir()
    variants = ["e\u0301\u0323.txt", "e\u0323\u0301.txt"]  # two non-NFC orders of the same NFC name
    for name in variants:
        try:
            (folder / name).write_bytes(name.encode())
        except OSError:
            return
    if len(os.listdir(folder)) != 2:
        return  # the file system treats them as one name
    nfc = unicodedata.normalize("NFC", variants[0])
    assert _code(lambda: SafeRoot(folder).read_bytes(nfc)) == "fs.ambiguous_name"


# ---- W06-3 creating and renaming ---------------------------------------------------------------------------


def test_create_file(layout: pathguard.Layout) -> None:
    assert _code(lambda: SafeRoot(layout.root).create_file("new.txt", b"x")) == "fs.read_only"
    root = SafeRoot(layout.root, writable=True, max_file_bytes=1024)
    root.create_file("sub/new.txt", b"created")
    assert root.read_bytes("sub/new.txt") == b"created"
    assert _code(lambda: root.create_file("sub/new.txt", b"again")) == "fs.exists"
    assert root.read_bytes("sub/new.txt") == b"created"  # never replaced
    assert _code(lambda: root.create_file("link_file", b"through the link")) == "fs.exists"
    assert layout.secret.read_bytes() == CANARY  # never followed
    assert _code(lambda: root.create_file("link_dir/new.txt", b"x")) == "fs.link"
    assert not (layout.outside / "new.txt").exists()
    assert _code(lambda: root.create_file("../new.txt", b"x")) == "fs.parent_reference"
    assert _code(lambda: root.create_file(".ssh/authorized_keys", b"x")) == "fs.blocked"
    assert _code(lambda: root.create_file("big.bin", b"x" * 2048)) == "fs.too_large"
    assert _code(lambda: root.create_file("missing/new.txt", b"x")) == "fs.not_found"


def test_rename(layout: pathguard.Layout) -> None:
    root = SafeRoot(layout.root, writable=True)
    assert _code(lambda: SafeRoot(layout.root).rename("inside.txt", "x.txt")) == "fs.read_only"
    root.rename("inside.txt", "sub/moved.txt")
    assert root.read_bytes("sub/moved.txt") == b"inside"
    assert _code(lambda: root.rename("sub/moved.txt", "sub/deep.txt")) == "fs.exists"
    assert (layout.root / "sub" / "deep.txt").read_bytes() == b"deep"  # never replaced
    if FALLBACKS:  # without renameat2 a folder cannot be renamed without the risk of replacing one
        assert _code(lambda: root.rename("sub", "renamed")) == "fs.unsupported_type"
        folder = "sub"
    else:
        root.rename("sub", "renamed")
        folder = "renamed"
    assert root.read_bytes(f"{folder}/deep.txt") == b"deep"
    assert _code(lambda: root.rename("link_file", "x")) == "fs.link"
    assert _code(lambda: root.rename("link_dir/secret.txt", "stolen.txt")) == "fs.link"
    assert _code(lambda: root.rename("hard", "x")) == "fs.hard_link"
    assert _code(lambda: root.rename(f"{folder}/deep.txt", "../deep.txt")) == "fs.parent_reference"
    assert _code(lambda: root.rename(f"{folder}/deep.txt", ".ssh/deep.txt")) == "fs.blocked"
    assert _code(lambda: root.rename("", "x")) == "fs.invalid_path"
    assert layout.secret.read_bytes() == CANARY


def test_rename_refuses_a_protected_folder(tmp_path: Path) -> None:
    home = tmp_path / "home"
    data = home / "data"
    data.mkdir(parents=True)
    root = SafeRoot(home, protected=[data], writable=True)
    assert _code(lambda: root.rename("data", "elsewhere")) == "fs.blocked"
    assert data.is_dir()


# ---- general -----------------------------------------------------------------------------------------------


def test_stat_and_errors(layout: pathguard.Layout, root: SafeRoot) -> None:
    assert root.stat("sub").kind == "dir"
    assert root.stat("inside.txt").size == len(b"inside")
    assert root.stat("").kind == "dir"
    assert _code(lambda: root.read_bytes("sub")) == "fs.is_a_directory"
    assert _code(lambda: root.list("inside.txt")) == "fs.not_a_directory"
    assert _code(lambda: root.read_bytes("inside.txt/x")) == "fs.not_a_directory"
    assert _code(lambda: root.read_bytes("missing.txt")) == "fs.not_found"
    error = FsError("fs.blocked", "message", path="a/b")
    import pickle

    again = pickle.loads(pickle.dumps(error))
    assert (again.code, again.path, str(again)) == ("fs.blocked", "a/b", "message")
    assert isinstance(error, PermissionError)


def test_the_listing_limit(tmp_path: Path) -> None:
    folder = tmp_path / "root"
    folder.mkdir()
    for number in range(5):
        (folder / f"{number}.txt").write_bytes(b"")
    assert len(SafeRoot(folder).list("")) == 5
    assert _code(lambda: SafeRoot(folder, max_entries=3).list("")) == "fs.too_many_entries"


@pytest.fixture(autouse=True)
def _no_leaked_threads() -> Iterator[None]:
    before = threading.active_count()
    yield
    deadline = time.monotonic() + 30
    while threading.active_count() > before and time.monotonic() < deadline:
        time.sleep(0.05)


def test_every_raised_code_is_documented() -> None:
    import re

    from orkexa.fs import FS_ERROR_CODES

    source = "".join(path.read_text(encoding="utf-8") for path in Path(fs.__file__).parent.glob("*.py"))
    raised = set(re.findall(r'"(fs\.[a-z_]+)"', source))
    assert raised <= set(FS_ERROR_CODES), sorted(raised - set(FS_ERROR_CODES))
    assert len(raised) >= 25
