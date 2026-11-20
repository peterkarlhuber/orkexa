"""The POSIX side of :mod:`orkexa.fs`: directory file descriptors, one component at a time.

Every component is opened relative to the descriptor of its folder with ``O_NOFOLLOW``, so a symbolic link is
never followed, wherever it is on the way; folders are opened with ``O_DIRECTORY`` and files with
``O_NONBLOCK`` (a FIFO cannot block the open) and ``O_NOCTTY``. On Linux the component is opened with
``openat2`` and ``RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_XDEV`` when the
kernel has it (5.6+), which also refuses a bind mount on the same file system; otherwise, and on macOS,
``openat`` with ``O_NOFOLLOW`` and a comparison of ``st_dev`` with the root's. Each descriptor is checked with
``fstat`` and the file is read through that same descriptor: no path is resolved twice. On macOS
``fcntl(F_GETPATH)`` of every opened descriptor must lie below the root's as a second check (case-folded: APFS
is usually case-insensitive).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
import ctypes
import errno
import os
import stat
import sys
from typing import Any

from ._names import FsError, fold, path_parts
from ._types import Info

_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_NOCTTY = getattr(os, "O_NOCTTY", 0)
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | _NOCTTY | _CLOEXEC
_ANY_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | _NOCTTY | _CLOEXEC
_DARWIN = sys.platform == "darwin"
_LINUX = sys.platform.startswith("linux")
_F_GETPATH = 50  # macOS <sys/fcntl.h>
#: What opening a link with O_NOFOLLOW answers: ELOOP (Linux, macOS), EMLINK (FreeBSD), EFTYPE (some BSDs),
#: and ENOTDIR with O_DIRECTORY (macOS); the entry itself tells which it was.
_LINK_ERRORS = frozenset(
    code
    for code in (errno.ELOOP, errno.EMLINK, errno.ENOTDIR, getattr(errno, "EFTYPE", None))
    if code is not None
)

# ---- openat2 (Linux 5.6+) ----------------------------------------------------------------------------------

_SYS_OPENAT2 = 437  # the same number on every Linux architecture (the unified syscall table)
_RESOLVE_NO_XDEV, _RESOLVE_NO_MAGICLINKS, _RESOLVE_NO_SYMLINKS, _RESOLVE_BENEATH = 0x01, 0x02, 0x04, 0x08


class _OpenHow(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint64), ("mode", ctypes.c_uint64), ("resolve", ctypes.c_uint64)]


_openat2_state: dict[str, Any] = {"usable": _LINUX, "syscall": None}


def _openat2(dir_fd: int, name: str, flags: int) -> int | None:
    """``openat2`` of one component, or ``None`` when the kernel (or a seccomp filter) does not offer it."""
    if not _openat2_state["usable"]:
        return None
    syscall = _openat2_state["syscall"]
    if syscall is None:
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            syscall = libc.syscall
            syscall.restype = ctypes.c_long
        except (OSError, AttributeError):
            _openat2_state["usable"] = False
            return None
        _openat2_state["syscall"] = syscall
    how = _OpenHow(
        flags, 0, _RESOLVE_BENEATH | _RESOLVE_NO_SYMLINKS | _RESOLVE_NO_MAGICLINKS | _RESOLVE_NO_XDEV
    )
    result = syscall(
        ctypes.c_long(_SYS_OPENAT2),
        ctypes.c_int(dir_fd),
        ctypes.c_char_p(os.fsencode(name)),
        ctypes.byref(how),
        ctypes.c_size_t(ctypes.sizeof(how)),
    )
    if result >= 0:
        return int(result)
    code = ctypes.get_errno()
    if code in (errno.ENOSYS, errno.EPERM, errno.E2BIG):  # no openat2 here (old kernel, seccomp)
        _openat2_state["usable"] = False
        return None
    raise OSError(code, os.strerror(code))


def openat2_available() -> bool:
    """Whether components are opened with ``openat2`` (Linux 5.6+ without a seccomp filter against it)."""
    if not _LINUX:
        return False
    fd = os.open(".", _DIRECTORY_FLAGS)
    try:
        with suppress(OSError):
            child = _openat2(fd, ".", _DIRECTORY_FLAGS)
            if child is not None:
                os.close(child)
    finally:
        os.close(fd)
    return bool(_openat2_state["usable"])


# ---- descriptors -------------------------------------------------------------------------------------------


def _info(fd: int) -> Info:
    st = os.fstat(fd)
    mode = st.st_mode
    kind = "dir" if stat.S_ISDIR(mode) else "file" if stat.S_ISREG(mode) else "other"
    return Info(
        kind=kind, dev=st.st_dev, ino=st.st_ino, size=st.st_size, nlink=st.st_nlink, mtime_ns=st.st_mtime_ns
    )


def _getpath(fd: int) -> str | None:
    if not _DARWIN:
        return None
    import fcntl

    raw = fcntl.fcntl(fd, _F_GETPATH, b"\0" * 1025)
    return os.fsdecode(raw.split(b"\0", 1)[0])


def stat_path(path: str) -> Info | None:
    """The identity of a folder named by a trusted path (links followed); ``None`` if it cannot be opened."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | _CLOEXEC)
    except OSError:
        return None
    try:
        return _info(fd)
    finally:
        os.close(fd)


def canonical(path: str) -> tuple[str, Info]:
    """The real path of an absolute folder (links resolved) and its identity."""
    try:
        real = os.path.realpath(path, strict=True)
    except FileNotFoundError:
        raise FsError("fs.not_found", "The root folder does not exist") from None
    except OSError as exc:
        raise FsError("fs.permission", f"The root folder cannot be opened ({exc.strerror})") from None
    try:
        fd = os.open(real, _DIRECTORY_FLAGS)
    except NotADirectoryError:
        raise FsError("fs.not_a_directory", "The root is not a folder") from None
    except OSError as exc:
        raise FsError("fs.permission", f"The root folder cannot be opened ({exc.strerror})") from None
    try:
        info = _info(fd)
        real = _getpath(fd) or real
    finally:
        os.close(fd)
    return real, info


def _open_root(path: str, identity: tuple[int, int]) -> tuple[int, Info]:
    try:
        fd = os.open(path, _DIRECTORY_FLAGS)
    except OSError:
        raise FsError("fs.root_changed", "The root folder is gone, moved or replaced") from None
    info = _info(fd)
    real = _getpath(fd)
    if (info.dev, info.ino) != identity or (real is not None and path_parts(real) != path_parts(path)):
        os.close(fd)
        raise FsError("fs.root_changed", "The root folder is not the folder that was checked")
    return fd, info


def _child(dir_fd: int, name: str, flags: int, shown: str) -> int:
    """Open one component (never following a link), mapping errors to :class:`FsError`."""
    try:
        opened = _openat2(dir_fd, name, flags)
        if opened is not None:
            return opened
        return os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        code = exc.errno
        if code == errno.ENOENT:
            raise FsError("fs.not_found", "No file or folder of this name", path=shown) from None
        if code == errno.EXDEV:
            raise FsError(
                "fs.cross_device", "The path crosses into another file system", path=shown
            ) from None
        if code in _LINK_ERRORS:
            try:
                lst = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
            except OSError:
                raise FsError("fs.not_found", "No file or folder of this name", path=shown) from None
            if stat.S_ISLNK(lst.st_mode):
                raise FsError(
                    "fs.link", "A symbolic link is on the way; links are not followed", path=shown
                ) from None
            if code == errno.ENOTDIR:
                raise FsError("fs.not_a_directory", "A folder was expected", path=shown) from None
            raise FsError("fs.link", "A link is on the way; links are not followed", path=shown) from None
        if code in (errno.ENXIO, errno.EOPNOTSUPP, getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)):
            raise FsError(
                "fs.not_regular_file", "Not a regular file (a socket or device)", path=shown
            ) from None
        if code in (errno.EACCES, errno.EPERM):
            raise FsError("fs.permission", "The operating system denied access", path=shown) from None
        raise FsError("fs.unreadable", f"Cannot be opened ({exc.strerror})", path=shown) from None


def _resolve(dir_fd: int, name: str, flags: int, shown: str) -> tuple[int, str]:
    """Open ``name`` exactly, or else the one entry whose NFC form is ``name`` (stored NFD, e.g. by macOS)."""
    try:
        return _child(dir_fd, name, flags, shown), name
    except FsError as exc:
        if exc.code != "fs.not_found" or name.isascii():
            raise
    import unicodedata

    found = [
        entry for entry in names(dir_fd) if unicodedata.normalize("NFC", entry) == name and entry != name
    ]
    if len(found) > 1:
        raise FsError("fs.ambiguous_name", "Several entries have this name after normalization", path=shown)
    if not found:
        raise FsError("fs.not_found", "No file or folder of this name", path=shown)
    return _child(dir_fd, found[0], flags, shown), found[0]


@contextmanager
def walk(
    root: str,
    identity: tuple[int, int],
    parts: tuple[str, ...],
    *,
    final: str,
    on_node: Callable[[int, Info, tuple[str, ...]], None],
    shown: str,
    allow_cloud: bool = False,  # no cloud placeholders on POSIX
) -> Iterator[tuple[int, Info]]:
    """Open the root and then ``parts`` one by one; yield the last descriptor and its :class:`Info`.

    ``final`` is ``"dir"``, ``"file"`` or ``"any"`` (the last component). ``on_node(index, info, names)`` is
    called for the root (index ``-1``) and every component with the names opened so far; it raises to refuse.
    Every descriptor is closed when the block ends.
    """
    del allow_cloud
    opened: list[int] = []
    try:
        fd, root_info = _open_root(root, identity)
        opened.append(fd)
        root_real = _getpath(fd) if _DARWIN else None
        on_node(-1, root_info, ())
        info = root_info
        actual: list[str] = []
        for index, name in enumerate(parts):
            last = index == len(parts) - 1
            flags = (
                _DIRECTORY_FLAGS
                if not last or final == "dir"
                else _FILE_FLAGS
                if final == "file"
                else _ANY_FLAGS
            )
            fd, stored = _resolve(opened[-1], name, flags, shown)
            opened.append(fd)
            actual.append(stored)
            info = _info(fd)
            if info.dev != root_info.dev:
                raise FsError("fs.cross_device", "The path crosses into another file system", path=shown)
            if info.kind == "other":
                raise FsError(
                    "fs.not_regular_file", "Not a regular file (a FIFO, socket or device)", path=shown
                )
            if root_real is not None and (info.kind == "dir" or info.nlink == 1):
                real = _getpath(fd)
                if real is None or not _below(real, root_real):
                    raise FsError("fs.outside_root", "The opened path is not below the root", path=shown)
            on_node(index, info, tuple(actual))
        if final == "file" and info.kind == "dir":
            raise FsError("fs.is_a_directory", "A file was expected, this is a folder", path=shown)
        if final == "dir" and info.kind != "dir":
            raise FsError("fs.not_a_directory", "A folder was expected", path=shown)
        yield opened[-1], info
    finally:
        for fd in reversed(opened):
            with suppress(OSError):
                os.close(fd)


def _below(path: str, root: str) -> bool:
    inner, outer = (
        [fold(part) for part in path.split("/") if part],
        [fold(part) for part in root.split("/") if part],
    )
    return len(inner) > len(outer) and inner[: len(outer)] == outer


def names(dir_fd: int) -> list[str]:
    """The entry names of an open folder."""
    with os.scandir(dir_fd) as entries:
        return [entry.name for entry in entries]


def listing(dir_fd: int, root: str, parts: tuple[str, ...]) -> Iterator[tuple[str, str, int]]:
    """``(name, kind, size)`` of every entry of an open folder (file, dir, link, other); no link followed."""
    del root, parts  # the descriptor names the folder
    for name in names(dir_fd):
        try:
            st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except OSError:
            continue
        mode = st.st_mode
        kind = (
            "link"
            if stat.S_ISLNK(mode)
            else "dir"
            if stat.S_ISDIR(mode)
            else "file"
            if stat.S_ISREG(mode)
            else "other"
        )
        yield name, kind, st.st_size if kind == "file" else 0


def read(fd: int, limit: int, cancelled: Callable[[], bool], shown: str) -> bytes:
    """At most ``limit + 1`` bytes of an open file, in chunks; stops when ``cancelled()``."""
    chunks: list[bytes] = []
    total = 0
    while total <= limit:
        if cancelled():
            raise FsError("fs.revoked", "The root was revoked while the file was read", path=shown)
        try:
            chunk = os.read(fd, min(1 << 20, limit + 1 - total))
        except BlockingIOError:
            raise FsError("fs.not_regular_file", "Not a regular file", path=shown) from None
        except OSError as exc:
            raise FsError("fs.unreadable", f"The file cannot be read ({exc.strerror})", path=shown) from None
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


# ---- creating and renaming ---------------------------------------------------------------------------------

_RENAME_NOREPLACE = 1  # Linux renameat2
_RENAME_EXCL = 0x4  # macOS renameatx_np
#: renameat2 by syscall number, where the C library has no wrapper (glibc before 2.28).
_SYS_RENAMEAT2 = {"x86_64": 316, "aarch64": 276, "arm64": 276, "riscv64": 276, "i686": 353, "i386": 353}


def create(dir_fd: int, root: str, parts: tuple[str, ...], name: str, data: bytes, shown: str) -> None:
    """Create ``name`` in the open folder ``dir_fd`` (``O_CREAT | O_EXCL | O_NOFOLLOW``) and write ``data``.

    ``O_EXCL`` never opens an existing entry, a symbolic link included, so nothing is overwritten or followed.
    """
    del root, parts
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | _CLOEXEC | _NOCTTY
    try:
        fd = os.open(name, flags, 0o666, dir_fd=dir_fd)
    except FileExistsError:
        raise FsError("fs.exists", "A file or folder of this name exists already", path=shown) from None
    except OSError as exc:
        raise FsError("fs.permission", f"The file cannot be created ({exc.strerror})", path=shown) from None
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
    except OSError as exc:
        os.close(fd)
        with suppress(OSError):
            os.unlink(name, dir_fd=dir_fd)
        raise FsError("fs.unreadable", f"The file cannot be written ({exc.strerror})", path=shown) from None
    os.close(fd)


def entry(dir_fd: int, name: str, shown: str, *, root: str = "", parts: object = None) -> tuple[str, int]:
    """``(kind, nlink)`` of one entry of an open folder, not following a link (kind ``link`` for one)."""
    del root, parts
    try:
        st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        raise FsError("fs.not_found", "No file or folder of this name", path=shown) from None
    mode = st.st_mode
    kind = (
        "link"
        if stat.S_ISLNK(mode)
        else "dir"
        if stat.S_ISDIR(mode)
        else "file"
        if stat.S_ISREG(mode)
        else "other"
    )
    return kind, st.st_nlink


def _libc_function(name: str) -> Any:
    with suppress(OSError, AttributeError):
        function = getattr(ctypes.CDLL(None, use_errno=True), name)
        function.restype = ctypes.c_int
        return function
    return None


def _rename_call(src_fd: int, src: str, dst_fd: int, dst: str) -> int | None:
    """The no-replace rename of this system: 0, an errno, or ``None`` when none is available."""
    args = (ctypes.c_int(src_fd), os.fsencode(src), ctypes.c_int(dst_fd), os.fsencode(dst))
    if _DARWIN:
        function = _libc_function("renameatx_np")
        if function is None:
            return None
        return 0 if function(*args, ctypes.c_uint(_RENAME_EXCL)) == 0 else ctypes.get_errno()
    if not _LINUX:
        return None
    function = _libc_function("renameat2")
    if function is not None:
        return 0 if function(*args, ctypes.c_uint(_RENAME_NOREPLACE)) == 0 else ctypes.get_errno()
    import platform

    number = _SYS_RENAMEAT2.get(platform.machine())
    syscall = _libc_function("syscall")
    if number is None or syscall is None:
        return None
    result = syscall(ctypes.c_long(number), *args, ctypes.c_uint(_RENAME_NOREPLACE))
    return 0 if result == 0 else ctypes.get_errno()


def rename(
    src_fd: int,
    src: str,
    dst_fd: int,
    dst: str,
    *,
    kind: str,
    shown: str,
    root: str = "",
    parts: object = None,
) -> None:
    """Rename ``src`` in the open folder ``src_fd`` to ``dst`` in ``dst_fd``, never replacing an entry.

    ``renameat2(RENAME_NOREPLACE)`` on Linux, ``renameatx_np(RENAME_EXCL)`` on macOS. Without either (or a
    file system that refuses the flag), a file is hard-linked to the new name (which fails if it exists) and
    its old name removed; a folder then cannot be renamed (``fs.unsupported_type``).
    """
    del root, parts
    code = _rename_call(src_fd, src, dst_fd, dst)
    if code == 0:
        return
    if code in (errno.EEXIST, errno.ENOTEMPTY):
        raise FsError("fs.exists", "A file or folder of the new name exists already", path=shown)
    if code == errno.ENOENT:
        raise FsError("fs.not_found", "No file or folder of this name", path=shown)
    if code == errno.EXDEV:
        raise FsError("fs.cross_device", "The new name is on another file system", path=shown)
    if code is not None and code not in (errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP):
        raise FsError("fs.permission", f"Cannot be renamed ({os.strerror(code)})", path=shown)
    if kind != "file":
        raise FsError(
            "fs.unsupported_type", "This system cannot rename a folder without replacing", path=shown
        )
    try:
        os.link(src, dst, src_dir_fd=src_fd, dst_dir_fd=dst_fd, follow_symlinks=False)
    except FileExistsError:
        raise FsError("fs.exists", "A file or folder of the new name exists already", path=shown) from None
    except OSError as exc:
        raise FsError("fs.permission", f"Cannot be renamed ({exc.strerror})", path=shown) from None
    os.unlink(src, dir_fd=src_fd)
