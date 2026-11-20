"""The POSIX side of :mod:`orkexa.fs`: directory file descriptors, one component at a time.

Every component is opened relative to the descriptor of its folder with ``O_NOFOLLOW``, so a symbolic link is
never followed, wherever it is on the way; folders are opened with ``O_DIRECTORY`` and files with
``O_NONBLOCK`` (a FIFO cannot block the open) and ``O_NOCTTY``. On Linux the component is opened with
``openat2`` and ``RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_XDEV`` when the
kernel has it (5.6+, probed once per process), which also refuses a bind mount on the same file system;
otherwise ``openat`` with ``O_NOFOLLOW``, a comparison of ``st_dev`` with the root's and, on Linux, of the
mount id (``/proc/self/fdinfo``), so a bind mount is refused there too. Each descriptor is checked with
``fstat`` and the file is read through that same descriptor: no path is resolved twice. The last component
of a file is opened with ``O_PATH`` on Linux (which opens nothing) and reopened through ``/proc/self/fd``
only once ``fstat`` shows a regular file or a folder; elsewhere (macOS, Linux without ``/proc``) it is looked
at with ``fstatat`` first. Either way a FIFO, socket or device is refused without being opened.
On macOS ``fcntl(F_GETPATH)`` of every opened descriptor must lie below the root's as a second check
(case-folded: APFS is usually case-insensitive), and the name the file system stored (``fgetattrlist``) must
fold to the name asked for; on Linux the same holds on every file system not known to compare names byte for
byte (FAT, exFAT, NTFS, SMB, NFS, 9p, FUSE, ISO 9660, ZFS ...) and in casefold folders.
Only the descriptor of the current folder stays open, so the depth of a path does not use up descriptors.
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

from ._names import FsError, caseless_file_system_type, fold, path_parts
from ._types import Info

_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_NOCTTY = getattr(os, "O_NOCTTY", 0)
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | _NOCTTY | _CLOEXEC
_ANY_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | _NOCTTY | _CLOEXEC
_DARWIN = sys.platform == "darwin"
_LINUX = sys.platform.startswith("linux")
_F_GETPATH = 50  # macOS <sys/fcntl.h>
_MAXPATHLEN = 1024  # macOS <sys/param.h>, the size F_GETPATH writes (with the NUL)
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


#: ``usable``: ``None`` until probed, then whether ``openat2`` works here. Tests set it to ``False`` to run
#: the fallback.
_openat2_state: dict[str, Any] = {"usable": None if _LINUX else False, "syscall": None}


def _call_openat2(dir_fd: int, name: str, flags: int) -> int:
    """``openat2`` of one component: a descriptor, or ``-errno``."""
    syscall = _openat2_state["syscall"]
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
    return int(result) if result >= 0 else -ctypes.get_errno()


def _probe_openat2() -> bool:
    """Whether this kernel (and its seccomp filter) offers ``openat2``: decided once, on the folder ``/``.

    Later errors of ``openat2`` are errors of the path, never a reason to switch to the fallback.
    """
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        syscall = libc.syscall
        syscall.restype = ctypes.c_long
    except (OSError, AttributeError):
        return False
    _openat2_state["syscall"] = syscall
    fd = os.open("/", _DIRECTORY_FLAGS)
    try:
        result = _call_openat2(fd, ".", _DIRECTORY_FLAGS)
    finally:
        os.close(fd)
    if result >= 0:
        os.close(result)
        return True
    return False  # ENOSYS (before 5.6), EPERM or ENOSYS from seccomp, E2BIG


def _openat2(dir_fd: int, name: str, flags: int) -> int | None:
    """``openat2`` of one component, or ``None`` when the kernel (or a seccomp filter) does not offer it."""
    usable = _openat2_state["usable"]
    if usable is None:
        usable = _openat2_state["usable"] = _probe_openat2()
    if not usable:
        return None
    result = _call_openat2(dir_fd, name, flags)
    if result >= 0:
        return result
    raise OSError(-result, os.strerror(-result))


def openat2_available() -> bool:
    """Whether components are opened with ``openat2`` (Linux 5.6+ without a seccomp filter against it)."""
    if not _LINUX:
        return False
    if _openat2_state["usable"] is None:
        _openat2_state["usable"] = _probe_openat2()
    return bool(_openat2_state["usable"])


def _mount_id(fd: int) -> int | None:
    """The mount id of an open descriptor (Linux 3.15+, ``/proc/self/fdinfo``), or ``None``."""
    try:
        with open(f"/proc/self/fdinfo/{fd}", "rb") as handle:
            for line in handle:
                if line.startswith(b"mnt_id:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


# ---- O_PATH (Linux): the last component is looked at before anything opens it ------------------------------

_PATH_FLAGS = getattr(os, "O_PATH", 0) | os.O_NOFOLLOW | _CLOEXEC
#: ``usable``: ``None`` until probed, then whether a descriptor opened with ``O_PATH`` can be reopened
#: through ``/proc/self/fd`` (Linux with ``/proc``). Tests set it to ``False`` for the ``fstatat`` precheck.
_reopen_state: dict[str, bool | None] = {"usable": None if _LINUX and hasattr(os, "O_PATH") else False}


def _probe_reopen() -> bool:
    try:
        fd = os.open("/", _PATH_FLAGS)
    except OSError:
        return False
    try:
        again = os.open(f"/proc/self/fd/{fd}", os.O_RDONLY | _CLOEXEC)
    except OSError:
        return False
    finally:
        os.close(fd)
    os.close(again)
    return True


def reopen_available() -> bool:
    """Whether the last component is opened with ``O_PATH`` first and reopened only once it is a regular
    file or a folder (Linux with ``/proc``); elsewhere ``fstatat`` looks at it before it is opened."""
    if _reopen_state["usable"] is None:
        _reopen_state["usable"] = _probe_reopen()
    return bool(_reopen_state["usable"])


def _reopen(path_fd: int, flags: int, shown: str) -> tuple[int, os.stat_result]:
    """Open the file an ``O_PATH`` descriptor names, once ``fstat`` shows a regular file or a folder.

    An ``O_PATH`` descriptor opens nothing: no FIFO, socket or device driver is touched by it. The reopen goes
    through ``/proc/self/fd``, which names that very inode, so no name is resolved again (review N7).
    """
    try:
        before = os.fstat(path_fd)
        if stat.S_ISLNK(before.st_mode):
            raise FsError("fs.link", "A symbolic link is on the way; links are not followed", path=shown)
        if not (stat.S_ISREG(before.st_mode) or stat.S_ISDIR(before.st_mode)):
            raise FsError("fs.not_regular_file", "Not a regular file (a FIFO, socket or device)", path=shown)
        try:
            return os.open(f"/proc/self/fd/{path_fd}", flags & ~os.O_NOFOLLOW), before
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EPERM):
                raise FsError("fs.permission", "The operating system denied access", path=shown) from None
            raise FsError("fs.unreadable", f"Cannot be opened ({exc.strerror})", path=shown) from None
    finally:
        os.close(path_fd)


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

    raw = fcntl.fcntl(fd, _F_GETPATH, b"\0" * _MAXPATHLEN)  # Python allows at most 1024 bytes here
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


#: Most entries read from one folder to resolve a name (NFD fallback, name check); more refuse the path.
MAX_SCAN = 100_000


def names(dir_fd: int, shown: str = "") -> Iterator[str]:
    """The entry names of an open folder, at most :data:`MAX_SCAN` (then ``fs.too_many_entries``)."""
    with os.scandir(dir_fd) as entries:
        for count, item in enumerate(entries, 1):
            if count > MAX_SCAN:
                raise FsError(
                    "fs.too_many_entries",
                    f"A folder on the way has more than {MAX_SCAN} entries",
                    path=shown,
                    reason="entries",
                )
            yield item.name


def _variant(dir_fd: int, name: str, shown: str) -> str:
    """The one entry whose NFC form is ``name`` but which is stored otherwise (NFD, e.g. by macOS)."""
    import unicodedata

    found = [
        item for item in names(dir_fd, shown) if unicodedata.normalize("NFC", item) == name and item != name
    ]
    if len(found) > 1:
        raise FsError("fs.ambiguous_name", "Several entries have this name after normalization", path=shown)
    if not found:
        raise FsError("fs.not_found", "No file or folder of this name", path=shown)
    return found[0]


def _resolve(
    dir_fd: int, name: str, flags: int, shown: str, *, precheck: bool = False
) -> tuple[int, str, os.stat_result | None]:
    """Open ``name`` exactly, or else the one entry whose NFC form is ``name``.

    With ``precheck`` the entry is looked at with ``fstatat`` before it is opened (:func:`_precheck`), and
    that result is returned to be compared with the opened descriptor.
    """
    if precheck:
        candidate = name
        before = _precheck(dir_fd, name, shown)
        if before is None and not name.isascii():
            candidate = _variant(dir_fd, name, shown)
            before = _precheck(dir_fd, candidate, shown)
        if before is None:
            raise FsError("fs.not_found", "No file or folder of this name", path=shown)
        return _child(dir_fd, candidate, flags, shown), candidate, before
    try:
        return _child(dir_fd, name, flags, shown), name, None
    except FsError as exc:
        if exc.code != "fs.not_found" or name.isascii():
            raise
    candidate = _variant(dir_fd, name, shown)
    return _child(dir_fd, candidate, flags, shown), candidate, None


# ---- the name the file system stored -----------------------------------------------------------------------

_FS_IOC_GETFLAGS = 0x80086601 if ctypes.sizeof(ctypes.c_long) == 8 else 0x80046601
_FS_CASEFOLD_FL = 0x40000000


class _StatFs(ctypes.Structure):
    _fields_ = [("f_type", ctypes.c_long), ("rest", ctypes.c_byte * 256)]


def _caseless_file_system(fd: int) -> bool:
    """Whether the (Linux) file system of ``fd`` may compare names without case (``fstatfs``): every file
    system not known to compare them byte for byte (``caseless_file_system_type`` in ``_names``)."""
    function = _libc_function("fstatfs")
    if function is None:
        return True  # unknown: check every name
    data = _StatFs()
    if function(ctypes.c_int(fd), ctypes.byref(data)) != 0:
        return True
    return caseless_file_system_type(data.f_type)


def _casefold_folder(fd: int) -> bool:
    """Whether a Linux folder has the casefold attribute (ext4, f2fs, tmpfs): names compare without case."""
    import fcntl

    buffer = bytearray(8)
    try:
        fcntl.ioctl(fd, _FS_IOC_GETFLAGS, buffer, True)
    except OSError:
        return False
    return bool(int.from_bytes(buffer[:4], sys.byteorder) & _FS_CASEFOLD_FL)


class _AttrList(ctypes.Structure):
    _fields_ = [
        ("bitmapcount", ctypes.c_ushort),
        ("reserved", ctypes.c_uint16),
        ("commonattr", ctypes.c_uint32),
        ("volattr", ctypes.c_uint32),
        ("dirattr", ctypes.c_uint32),
        ("fileattr", ctypes.c_uint32),
        ("forkattr", ctypes.c_uint32),
    ]


def _attr_name(fd: int) -> str | None:
    """The name a macOS file system stored for an open file (``fgetattrlist`` with ``ATTR_CMN_NAME``)."""
    function = _libc_function("fgetattrlist")
    if function is None:
        return None
    request = _AttrList(5, 0, 0x00000001, 0, 0, 0, 0)  # ATTR_BIT_MAP_COUNT, ATTR_CMN_NAME
    buffer = ctypes.create_string_buffer(4 + 8 + 1024 * 3 + 8)
    if function(ctypes.c_int(fd), ctypes.byref(request), buffer, ctypes.sizeof(buffer), ctypes.c_uint(0)):
        return None
    raw = buffer.raw
    offset = int.from_bytes(raw[4:8], sys.byteorder, signed=True)
    length = int.from_bytes(raw[8:12], sys.byteorder)
    start = 4 + offset
    if offset <= 0 or length < 1 or start + length > len(raw):
        return None
    return os.fsdecode(raw[start : start + length].split(b"\0", 1)[0])


def _scan_name(dir_fd: int, name: str, info: Info, shown: str) -> str | None:
    """The stored name of the entry of ``dir_fd`` that folds to ``name`` and is the file ``info``."""
    wanted = fold(name)
    for entry in names(dir_fd, shown):
        if fold(entry) != wanted:
            continue
        with suppress(OSError):
            st = os.stat(entry, dir_fd=dir_fd, follow_symlinks=False)
            if (st.st_dev, st.st_ino) == (info.dev, info.ino):
                return entry
    return None


def _stored_name(dir_fd: int, fd: int, name: str, info: Info, caseless: bool, shown: str) -> str | None:
    """The name the file system stored for the entry just opened as ``name``, where it may differ from
    ``name`` by more than Unicode normalization; ``None`` where names are compared byte for byte."""
    if _DARWIN:
        if info.kind != "dir" and info.nlink > 1:
            return None  # the stored name of a hard link may be another link's (and is refused anyway)
        stored = _attr_name(fd)
        return stored if stored is not None else (_scan_name(dir_fd, name, info, shown) or "")
    if _LINUX and (caseless or _casefold_folder(dir_fd)):
        return _scan_name(dir_fd, name, info, shown) or ""
    return None


# ---- the walk ----------------------------------------------------------------------------------------------


def _precheck(dir_fd: int, name: str, shown: str) -> os.stat_result | None:
    """``fstatat`` of the last component before it is opened: links and special files are never opened."""
    try:
        st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None  # not found under this spelling: the open (or the NFD fallback) decides
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM):
            raise FsError("fs.permission", "The operating system denied access", path=shown) from None
        raise FsError("fs.unreadable", f"Cannot be opened ({exc.strerror})", path=shown) from None
    if stat.S_ISLNK(st.st_mode):
        raise FsError("fs.link", "A symbolic link is on the way; links are not followed", path=shown)
    if not (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)):
        raise FsError("fs.not_regular_file", "Not a regular file (a FIFO, socket or device)", path=shown)
    return st


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
    A folder's descriptor is closed as soon as the next component is open; the last one when the block ends.
    """
    del allow_cloud
    current: int | None = None
    try:
        current, root_info = _open_root(root, identity)
        root_real = _getpath(current) if _DARWIN else None
        checked_mounts = _LINUX and not openat2_available()
        root_mount = _mount_id(current) if checked_mounts else None
        if checked_mounts and root_mount is None:
            raise FsError(
                "fs.cross_device",
                "Mounts below the root cannot be checked here (no openat2 and no /proc)",
                path=shown,
            )
        caseless = _LINUX and _caseless_file_system(current)
        on_node(-1, root_info, ())
        info = root_info
        actual: list[str] = []
        for index, name in enumerate(parts):
            last = index == len(parts) - 1
            directory = not last or final == "dir"
            flags = _DIRECTORY_FLAGS if directory else _FILE_FLAGS if final == "file" else _ANY_FLAGS
            before: os.stat_result | None
            if not directory and reopen_available():
                path_fd, stored, _ = _resolve(current, name, _PATH_FLAGS, shown)
                fd, before = _reopen(path_fd, flags, shown)
            else:
                fd, stored, before = _resolve(current, name, flags, shown, precheck=not directory)
            parent, current = current, fd
            try:
                info = _info(fd)
                if before is not None and (before.st_dev, before.st_ino) != (info.dev, info.ino):
                    raise FsError("fs.unreadable", "The entry changed while it was opened", path=shown)
                if info.dev != root_info.dev:
                    raise FsError("fs.cross_device", "The path crosses into another file system", path=shown)
                if checked_mounts and _mount_id(fd) != root_mount:
                    raise FsError("fs.cross_device", "The path crosses into another mount", path=shown)
                if info.kind == "other":
                    raise FsError(
                        "fs.not_regular_file", "Not a regular file (a FIFO, socket or device)", path=shown
                    )
                if root_real is not None and (info.kind == "dir" or info.nlink == 1):
                    real = _getpath(fd)
                    if real is None or not _below(real, root_real):
                        raise FsError("fs.outside_root", "The opened path is not below the root", path=shown)
                answered = _stored_name(parent, fd, stored, info, caseless, shown)
                if answered is not None and fold(answered) != fold(stored):
                    raise FsError(
                        "fs.ambiguous_name",
                        "The file system answered with an entry of another name",
                        path=shown,
                    )
            finally:
                os.close(parent)
            actual.append(stored)
            on_node(index, info, tuple(actual))
        if final == "file" and info.kind == "dir":
            raise FsError("fs.is_a_directory", "A file was expected, this is a folder", path=shown)
        if final == "dir" and info.kind != "dir":
            raise FsError("fs.not_a_directory", "A folder was expected", path=shown)
        yield current, info
    finally:
        if current is not None:
            with suppress(OSError):
                os.close(current)


def _below(path: str, root: str) -> bool:
    inner, outer = (
        [fold(part) for part in path.split("/") if part],
        [fold(part) for part in root.split("/") if part],
    )
    return len(inner) > len(outer) and inner[: len(outer)] == outer


def listing(dir_fd: int, root: str, parts: tuple[str, ...]) -> Iterator[tuple[str, str, int, int]]:
    """``(name, kind, size, nlink)`` of the entries of an open folder, one at a time; no link followed."""
    del root, parts  # the descriptor names the folder
    with os.scandir(dir_fd) as entries:
        for item in entries:
            try:
                st = item.stat(follow_symlinks=False)
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
            yield item.name, kind, st.st_size if kind == "file" else 0, st.st_nlink


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
    its old name removed, after checking that the old name still is that very file; if anything fails, the
    new name is removed again, so no second link stays behind. A folder then cannot be renamed
    (``fs.unsupported_type``).
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
    try:
        old = os.stat(src, dir_fd=src_fd, follow_symlinks=False)
        new = os.stat(dst, dir_fd=dst_fd, follow_symlinks=False)
        if (old.st_dev, old.st_ino) != (new.st_dev, new.st_ino):
            raise FsError("fs.unreadable", "The file changed while it was renamed", path=shown)
        os.unlink(src, dir_fd=src_fd)
    except BaseException as exc:
        with suppress(OSError):
            os.unlink(dst, dir_fd=dst_fd)
        if isinstance(exc, FsError):
            raise
        if isinstance(exc, OSError):
            raise FsError("fs.permission", f"Cannot be renamed ({exc.strerror})", path=shown) from None
        raise
