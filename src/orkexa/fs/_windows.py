"""The Windows side of :mod:`orkexa.fs`: handles, one component at a time, with ``ctypes`` only.

Windows has no ``openat`` in its documented Win32 API, so the walk works on handles that pin the path instead:

1. The root is opened with ``FILE_FLAG_OPEN_REPARSE_POINT`` (a junction or symbolic link that replaced it is
   opened itself, never its target); its volume serial number and file index must be the ones checked when the
   root was created, and its final path (``GetFinalPathNameByHandleW``) must still be the canonical root.
2. Every component is opened as ``\\\\?\\<root>\\<a>\\<b>`` with ``FILE_FLAG_OPEN_REPARSE_POINT |
   FILE_FLAG_BACKUP_SEMANTICS`` and a share mode **without** ``FILE_SHARE_DELETE``. While a folder's handle is
   open, nobody can rename, delete or replace that folder, so the next component's path resolves through
   exactly the folders already checked. ``\\\\?\\`` turns off Win32 name rewriting (trailing dots, device
   names).
3. A component that is a name-surrogate reparse point (symbolic link, junction, volume mount point) is refused
   (``fs.link``); cloud placeholders (OneDrive files on demand) are refused unless the root allows them
   (``fs.cloud_file``); other reparse points are refused as links too, except the data-preserving tags of
   deduplication and WOF compression.
4. The final path of every opened handle (``FILE_NAME_NORMALIZED``, which also expands 8.3 short names) must
   be the root plus exactly the names asked for, compared case-folded: this is the binding containment check,
   on the handle that is then read (``ReadFile``). It is not repeated with a path string.

``GetFileInformationByHandle`` gives the identity, the size and ``nNumberOfLinks`` (hard links). Roots on
network drives (``GetDriveTypeW`` is ``DRIVE_REMOTE``) or whose final path is a UNC path are refused
(``fs.network_path``). Nothing here needs administrator rights.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import os
import stat
import sys
import unicodedata

from ._names import FsError, fold, reparse_problem
from ._types import Info

assert sys.platform == "win32"  # mypy checks the rest of this module only for Windows

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)

_GENERIC_READ = 0x80000000
_FILE_LIST_DIRECTORY = 0x0001
_FILE_READ_ATTRIBUTES = 0x0080
_SYNCHRONIZE = 0x00100000
_SHARE = 0x1 | 0x2  # FILE_SHARE_READ | FILE_SHARE_WRITE, never FILE_SHARE_DELETE
_OPEN_EXISTING = 3
_BACKUP_SEMANTICS = 0x02000000
_OPEN_REPARSE_POINT = 0x00200000
_ATTRIBUTE_DIRECTORY = 0x10
_ATTRIBUTE_REPARSE_POINT = 0x400
_FILE_ATTRIBUTE_TAG_INFO_CLASS = 9
_FILE_TYPE_DISK = 1
_DRIVE_NO_ROOT_DIR = 1
_DRIVE_REMOTE = 4
_INVALID_HANDLE = ctypes.c_void_p(-1).value

_ERROR_NOT_FOUND = {2, 3, 161, 267}  # file or path not found, bad path name, the directory name is invalid
_ERROR_DENIED = {5, 1314}
_ERROR_IN_USE = {32, 33}
_ERROR_INVALID_NAME = {123}
_ERROR_CLOUD = {362, 380, 381, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404}


class _FILETIME(ctypes.Structure):
    _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]


class _HandleInfo(ctypes.Structure):
    _fields_ = [
        ("dwFileAttributes", wintypes.DWORD),
        ("ftCreationTime", _FILETIME),
        ("ftLastAccessTime", _FILETIME),
        ("ftLastWriteTime", _FILETIME),
        ("dwVolumeSerialNumber", wintypes.DWORD),
        ("nFileSizeHigh", wintypes.DWORD),
        ("nFileSizeLow", wintypes.DWORD),
        ("nNumberOfLinks", wintypes.DWORD),
        ("nFileIndexHigh", wintypes.DWORD),
        ("nFileIndexLow", wintypes.DWORD),
    ]


class _TagInfo(ctypes.Structure):
    _fields_ = [("FileAttributes", wintypes.DWORD), ("ReparseTag", wintypes.DWORD)]


_k32.CreateFileW.restype = wintypes.HANDLE
_k32.CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
]
_k32.CloseHandle.argtypes = [wintypes.HANDLE]
_k32.CloseHandle.restype = wintypes.BOOL
_k32.GetFinalPathNameByHandleW.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
_k32.GetFinalPathNameByHandleW.restype = wintypes.DWORD
_k32.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(_HandleInfo)]
_k32.GetFileInformationByHandle.restype = wintypes.BOOL
_k32.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
_k32.GetFileInformationByHandleEx.restype = wintypes.BOOL
_k32.GetFileType.argtypes = [wintypes.HANDLE]
_k32.GetFileType.restype = wintypes.DWORD
_k32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
_k32.GetDriveTypeW.restype = wintypes.UINT
_k32.ReadFile.argtypes = [
    wintypes.HANDLE,
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    wintypes.LPVOID,
]
_k32.ReadFile.restype = wintypes.BOOL


def _long(path: str) -> str:
    return "\\\\?\\" + path


def _create(path: str, *, directory: bool, reparse: bool = True) -> tuple[int | None, int]:
    """``CreateFileW`` of an absolute path: ``(handle, 0)`` or ``(None, last error)``."""
    access = (_FILE_LIST_DIRECTORY | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE) if directory else _GENERIC_READ
    flags = _BACKUP_SEMANTICS | (_OPEN_REPARSE_POINT if reparse else 0)
    handle = _k32.CreateFileW(_long(path), access, _SHARE, None, _OPEN_EXISTING, flags, None)
    if handle is None or handle == _INVALID_HANDLE:
        return None, ctypes.get_last_error()
    return int(handle), 0


def _close(handle: int) -> None:
    _k32.CloseHandle(handle)


def _final(handle: int) -> str:
    """The final path of an open handle (normalized long names, drive letter), without ``\\\\?\\``."""
    size = 1024
    while True:
        buffer = ctypes.create_unicode_buffer(size)
        length = _k32.GetFinalPathNameByHandleW(handle, buffer, size, 0)
        if length == 0:
            raise FsError("fs.unreadable", "The final path of a handle is not available")
        if length < size:
            text = buffer.value
            break
        size = length + 1
    if text.startswith("\\\\?\\UNC\\"):
        return "\\\\" + text[8:]
    return text[4:] if text.startswith("\\\\?\\") else text


def _tag(handle: int) -> tuple[int, int]:
    info = _TagInfo()
    if not _k32.GetFileInformationByHandleEx(
        handle, _FILE_ATTRIBUTE_TAG_INFO_CLASS, ctypes.byref(info), ctypes.sizeof(info)
    ):
        raise FsError("fs.unreadable", "The attributes of a handle are not available")
    return int(info.FileAttributes), int(info.ReparseTag)


def _info(handle: int) -> Info:
    data = _HandleInfo()
    if not _k32.GetFileInformationByHandle(handle, ctypes.byref(data)):
        raise FsError("fs.unreadable", "The file information of a handle is not available")
    if data.dwFileAttributes & _ATTRIBUTE_DIRECTORY:
        kind = "dir"
    elif _k32.GetFileType(handle) == _FILE_TYPE_DISK:
        kind = "file"
    else:
        kind = "other"
    written = (data.ftLastWriteTime.high << 32) | data.ftLastWriteTime.low
    return Info(
        kind=kind,
        dev=int(data.dwVolumeSerialNumber),
        ino=(int(data.nFileIndexHigh) << 32) | int(data.nFileIndexLow),
        size=(int(data.nFileSizeHigh) << 32) | int(data.nFileSizeLow),
        nlink=int(data.nNumberOfLinks),
        mtime_ns=(written - 116444736000000000) * 100,
    )


def _drive_type(path: str) -> int:
    return int(_k32.GetDriveTypeW(path[:3]))


def _error(code: int, shown: str | None) -> FsError:
    if code in _ERROR_NOT_FOUND:
        return FsError("fs.not_found", "No file or folder of this name", path=shown)
    if code in _ERROR_DENIED:
        return FsError("fs.permission", "The operating system denied access", path=shown)
    if code in _ERROR_IN_USE:
        return FsError("fs.unreadable", "The file is in use by another program", path=shown)
    if code in _ERROR_INVALID_NAME:
        return FsError("fs.invalid_path", "Windows refused the name", path=shown)
    if code in _ERROR_CLOUD:
        return FsError("fs.cloud_file", "A cloud file that is not available on this computer", path=shown)
    return FsError("fs.unreadable", f"Cannot be opened (Windows error {code})", path=shown)


def stat_path(path: str) -> Info | None:
    """The identity of a folder named by a trusted path (links followed); ``None`` if it cannot be opened."""
    handle, _ = _create(os.path.abspath(path), directory=True, reparse=False)
    if handle is None:
        return None
    try:
        return _info(handle)
    except FsError:
        return None
    finally:
        _close(handle)


def canonical(path: str) -> tuple[str, Info]:
    """The final path of a local folder (links resolved, 8.3 names expanded) and its identity."""
    text = os.path.abspath(path)
    kind = _drive_type(text)
    if kind == _DRIVE_REMOTE:
        raise FsError("fs.network_path", "A network drive cannot be a root; copy the files to this computer")
    if kind == _DRIVE_NO_ROOT_DIR:
        raise FsError("fs.not_found", "The drive of the root does not exist")
    handle, error = _create(text, directory=True, reparse=False)
    if handle is None:
        if error in _ERROR_NOT_FOUND:
            raise FsError("fs.not_found", "The root folder does not exist")
        raise FsError("fs.permission", f"The root folder cannot be opened (Windows error {error})")
    try:
        final = _final(handle)
        info = _info(handle)
    finally:
        _close(handle)
    if final.startswith("\\\\") or _drive_type(final) == _DRIVE_REMOTE:
        raise FsError("fs.network_path", "The root is on a network share; copy the files to this computer")
    if info.kind != "dir":
        raise FsError("fs.not_a_directory", "The root is not a folder")
    return final, info


def _same(path: str, expected: str) -> bool:
    return fold(path) == fold(expected)


def _open_root(path: str, identity: tuple[int, int], allow_cloud: bool) -> tuple[int, Info]:
    handle, _ = _create(path, directory=True)
    if handle is None:
        raise FsError("fs.root_changed", "The root folder is gone, moved or replaced")
    try:
        attributes, tag = _tag(handle)
        info = _info(handle)
        if (
            reparse_problem(attributes, tag, allow_cloud=allow_cloud) is not None
            or (info.dev, info.ino) != identity
            or not _same(_final(handle), path)
        ):
            raise FsError("fs.root_changed", "The root folder is not the folder that was checked")
    except BaseException:
        _close(handle)
        raise
    return handle, info


def _listing(path: str) -> list[os.DirEntry[str]]:
    with os.scandir(_long(path)) as found:
        return list(found)


def _resolve(base: str, done: list[str], name: str, directory: bool, shown: str) -> tuple[int, str]:
    """Open ``name`` below the folder ``base\\done``, or else the one entry whose NFC form is ``name``."""
    folder = "\\".join([base, *done])
    handle, error = _create(folder + "\\" + name, directory=directory)
    if handle is not None:
        return handle, name
    if error not in _ERROR_NOT_FOUND or name.isascii():
        raise _error(error, shown)
    found = [
        entry.name
        for entry in _listing(folder)
        if unicodedata.normalize("NFC", entry.name) == name and entry.name != name
    ]
    if len(found) > 1:
        raise FsError("fs.ambiguous_name", "Several entries have this name after normalization", path=shown)
    if not found:
        raise _error(error, shown)
    handle, error = _create(folder + "\\" + found[0], directory=directory)
    if handle is None:
        raise _error(error, shown)
    return handle, found[0]


@contextmanager
def walk(
    root: str,
    identity: tuple[int, int],
    parts: tuple[str, ...],
    *,
    final: str,
    on_node: Callable[[int, Info, tuple[str, ...]], None],
    shown: str,
    allow_cloud: bool = False,
) -> Iterator[tuple[int, Info]]:
    """Open the root and then ``parts`` one by one; yield the last handle and its :class:`Info`.

    Same contract as ``_posix.walk``. Every handle stays open (without ``FILE_SHARE_DELETE``) until the block
    ends.
    """
    opened: list[int] = []
    try:
        handle, root_info = _open_root(root, identity, allow_cloud)
        opened.append(handle)
        on_node(-1, root_info, ())
        info = root_info
        actual: list[str] = []
        for index, name in enumerate(parts):
            last = index == len(parts) - 1
            directory = not last or final == "dir"
            handle, stored = _resolve(root, actual, name, directory, shown)
            opened.append(handle)
            actual.append(stored)
            attributes, tag = _tag(handle)
            problem = reparse_problem(attributes, tag, allow_cloud=allow_cloud)
            if problem == "fs.link":
                raise FsError(
                    "fs.link", "A link, junction or mount point is on the way; it is not followed", path=shown
                )
            if problem is not None:
                raise FsError(problem, "A cloud placeholder (files on demand) is on the way", path=shown)
            info = _info(handle)
            if not _same(_final(handle), "\\".join([root, *actual])):
                raise FsError("fs.outside_root", "The opened path is not below the root", path=shown)
            if info.dev != root_info.dev:
                raise FsError("fs.cross_device", "The path crosses into another volume", path=shown)
            if not last and info.kind != "dir":
                raise FsError("fs.not_a_directory", "A folder was expected", path=shown)
            if info.kind == "other":
                raise FsError("fs.not_regular_file", "Not a regular file (a device)", path=shown)
            if last and info.kind == "file" and attributes & _ATTRIBUTE_REPARSE_POINT:
                # A data-preserving reparse point (dedup, WOF, an allowed cloud file): read through a handle
                # that the file system filter serves, and make sure it is the very file just checked.
                plain, error = _create("\\".join([root, *actual]), directory=False, reparse=False)
                if plain is None:
                    raise _error(error, shown)
                opened.append(plain)
                again = _info(plain)
                if (again.dev, again.ino) != (info.dev, info.ino):
                    raise FsError("fs.outside_root", "The file changed while it was opened", path=shown)
            on_node(index, info, tuple(actual))
        if final == "file" and info.kind == "dir":
            raise FsError("fs.is_a_directory", "A file was expected, this is a folder", path=shown)
        if final == "dir" and info.kind != "dir":
            raise FsError("fs.not_a_directory", "A folder was expected", path=shown)
        yield opened[-1], info
    finally:
        for handle in reversed(opened):
            _close(handle)


def listing(handle: int, root: str, names: tuple[str, ...]) -> Iterator[tuple[str, str, int]]:
    """``(name, kind, size)`` of the entries of the folder ``root\\names``, which :func:`walk` holds open
    (``handle``), so the path names that very folder. Links are reported as ``link``, never followed."""
    del handle
    for entry in _listing("\\".join([root, *names])):
        try:
            st = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        attributes = getattr(st, "st_file_attributes", 0)
        tag = getattr(st, "st_reparse_tag", 0)
        if (
            attributes & _ATTRIBUTE_REPARSE_POINT
            and reparse_problem(attributes, tag, allow_cloud=True) == "fs.link"
        ):
            kind = "link"
        elif stat.S_ISDIR(st.st_mode):
            kind = "dir"
        elif stat.S_ISREG(st.st_mode):
            kind = "file"
        else:
            kind = "other"
        yield entry.name, kind, st.st_size if kind == "file" else 0


def read(handle: int, limit: int, cancelled: Callable[[], bool], shown: str) -> bytes:
    """At most ``limit + 1`` bytes of an open file, in chunks; stops when ``cancelled()``."""
    chunks: list[bytes] = []
    total = 0
    while total <= limit:
        if cancelled():
            raise FsError("fs.revoked", "The root was revoked while the file was read", path=shown)
        size = min(1 << 20, limit + 1 - total)
        buffer = ctypes.create_string_buffer(size)
        got = wintypes.DWORD(0)
        if not _k32.ReadFile(handle, buffer, size, ctypes.byref(got), None):
            raise FsError(
                "fs.unreadable",
                f"The file cannot be read (Windows error {ctypes.get_last_error()})",
                path=shown,
            )
        if got.value == 0:
            break
        chunks.append(buffer.raw[: got.value])
        total += got.value
    return b"".join(chunks)


# ---- creating and renaming ---------------------------------------------------------------------------------

_GENERIC_WRITE = 0x40000000
_CREATE_NEW = 1
_ATTRIBUTE_NORMAL = 0x80
_ERROR_EXISTS = {80, 183}

_k32.WriteFile.argtypes = [
    wintypes.HANDLE,
    wintypes.LPCVOID,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    wintypes.LPVOID,
]
_k32.WriteFile.restype = wintypes.BOOL
_k32.MoveFileExW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
_k32.MoveFileExW.restype = wintypes.BOOL
_k32.DeleteFileW.argtypes = [wintypes.LPCWSTR]
_k32.DeleteFileW.restype = wintypes.BOOL


def create(handle: int, root: str, parts: tuple[str, ...], name: str, data: bytes, shown: str) -> None:
    """Create ``name`` in the folder ``root\\parts`` (held open as ``handle``) with ``CREATE_NEW``.

    ``CREATE_NEW`` fails on any existing entry, a link or junction included; the new handle's final path must
    be the folder plus ``name``.
    """
    del handle
    path = "\\".join([root, *parts, name])
    created = _k32.CreateFileW(
        _long(path), _GENERIC_WRITE, 0, None, _CREATE_NEW, _ATTRIBUTE_NORMAL | _OPEN_REPARSE_POINT, None
    )
    if created is None or created == _INVALID_HANDLE:
        error = ctypes.get_last_error()
        if error in _ERROR_EXISTS:
            raise FsError("fs.exists", "A file or folder of this name exists already", path=shown)
        raise _error(error, shown)
    new = int(created)
    try:
        if not _same(_final(new), path):
            raise FsError("fs.outside_root", "The new file is not below the root", path=shown)
        view = memoryview(data)
        while view:
            size = min(len(view), 1 << 20)
            done = wintypes.DWORD(0)
            chunk = ctypes.create_string_buffer(bytes(view[:size]), size)
            if not _k32.WriteFile(new, chunk, size, ctypes.byref(done), None):
                raise FsError("fs.unreadable", "The file cannot be written", path=shown)
            view = view[done.value :]
    except BaseException:
        _close(new)
        _k32.DeleteFileW(_long(path))
        raise
    _close(new)


def entry(handle: int, name: str, shown: str, *, root: str, parts: tuple[str, ...]) -> tuple[str, int]:
    """``(kind, nlink)`` of one entry of the folder ``root\\parts`` (``link`` for any reparse point)."""
    del handle
    path = "\\".join([root, *parts, name])
    opened, error = _create(path, directory=True)
    if opened is None:
        raise _error(error, shown)
    try:
        attributes, tag = _tag(opened)
        info = _info(opened)
    finally:
        _close(opened)
    if reparse_problem(attributes, tag, allow_cloud=True) is not None:
        return "link", info.nlink
    return info.kind, info.nlink


def rename(
    src_handle: int,
    src: str,
    dst_handle: int,
    dst: str,
    *,
    kind: str,
    shown: str,
    root: str = "",
    parts: object = None,
) -> None:
    """Rename ``src`` to ``dst`` (``parts`` is the pair of folder names) with ``MoveFileExW`` and no flags.

    Without ``MOVEFILE_REPLACE_EXISTING`` an existing entry is never replaced, and without
    ``MOVEFILE_COPY_ALLOWED`` nothing is copied to another volume. Both folders are held open by :func:`walk`.
    """
    del src_handle, dst_handle, kind
    src_parts, dst_parts = parts  # type: ignore[misc]
    old = "\\".join([root, *src_parts, src])
    new = "\\".join([root, *dst_parts, dst])
    if not _k32.MoveFileExW(_long(old), _long(new), 0):
        error = ctypes.get_last_error()
        if error in _ERROR_EXISTS:
            raise FsError("fs.exists", "A file or folder of the new name exists already", path=shown)
        if error == 17:  # ERROR_NOT_SAME_DEVICE
            raise FsError("fs.cross_device", "The new name is on another volume", path=shown)
        raise _error(error, shown)
