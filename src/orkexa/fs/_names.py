"""Names, patterns and folders of :mod:`orkexa.fs`: everything that is decided from text alone.

Nothing here opens a file. The file-system checks (links, identities, the final path of a handle) are in
``_posix`` and ``_windows``; this module only refuses input that must never reach them and folds names for the
blocklist and the include and exclude patterns.
"""

from __future__ import annotations

from collections.abc import Iterable
from fnmatch import fnmatchcase
import os
import re
import unicodedata

from ..errors import OrkexaError

#: Longest relative path accepted, in UTF-8 bytes.
MAX_PATH_BYTES = 4096
#: Longest single component accepted, in UTF-8 bytes.
MAX_NAME_BYTES = 255
#: Most components of a relative path (and of a pattern). Bounds the work of every check and the handles one
#: walk holds on Windows.
MAX_DEPTH = 128

FS_ERROR_CODES: dict[str, str] = {
    "fs.invalid_path": "The path breaks a name rule (control character, NUL, ':', '\\', a reserved device "
    "name, a trailing dot or space, an 8.3 short name, a look-alike separator, bad UTF-8, too long, an empty "
    "or '.' component, or a Windows device path as root)",
    "fs.absolute_path": "A relative path was expected; this one is absolute, names a drive or UNC share, or "
    "starts with '~'",
    "fs.parent_reference": "The path contains '..'",
    "fs.link": "A symbolic link, junction, mount point or other name-surrogate reparse point is on the way; "
    "links are never followed",
    "fs.outside_root": "The opened file or folder is not below the root (says the final path of its handle)",
    "fs.cross_device": "The path crosses into another file system or mount below the root",
    "fs.hard_link": "The file has more than one hard link and hard links are not allowed for this root",
    "fs.not_regular_file": "Not a regular file (a FIFO, socket, device or other special file)",
    "fs.not_a_directory": "A folder was expected",
    "fs.is_a_directory": "A file was expected, this is a folder",
    "fs.not_found": "Nothing with this name exists below the root",
    "fs.ambiguous_name": "Several entries of the folder have this name after Unicode normalization, or the "
    "file system answered with an entry of another name",
    "fs.blocked": "The path is on the blocklist or inside a protected folder (such as the product's data)",
    "fs.excluded": "The path is excluded by the root's include and exclude patterns",
    "fs.too_large": "The file is larger than the size limit",
    "fs.too_many_entries": "The folder has more entries than the listing limit, or listing it took longer "
    "than the time limit (reason: entries, timeout)",
    "fs.protected_root": "This folder cannot be a root: a drive or file-system root, the home folder itself "
    "or a folder that contains it, a folder that holds home folders, a system folder, a blocked folder, or a "
    "protected folder (or a folder inside one of these)",
    "fs.network_path": "Network paths (UNC shares, also of this computer, network drives) cannot be a root",
    "fs.cloud_file": "A cloud placeholder (OneDrive files on demand); reading it could download it",
    "fs.root_changed": "The root is no longer the folder that was checked (moved, replaced or a link now)",
    "fs.revoked": "The root was revoked",
    "fs.read_only": "The root is read only (create it with writable=True to create or rename files)",
    "fs.exists": "A file or folder of the new name exists already; nothing is replaced",
    "fs.permission": "The operating system denied access",
    "fs.unreadable": "The file could not be read",
    "fs.unsupported_type": "No text extraction exists for this kind of file",
    "fs.not_browsable": "The folder browser neither lists nor opens this folder: a system folder, another "
    "user's home folder, a blocked folder, a protected folder, or a folder inside one of these",
    "fs.too_complex": "The text extraction exceeded a budget (reason: pages, chars, timeout, memory, limit, "
    "crash)",
}
"""Every :attr:`FsError.code`, with its meaning. Products map these codes, never message texts."""


class FsError(OrkexaError, PermissionError):
    """A path was refused, or a file could not be read, by :mod:`orkexa.fs`.

    ``code`` is one of :data:`FS_ERROR_CODES`. ``path`` is the relative path as given (or ``None``); messages
    never contain an absolute path. ``reason`` refines ``fs.too_complex`` (which budget) and a few other
    codes. It is a ``PermissionError``, so a generic ``except OSError`` still catches it.
    """

    def __init__(
        self, code: str, message: str, *, path: str | None = None, reason: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        self.reason = reason

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else self.code

    def __reduce__(self) -> tuple[object, ...]:  # picklable with its keyword arguments
        return (_rebuild, (self.code, str(self), self.path, self.reason))


def _rebuild(code: str, message: str, path: str | None, reason: str | None) -> FsError:
    return FsError(code, message, path=path, reason=reason)


# ---- relative paths ----------------------------------------------------------------------------------------

#: Windows device names, also with an extension (``NUL.txt``) and with superscript digits (``COM¹``).
_RESERVED = re.compile(
    r"^(con|prn|aux|nul|com[0-9¹²³]|lpt[0-9¹²³]|conin\$|conout\$|clock\$)(\..*)?$", re.IGNORECASE
)
#: Characters that look like a path separator but are none: fullwidth solidus and reverse solidus, division
#: slash, fraction slash, big solidus, set minus, small reverse solidus.
_LOOKALIKE_SEPARATORS = frozenset("\uff0f\uff3c\u2215\u2044\u29f8\u29f9\u2216\ufe68")
_DRIVE = re.compile(r"^[A-Za-z]:")
#: Bidirectional controls and zero-width characters: allowed in names (P0-2 Unicode decision), escaped for
#: display.
_INVISIBLE = frozenset(
    "\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2060\u2066\u2067\u2068\u2069\ufeff"
)


def is_short_name(name: str) -> bool:
    """Whether ``name`` has the shape of an 8.3 short name (``PROGRA~1``, ``SECRET~1.TXT``).

    Windows may answer to such a name with a file whose real (long) name is something else, so the blocklist
    and the patterns would compare the wrong name. Only the exact 8.3 shape is refused, so ``notes.txt.~1~``
    is fine.
    """
    base, dot, extension = name.rpartition(".")
    if not dot:
        base, extension = name, ""
    if len(extension) > 3 or len(base) > 8:
        return False
    head, tilde, number = base.rpartition("~")
    return bool(tilde) and 1 <= len(head) <= 6 and number.isdigit() and number.isascii() and "." not in head


def name_problem(name: str) -> str | None:
    """Why one component of a relative path is refused (the message), or ``None``. ``name`` is NFC already."""
    if name in ("", "."):
        return "A path component is empty or '.'"
    if len(name.encode("utf-8")) > MAX_NAME_BYTES:
        return f"A path component is longer than {MAX_NAME_BYTES} bytes"
    if ":" in name:
        return "A path component contains ':' (an alternate data stream or a drive)"
    if any(char in _LOOKALIKE_SEPARATORS for char in name):
        return "A path component contains a character that looks like a path separator"
    if name.endswith((".", " ")):
        return "A path component ends with a dot or a space"
    if _RESERVED.match(name):
        return "A path component is a reserved Windows device name"
    if is_short_name(name):
        return "A path component is an 8.3 short name"
    return None


def split_relative(path: str | bytes, *, allow_empty: bool = False) -> tuple[str, ...]:
    """The NFC components of a relative path, or :class:`FsError`.

    The rules are the same on every operating system (P0-2 rule 1), so a path that one system accepts is never
    a different file, or a device, on another: ``/``-separated, no ``\\``, no ``:``, no NUL or control
    characters, no empty, ``.`` or ``..`` component, no reserved Windows device name, no trailing dot or
    space, no 8.3 short name, no look-alike separator (fullwidth solidus, division slash), at most
    :data:`MAX_NAME_BYTES` per component and :data:`MAX_PATH_BYTES` in all. Bytes must be strict UTF-8 (an
    overlong encoding is refused).
    Nothing is decoded further: ``%2e%2e%2f`` is a name, not ``../``. ``""`` names the root itself when
    ``allow_empty``.
    """
    if isinstance(path, bytes):
        try:
            path = path.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise FsError("fs.invalid_path", "The path is not valid UTF-8") from None
    if not isinstance(path, str):
        raise TypeError("A path is a str (or UTF-8 bytes)")
    shown = path if len(path) <= MAX_PATH_BYTES else path[:64]
    try:
        path.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise FsError("fs.invalid_path", "The path is not valid Unicode", path=None) from None
    if any(unicodedata.category(char) == "Cc" for char in path):
        raise FsError("fs.invalid_path", "The path contains NUL or a control character", path=None)
    if path == "":
        if allow_empty:
            return ()
        raise FsError("fs.invalid_path", "The path is empty", path=shown)
    if path.startswith(("/", "\\")) or _DRIVE.match(path) or path == "~" or path.startswith("~/"):
        raise FsError("fs.absolute_path", "The path must be relative to the root", path=shown)
    text = unicodedata.normalize("NFC", path)
    if "\\" in text:
        raise FsError("fs.invalid_path", "The path contains '\\'; separate folders with '/'", path=shown)
    if len(text.encode("utf-8")) > MAX_PATH_BYTES:
        raise FsError("fs.invalid_path", f"The path is longer than {MAX_PATH_BYTES} bytes", path=shown)
    parts = tuple(text.split("/"))
    if len(parts) > MAX_DEPTH:
        raise FsError("fs.invalid_path", f"The path has more than {MAX_DEPTH} components", path=shown[:256])
    if ".." in parts:
        raise FsError("fs.parent_reference", "The path must not contain '..'", path=text)
    for part in parts:
        problem = name_problem(part)
        if problem is not None:
            raise FsError("fs.invalid_path", problem, path=text)
    return parts


#: Unicode Default_Ignorable_Code_Point ranges (DerivedCoreProperties.txt). Some file systems ignore them in
#: names (HFS+, the casefold of ext4 and f2fs), so ``.s\u200bsh`` may open ``.ssh`` there.
_IGNORABLE_RANGES = (
    (0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C), (0x115F, 0x1160), (0x17B4, 0x17B5),
    (0x180B, 0x180F), (0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x206F), (0x3164, 0x3164),
    (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF), (0xFFA0, 0xFFA0), (0xFFF0, 0xFFF8), (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A), (0xE0000, 0xE0FFF),
)  # fmt: skip
_IGNORABLE = re.compile("[" + "".join(f"\\U{low:08x}-\\U{high:08x}" for low, high in _IGNORABLE_RANGES) + "]")
#: Letters that case-insensitive file systems (the exFAT and NTFS upcase tables) equate although case folding
#: does not: dotless i is I there.
_FOLD_EXTRA = str.maketrans({"\u0131": "i"})


def fold(name: str) -> str:
    """``name`` for comparisons with the blocklist and the patterns.

    NFC, default-ignorable code points removed, case-folded, dotless i as i, NFC again: two names that any
    supported file system could treat as one fold to the same text (fails closed: the blocklist then covers
    both).
    """
    text = _IGNORABLE.sub("", unicodedata.normalize("NFC", name))
    return unicodedata.normalize("NFC", text.casefold().translate(_FOLD_EXTRA))


def display_name(name: str) -> str:
    """``name`` with bidirectional controls and zero-width characters escaped (``\\u202e``), for display."""
    return "".join(f"\\u{ord(char):04x}" if char in _INVISIBLE else char for char in name)


# ---- patterns ----------------------------------------------------------------------------------------------


State = frozenset[int]


class Pattern:
    """A glob over relative paths: ``*`` and ``?`` within one component, ``**`` for any number of components.

    A pattern is anchored at the root unless it starts with ``**/`` (:attr:`floating`). Comparison is NFC
    plus case-folded on every operating system (fails closed on case-insensitive file systems). For exclusions
    and the blocklist a pattern that matches a folder covers everything below it.

    Matching runs the pattern as a set of positions over the components, one pass from left to right: the
    work is at most (components of the path) x (components of the pattern) component comparisons, whatever
    the number of ``**``, so no path can make a check slow. A state (:meth:`start`, :meth:`step`) tells
    everything the pattern can still decide about the paths below a folder.
    """

    __slots__ = ("floating", "parts", "text")

    def __init__(self, text: str) -> None:
        if not isinstance(text, str) or not text or len(text) > MAX_PATH_BYTES:
            raise ValueError("A pattern is a non-empty string")
        cleaned = unicodedata.normalize("NFC", text).strip("/")
        if not cleaned or "\\" in cleaned or ".." in cleaned.split("/") or "\x00" in cleaned:
            raise ValueError(f"Invalid pattern {text!r}")
        parts = tuple(fold(part) for part in cleaned.split("/"))
        if len(parts) > MAX_DEPTH:
            raise ValueError(f"A pattern has at most {MAX_DEPTH} components")
        self.text = text
        self.parts = parts
        self.floating = parts[0] == "**"

    def __repr__(self) -> str:
        return f"Pattern({self.text!r})"

    def _closure(self, positions: Iterable[int]) -> State:
        found = set(positions)
        pending = list(found)
        while pending:
            position = pending.pop()
            if position < len(self.parts) and self.parts[position] == "**" and position + 1 not in found:
                found.add(position + 1)
                pending.append(position + 1)
        return frozenset(found)

    def start(self) -> State:
        """The state before the first component."""
        return self._closure((0,))

    def step(self, state: State, part: str) -> State:
        """The state after one more (folded) component."""
        following: set[int] = set()
        for position in state:
            if position >= len(self.parts):
                continue
            head = self.parts[position]
            if head == "**":
                following.add(position)
            elif fnmatchcase(part, head):
                following.add(position + 1)
        return self._closure(following)

    def done(self, state: State) -> bool:
        """Whether the path read so far matches the whole pattern."""
        return len(self.parts) in state

    def run(self, parts: Iterable[str], state: State | None = None) -> tuple[State, bool]:
        """The state after ``parts`` and whether the pattern matched after one of them (it *covers* them)."""
        current = self.start() if state is None else state
        covered = False
        for part in parts:
            if not current:
                break
            current = self.step(current, part)
            covered = covered or self.done(current)
        return current, covered

    def matches(self, parts: tuple[str, ...]) -> bool:
        """Whether the pattern matches exactly this path (folded components)."""
        state, _ = self.run(parts)
        return self.done(state)

    def covers(self, parts: tuple[str, ...]) -> bool:
        """Whether the pattern matches this path or one of its folders."""
        return self.run(parts)[1]


def patterns(values: Iterable[str]) -> tuple[Pattern, ...]:
    if isinstance(values, str):
        raise TypeError("Patterns are a list of strings, not one string")
    return tuple(Pattern(value) for value in values)


#: Names below a root that are never read, wherever they are (credentials, keys, keychains). Compared NFC plus
#: case-folded, so ``.SSH``, ``LIBRARY/KEYCHAINS`` and the NFD spelling of a name are blocked too.
DEFAULT_BLOCKED: tuple[str, ...] = (
    "**/.ssh",
    "**/.gnupg",
    "**/.aws",
    "**/.azure",
    "**/.kube",
    "**/.docker",
    "**/.password-store",
    "**/.netrc",
    "**/_netrc",
    "**/.pgpass",
    "**/.git-credentials",
    "**/.config/gcloud",
    "**/.config/gh",
    "**/.local/share/keyrings",
    "**/Library/Keychains",
    "**/Library/Cookies",
    "**/AppData/Roaming/Microsoft/Credentials",
    "**/AppData/Local/Microsoft/Credentials",
    "**/AppData/Roaming/Microsoft/Protect",
    "**/AppData/Roaming/Microsoft/Crypto",
    "**/AppData/Roaming/Microsoft/SystemCertificates",
)


# ---- folders that cannot be a root -------------------------------------------------------------------------

#: System folders of Linux and macOS (taken from nodestro-api's ``output_folder.py``, plus ``/root``, which
#: nodestro-runtime's access grants protect as well).
POSIX_SYSTEM_FOLDERS: tuple[str, ...] = (
    "/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/libx32", "/opt", "/proc", "/run", "/sbin",
    "/snap", "/sys", "/usr", "/var", "/System", "/Library", "/Applications", "/private", "/cores", "/root",
)  # fmt: skip
#: Places below a system folder where people keep their own files (Fedora Atomic's /var/home, removable media,
#: macOS' temporary folders), as in ``output_folder.py``.
POSIX_ALLOWED_FOLDERS: tuple[str, ...] = (
    "/var/home", "/var/mnt", "/var/tmp", "/run/media", "/private/tmp", "/private/var/folders",
)  # fmt: skip
#: Folders that hold the home folders of all users: never a root (each user's files are below them).
POSIX_HOME_PARENTS: tuple[str, ...] = ("/home", "/Users", "/var/home", "/export/home", "/usr/home")
#: Environment variables that name Windows system folders (on any drive).
WINDOWS_SYSTEM_ENV: tuple[str, ...] = (
    "SystemRoot", "windir", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "ProgramData",
)  # fmt: skip
#: Windows system folders when the environment does not name them.
WINDOWS_SYSTEM_FOLDERS: tuple[str, ...] = (
    "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)", "C:\\ProgramData",
)  # fmt: skip


def system_folders() -> list[str]:
    """The system folders of this operating system: no root may be one of them or lie inside one."""
    if os.name == "nt":
        named = [os.environ.get(name, "") for name in WINDOWS_SYSTEM_ENV]
        return list(dict.fromkeys([*WINDOWS_SYSTEM_FOLDERS, *(value for value in named if value)]))
    return list(POSIX_SYSTEM_FOLDERS)


def home_parents() -> list[str]:
    """The folders that hold the home folders of all users on this operating system."""
    if os.name == "nt":
        drive = os.environ.get("SYSTEMDRIVE", "C:")
        profile = os.environ.get("USERPROFILE", "").rstrip("\\")
        named = [os.path.dirname(profile) if profile else "", os.environ.get("PUBLIC", "")]
        return list(dict.fromkeys([f"{drive}\\Users", *(value for value in named if value)]))
    return list(POSIX_HOME_PARENTS)


#: Magic numbers (``statfs.f_type``) of Linux file systems that always compare names byte for byte, apart from
#: folders with the casefold attribute (checked separately): ext2/3/4, Btrfs, XFS, tmpfs, F2FS, overlayfs,
#: ramfs, bcachefs, squashfs, EROFS, NILFS and ReiserFS. Every other file system (FAT, exFAT, NTFS, HFS+,
#: SMB, NFS, 9p, FUSE and virtiofs, ISO 9660, ZFS with ``casesensitivity``, JFS, unknown ones) is treated as
#: possibly case-insensitive: the stored name of each opened entry is checked (review N7).
CASE_SENSITIVE_FILE_SYSTEMS = frozenset(
    {
        0xEF53, 0x9123683E, 0x58465342, 0x01021994, 0xF2F52010, 0x794C7630, 0x858458F6, 0xCA451A4E,
        0x73717368, 0xE0F5E1E2, 0x3434, 0x52654973,
    }
)  # fmt: skip


def caseless_file_system_type(f_type: int) -> bool:
    """Whether names on a Linux file system of this ``statfs.f_type`` may compare without case (conservative:
    ``True`` unless the type is in :data:`CASE_SENSITIVE_FILE_SYSTEMS`)."""
    return (f_type & 0xFFFFFFFF) not in CASE_SENSITIVE_FILE_SYSTEMS


def allowed_folders() -> list[str]:
    """Folders inside a system folder that may still hold a root (temporary folders, removable media)."""
    return [] if os.name == "nt" else list(POSIX_ALLOWED_FOLDERS)


def path_parts(path: str) -> tuple[str, ...]:
    """The folded components of an absolute, canonical path, for comparing folders component by component."""
    text = path.replace("\\", "/") if os.name == "nt" else path
    return tuple(fold(part) for part in text.split("/") if part)


def inside(path: str, folder: str) -> bool:
    """Whether ``path`` is ``folder`` or below it (whole components, NFC plus case-folded)."""
    inner, outer = path_parts(path), path_parts(folder)
    return len(inner) >= len(outer) and inner[: len(outer)] == outer


# ---- Windows reparse points ----------------------------------------------------------------------------

_ATTRIBUTE_REPARSE_POINT = 0x400
_ATTRIBUTE_OFFLINE = 0x1000
_ATTRIBUTE_RECALL_ON_OPEN = 0x00040000
_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000
_NAME_SURROGATE = 0x20000000
#: Reparse tags that keep a file's data in place (no redirection): deduplication and WOF compression.
DATA_REPARSE_TAGS = frozenset({0x80000013, 0x80000017})


def is_cloud_tag(tag: int) -> bool:
    """``IO_REPARSE_TAG_CLOUD`` and its variants ``CLOUD_1`` to ``CLOUD_F`` (OneDrive files on demand)."""
    return (tag & 0xFFFF0FFF) == 0x9000001A


def reparse_problem(attributes: int, tag: int, *, allow_cloud: bool) -> str | None:
    """The :class:`FsError` code for a Windows file with these attributes and reparse tag, or ``None``.

    A cloud placeholder (a cloud tag, or the recall and offline attributes) is ``fs.cloud_file`` unless
    ``allow_cloud``. Every other reparse point is ``fs.link``: name surrogates (symbolic links, junctions,
    volume mount points) and unknown tags alike, except :data:`DATA_REPARSE_TAGS`.
    """
    recall = _ATTRIBUTE_RECALL_ON_DATA_ACCESS | _ATTRIBUTE_RECALL_ON_OPEN | _ATTRIBUTE_OFFLINE
    if (attributes & _ATTRIBUTE_REPARSE_POINT and is_cloud_tag(tag)) or attributes & recall:
        return None if allow_cloud else "fs.cloud_file"
    if not attributes & _ATTRIBUTE_REPARSE_POINT:
        return None
    if tag & _NAME_SURROGATE or tag not in DATA_REPARSE_TAGS:
        return "fs.link"
    return None


# ---- root paths --------------------------------------------------------------------------------------------

_WINDOWS_DEVICE = ("\\\\.\\", "\\??\\", "\\\\?\\globalroot", "\\\\?\\volume{", "\\\\?\\pipe")


def root_syntax(path: str) -> str:
    """``path`` as the root text to canonicalize, after the checks that need no file system.

    Refused here, on every operating system: empty or white space at either end, NUL or control characters,
    Windows device paths (``\\\\.\\``, ``\\??\\``, ``\\\\?\\GLOBALROOT``, ``\\\\?\\Volume{...}``) as
    ``fs.invalid_path``, and network paths (two leading separators of either kind: UNC ``\\\\host\\share``,
    also ``\\\\localhost\\C$`` and this computer's own name, and ``\\\\?\\UNC\\``) as ``fs.network_path``. A
    ``\\\\?\\C:\\x`` path is ``C:\\x``, as in nodestro-runtime's ``folder_parts``.
    """
    if not isinstance(path, str):
        path = os.fspath(path)
    if not path or path != path.strip():
        raise FsError("fs.invalid_path", "A root path must not be empty or start or end with white space")
    if any(unicodedata.category(char) == "Cc" for char in path):
        raise FsError("fs.invalid_path", "A root path contains NUL or a control character")
    lowered = path.casefold().replace("/", "\\")
    if lowered.startswith("\\\\?\\unc\\"):
        raise FsError("fs.network_path", "A network path cannot be a root; copy the files to this computer")
    if lowered.startswith(_WINDOWS_DEVICE):
        raise FsError("fs.invalid_path", "A Windows device path cannot be a root")
    if lowered.startswith("\\\\?\\"):
        rest = path[4:]
        if not re.match(r"^[A-Za-z]:[\\/]", rest):
            raise FsError("fs.invalid_path", "A Windows device path cannot be a root")
        path, lowered = rest, lowered[4:]
    if lowered.startswith("\\\\"):
        raise FsError(
            "fs.network_path",
            "A network path (UNC share, also of this computer) cannot be a root; use the local drive path",
        )
    if os.name == "nt":
        if not re.match(r"^[A-Za-z]:[\\/]", path):
            raise FsError("fs.invalid_path", "A root on Windows is an absolute path with a drive (C:\\...)")
        if ":" in path[2:]:
            raise FsError(
                "fs.invalid_path", "A root path contains ':' after the drive (an alternate data stream)"
            )
    elif not path.startswith("/"):
        raise FsError("fs.invalid_path", "A root path must be absolute")
    return path
