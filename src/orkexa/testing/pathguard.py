"""The attack catalog of the path guard, reusable against any file reader (Nodestro N-10, P0-2 d, W0-6).

:data:`CATALOG` lists every N-10 case of the P0-2 attack catalog (``N10-01`` to ``N10-23``; the cases 24 to 35
belong to N-22) and every gap of the W0-6 spike (``W06-1`` to ``W06-5``), with the platforms where it is
physically possible and the :mod:`orkexa.fs` codes that refuse it. Orkexa's own tests (``tests/fs``) run all
of them against :class:`orkexa.fs.SafeRoot`; a product runs the portable part against its own tools
(conformance, N-63)::

    from orkexa.testing import pathguard

    layout = pathguard.build_layout(tmp_path)
    findings = pathguard.check_reader(lambda path: my_read_tool(layout.root, path), layout)
    assert not findings, findings

``check_reader`` sends every name attack (:data:`NAME_ATTACKS`) and every link attack the layout could create
(:func:`link_attacks`) to ``read`` and reports each one that returned data, returned the planted canary or,
with ``code_of``, failed with an unexpected code. :func:`swap_race` replaces a file by a link to the outside
while ``read`` runs, many times. Nothing here needs pytest or imports another Orkexa module.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import sys
import threading

__all__ = [
    "BLOCKED_CANARY",
    "CANARY",
    "CATALOG",
    "NAME_ATTACKS",
    "AttackCase",
    "Finding",
    "Layout",
    "build_layout",
    "check_reader",
    "link_attacks",
    "make_junction",
    "swap_race",
]

#: The content of every file outside the root; a reader that returns it has been led outside.
CANARY = b"orkexa-pathguard-canary-outside-the-root"
#: The content of the blocked files inside the root (``.ssh/id_rsa``, ``Library/Keychains/...``).
BLOCKED_CANARY = b"orkexa-pathguard-canary-blocked-inside-the-root"

ALL = frozenset({"linux", "darwin", "win32"})
POSIX = frozenset({"linux", "darwin"})


@dataclass(frozen=True)
class AttackCase:
    """One case of the catalog.

    ``platforms`` are the systems where the case can physically occur (``linux``, ``darwin``, ``win32``);
    ``codes`` the :mod:`orkexa.fs` error codes that refuse it (empty for cases whose expectation is that a
    file *is* read, such as a Finder alias); ``tests`` the Orkexa tests that cover it.
    """

    id: str
    title: str
    platforms: frozenset[str]
    codes: frozenset[str]
    tests: tuple[str, ...] = ()


def _case(
    id: str, title: str, codes: Iterable[str], tests: Iterable[str], platforms: frozenset[str] = ALL
) -> AttackCase:
    return AttackCase(id, title, platforms, frozenset(codes), tuple(tests))


CATALOG: tuple[AttackCase, ...] = (
    _case(
        "N10-01",
        "'../', 'a/../../etc/passwd', URL-encoded %2e%2e%2f, overlong UTF-8",
        ["fs.parent_reference", "fs.not_found", "fs.invalid_path"],
        ["test_attack_catalog.py::test_name_attacks_are_refused"],
    ),
    _case(
        "N10-02",
        "Absolute paths: /etc/passwd, C:\\Windows\\win.ini, ~/.ssh/id_rsa",
        ["fs.absolute_path"],
        ["test_attack_catalog.py::test_name_attacks_are_refused", "test_safe_root.py::test_absolute_outside"],
    ),
    _case(
        "N10-03",
        "Symbolic link to the outside (file, folder), a link chain, a relative '../..' link",
        ["fs.link"],
        ["test_attack_catalog.py::test_link_attacks_are_refused", "test_safe_root.py::test_internal_links"],
    ),
    _case(
        "N10-04",
        "Swap race: a file replaced by a link to the outside while it is opened (10,000 times)",
        ["fs.link", "fs.not_found", "fs.root_changed", "fs.unreadable", "fs.permission"],
        ["test_safe_root.py::test_swap_race_file", "test_safe_root.py::test_swap_race_folder"],
    ),
    _case(
        "N10-05",
        "The root renamed and replaced by a link (or another folder) after the grant",
        ["fs.root_changed"],
        ["test_safe_root.py::test_root_replaced", "test_safe_root.py::test_expect_identity"],
    ),
    _case(
        "N10-06",
        "Hard link inside the root to a file outside (same file system)",
        ["fs.hard_link"],
        ["test_attack_catalog.py::test_link_attacks_are_refused", "test_safe_root.py::test_hard_links"],
    ),
    _case(
        "N10-07",
        "Bind mount or another file system below the root",
        ["fs.cross_device", "fs.link"],
        [
            "test_safe_root.py::test_linux_mounts_below_the_root",
            "test_safe_root.py::test_macos_disk_image_below_the_root",
            "test_windows.py::test_junction_to_another_volume",
        ],
    ),
    _case(
        "N10-08",
        "Linux magic links: /proc/self/root/..., /proc/<pid>/fd/N",
        ["fs.link", "fs.absolute_path", "fs.protected_root"],
        ["test_safe_root.py::test_linux_magic_links"],
        frozenset({"linux"}),
    ),
    _case(
        "N10-09",
        "FIFO, Unix socket, /dev/zero, /dev/random in the root: no hang, refused",
        ["fs.not_regular_file", "fs.link", "fs.invalid_path", "fs.protected_root"],
        ["test_safe_root.py::test_special_files", "test_attack_catalog.py::test_name_attacks_are_refused"],
    ),
    _case(
        "N10-10",
        "Very large or sparse 100 GB file",
        ["fs.too_large"],
        ["test_safe_root.py::test_large_and_sparse_files"],
    ),
    _case(
        "N10-11",
        "Case (~/LIBRARY/Keychains) and NFD spellings of blocked names; case-insensitive file systems",
        ["fs.blocked", "fs.excluded", "fs.not_found"],
        [
            "test_attack_catalog.py::test_name_attacks_are_refused",
            "test_safe_root.py::test_blocked_spellings",
            "test_safe_root.py::test_protected_folder_any_spelling",
            "test_safe_root.py::test_patterns_fold_case",
        ],
    ),
    _case(
        "N10-12",
        "macOS firmlink detour (/System/Volumes/Data); a Finder alias is an ordinary file",
        ["fs.protected_root"],
        ["test_safe_root.py::test_macos_firmlink_root", "test_safe_root.py::test_finder_alias_is_a_file"],
        frozenset({"darwin"}),
    ),
    _case(
        "N10-13",
        "Windows junction and folder symbolic link in the root; mount point to another drive",
        ["fs.link"],
        [
            "test_attack_catalog.py::test_link_attacks_are_refused",
            "test_windows.py::test_junction_and_folder_link",
            "test_windows.py::test_junction_to_another_volume",
        ],
        frozenset({"win32"}),
    ),
    _case(
        "N10-14",
        "Windows 8.3 short names PROGRA~1, SECRET~1.TXT on blocked names",
        ["fs.invalid_path"],
        ["test_attack_catalog.py::test_name_attacks_are_refused", "test_windows.py::test_short_names"],
    ),
    _case(
        "N10-15",
        "Alternate data streams: file.txt:secret, file.txt::$DATA, folder:$I30:$INDEX_ALLOCATION",
        ["fs.invalid_path"],
        [
            "test_attack_catalog.py::test_name_attacks_are_refused",
            "test_windows.py::test_alternate_data_stream",
        ],
    ),
    _case(
        "N10-16",
        "\\\\?\\C:\\, \\\\.\\PhysicalDrive0, \\\\?\\GLOBALROOT, \\??\\, \\\\localhost\\c$, \\\\?\\UNC\\",
        ["fs.absolute_path", "fs.invalid_path", "fs.network_path"],
        [
            "test_attack_catalog.py::test_name_attacks_are_refused",
            "test_safe_root.py::test_device_and_unc_roots",
        ],
    ),
    _case(
        "N10-17",
        "Reserved device names CON, NUL.txt, COM1; trailing dot and space",
        ["fs.invalid_path"],
        ["test_attack_catalog.py::test_name_attacks_are_refused", "test_windows.py::test_trailing_dot_file"],
    ),
    _case(
        "N10-18",
        "Case, Turkish dotted I and sharp s (casefold) against the blocklist",
        ["fs.blocked", "fs.not_found"],
        [
            "test_attack_catalog.py::test_name_attacks_are_refused",
            "test_safe_root.py::test_blocked_spellings",
        ],
    ),
    _case(
        "N10-19",
        "Fullwidth solidus, division slash, NUL byte; right-to-left override (readable, shown escaped)",
        ["fs.invalid_path"],
        ["test_attack_catalog.py::test_name_attacks_are_refused", "test_safe_root.py::test_bidi_names"],
    ),
    _case(
        "N10-20",
        "Path longer than 4096 bytes, component longer than 255 bytes",
        ["fs.invalid_path"],
        ["test_attack_catalog.py::test_name_attacks_are_refused"],
    ),
    _case(
        "N10-21",
        "Grant revoked during a listing or read: the next call is refused, a running read stops",
        ["fs.revoked"],
        ["test_safe_root.py::test_revoke"],
    ),
    _case(
        "N10-22",
        "Product data folder, vault and keychain files inside a too wide root (~)",
        ["fs.blocked"],
        [
            "test_safe_root.py::test_protected_inside_a_wide_root",
            "test_attack_catalog.py::test_name_attacks_are_refused",
        ],
    ),
    _case(
        "N10-23",
        "Root on /, C:\\, ~, a system folder or the data folder: refused when created",
        ["fs.protected_root"],
        ["test_safe_root.py::test_roots_that_cannot_be_granted"],
    ),
    _case(
        "W06-1",
        "Hard links refused (st_nlink > 1, Windows nNumberOfLinks), opt-in per root",
        ["fs.hard_link"],
        ["test_safe_root.py::test_hard_links"],
    ),
    _case(
        "W06-2",
        "NFC normalization: NFC input, NFD names on disk, ambiguous names",
        ["fs.ambiguous_name"],
        ["test_safe_root.py::test_nfc_and_nfd_names"],
    ),
    _case(
        "W06-3",
        "Create and rename without replacing or following (O_CREAT|O_EXCL, renameat2 RENAME_NOREPLACE)",
        ["fs.exists", "fs.link", "fs.read_only", "fs.hard_link"],
        ["test_safe_root.py::test_create_file", "test_safe_root.py::test_rename"],
    ),
    _case(
        "W06-4",
        "Network drives and UNC shares (also of this computer); OneDrive cloud placeholders",
        ["fs.network_path", "fs.cloud_file"],
        [
            "test_safe_root.py::test_device_and_unc_roots",
            "test_safe_root.py::test_cloud_placeholders",
            "test_windows.py::test_network_drive",
        ],
    ),
    _case(
        "W06-5",
        "Windows without administrator rights: junction case and reading still work and refuse",
        ["fs.link"],
        ["test_windows.py::test_without_administrator_rights"],
        frozenset({"win32"}),
    ),
)
"""Every case, in catalog order."""


#: ``(case id, relative path, codes that refuse it)``: attacks that need nothing but the layout's names. Each
_NAME_ATTACKS: list[tuple[str, str | bytes, list[str]]] = [
    ("N10-01", "../outside/secret.txt", ["fs.parent_reference"]),
    ("N10-01", "sub/../../outside/secret.txt", ["fs.parent_reference"]),
    ("N10-01", "..", ["fs.parent_reference"]),
    ("N10-01", "%2e%2e%2foutside%2fsecret.txt", ["fs.not_found"]),
    ("N10-01", "..%2f..%2foutside%2fsecret.txt", ["fs.not_found"]),
    ("N10-01", b"\xc0\xae\xc0\xae/outside/secret.txt", ["fs.invalid_path"]),
    ("N10-01", b"%c0%ae%c0%ae/outside/secret.txt", ["fs.not_found"]),
    ("N10-02", "/etc/passwd", ["fs.absolute_path"]),
    ("N10-02", "C:\\Windows\\win.ini", ["fs.absolute_path"]),
    ("N10-02", "C:/Windows/win.ini", ["fs.absolute_path"]),
    ("N10-02", "~/.ssh/id_rsa", ["fs.absolute_path"]),
    ("N10-02", "~", ["fs.absolute_path"]),
    ("N10-09", "NUL", ["fs.invalid_path"]),
    ("N10-11", ".ssh/id_rsa", ["fs.blocked"]),
    ("N10-11", ".SSH/id_rsa", ["fs.blocked"]),
    ("N10-11", "Library/Keychains/login.keychain-db", ["fs.blocked"]),
    ("N10-11", "LIBRARY/KEYCHAINS/login.keychain-db", ["fs.blocked"]),
    ("N10-11", "library/keychains", ["fs.blocked"]),
    ("N10-14", "PROGRA~1", ["fs.invalid_path"]),
    ("N10-14", "SECRET~1.TXT", ["fs.invalid_path"]),
    ("N10-14", "sub/DEEP~1.TXT", ["fs.invalid_path"]),
    ("N10-15", "inside.txt:secret", ["fs.invalid_path"]),
    ("N10-15", "inside.txt::$DATA", ["fs.invalid_path"]),
    ("N10-15", "sub:$I30:$INDEX_ALLOCATION", ["fs.invalid_path"]),
    ("N10-16", "\\\\?\\C:\\Windows\\win.ini", ["fs.absolute_path"]),
    ("N10-16", "\\\\.\\PhysicalDrive0", ["fs.absolute_path"]),
    ("N10-16", "\\\\?\\GLOBALROOT\\Device\\HarddiskVolume1", ["fs.absolute_path"]),
    ("N10-16", "\\??\\C:\\Windows", ["fs.absolute_path"]),
    ("N10-16", "\\\\localhost\\c$\\Windows\\win.ini", ["fs.absolute_path"]),
    ("N10-16", "\\\\?\\UNC\\localhost\\c$", ["fs.absolute_path"]),
    ("N10-16", "//localhost/c$/Windows", ["fs.absolute_path"]),
    ("N10-16", "sub\\..\\..\\outside\\secret.txt", ["fs.invalid_path"]),
    ("N10-17", "CON", ["fs.invalid_path"]),
    ("N10-17", "NUL.txt", ["fs.invalid_path"]),
    ("N10-17", "COM1", ["fs.invalid_path"]),
    ("N10-17", "lpt¹.log", ["fs.invalid_path"]),
    ("N10-17", "sub/aux", ["fs.invalid_path"]),
    ("N10-17", "inside.txt.", ["fs.invalid_path"]),
    ("N10-17", "inside.txt ", ["fs.invalid_path"]),
    ("N10-18", ".\u017fsh/id_rsa", ["fs.blocked"]),  # LATIN SMALL LETTER LONG S folds to 's'
    (
        "N10-18",
        "LİBRARY/Keychains",
        ["fs.not_found", "fs.blocked"],
    ),  # dotted capital I does not fold to i
    ("N10-19", "sub\uff0fdeep.txt", ["fs.invalid_path"]),
    ("N10-19", "sub\u2215deep.txt", ["fs.invalid_path"]),
    ("N10-19", "sub\u2044deep.txt", ["fs.invalid_path"]),
    ("N10-19", "inside.txt\x00.pdf", ["fs.invalid_path"]),
    ("N10-19", "sub/\x1b[2Jdeep.txt", ["fs.invalid_path"]),
    ("N10-20", "a" * 256, ["fs.invalid_path"]),
    ("N10-20", "/".join(["a"] * 2049), ["fs.invalid_path"]),
    ("N10-20", "", ["fs.invalid_path"]),
    ("N10-20", "sub//deep.txt", ["fs.invalid_path"]),
    ("N10-20", "./inside.txt", ["fs.invalid_path"]),
    ("N10-22", "_netrc", ["fs.blocked"]),
    ("N10-22", "AppData/Roaming/Microsoft/Credentials/x", ["fs.blocked"]),
    ("N10-22", "sub/.ssh/known_hosts", ["fs.blocked"]),
]
#: must be refused; none may return :data:`CANARY` or :data:`BLOCKED_CANARY`.
NAME_ATTACKS: tuple[tuple[str, str | bytes, frozenset[str]], ...] = tuple(
    (case, path, frozenset(codes)) for case, path, codes in _NAME_ATTACKS
)


@dataclass
class Layout:
    """The files of an attack: ``root`` (granted), ``outside`` (holds :data:`CANARY`) and what could be made.

    ``created`` names the links that exist (``link_file``, ``link_dir``, ``link_rel``, ``chain``,
    ``link_inside``, ``hard``, ``junction``); creating symbolic links needs a privilege or developer mode on
    Windows, a hard link the same file system.
    """

    base: Path
    root: Path
    outside: Path
    secret: Path
    created: dict[str, bool] = field(default_factory=dict)


def make_junction(link: Path, target: Path) -> bool:
    """Make a Windows directory junction (``mklink /J``, no privilege needed); ``False`` if not possible."""
    if sys.platform != "win32":
        return False
    done = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=False, timeout=60
    )
    return done.returncode == 0


def _try(action: Callable[[], object]) -> bool:
    try:
        action()
    except (OSError, NotImplementedError):
        return False
    return True


def build_layout(base: Path) -> Layout:
    """Create the attack layout below ``base`` (an empty folder) and return it."""
    root, outside = base / "root", base / "outside"
    (root / "sub").mkdir(parents=True)
    (outside / "dir").mkdir(parents=True)
    secret = outside / "secret.txt"
    secret.write_bytes(CANARY)
    (outside / "dir" / "secret.txt").write_bytes(CANARY)
    (root / "inside.txt").write_bytes(b"inside")
    (root / "sub" / "deep.txt").write_bytes(b"deep")
    for blocked in (root / ".ssh" / "id_rsa", root / "Library" / "Keychains" / "login.keychain-db"):
        blocked.parent.mkdir(parents=True)
        blocked.write_bytes(BLOCKED_CANARY)
    (root / "_netrc").write_bytes(BLOCKED_CANARY)
    (root / "sub" / ".ssh").mkdir()
    (root / "sub" / ".ssh" / "known_hosts").write_bytes(BLOCKED_CANARY)
    created = {
        "link_file": _try(lambda: os.symlink(secret, root / "link_file")),
        "link_dir": _try(lambda: os.symlink(outside, root / "link_dir", target_is_directory=True)),
        "link_rel": _try(lambda: os.symlink(os.path.join("..", "outside", "secret.txt"), root / "link_rel")),
        "chain": _try(lambda: os.symlink("chain2", root / "chain1"))
        and _try(
            lambda: os.symlink(
                os.path.join("..", "outside", "dir"), root / "chain2", target_is_directory=True
            )
        ),
        "link_inside": _try(lambda: os.symlink(os.path.join("sub", "deep.txt"), root / "link_inside")),
        "hard": _try(lambda: os.link(secret, root / "hard")),
        "junction": make_junction(root / "junction", outside),
    }
    return Layout(base=base, root=root, outside=outside, secret=secret, created=created)


def link_attacks(layout: Layout) -> list[tuple[str, str, frozenset[str]]]:
    """The link attacks the layout could create: ``(case id, relative path, codes)``."""
    link = frozenset({"fs.link"})
    candidates = [
        ("link_file", "N10-03", "link_file", link),
        ("link_dir", "N10-03", "link_dir/secret.txt", link),
        ("link_dir", "N10-03", "link_dir/dir/secret.txt", link),
        ("link_rel", "N10-03", "link_rel", link),
        ("chain", "N10-03", "chain1/secret.txt", link),
        ("hard", "N10-06", "hard", frozenset({"fs.hard_link"})),
        ("junction", "N10-13", "junction/secret.txt", link),
        ("junction", "N10-13", "junction/dir/secret.txt", link),
    ]
    return [(case, path, codes) for key, case, path, codes in candidates if layout.created.get(key)]


@dataclass(frozen=True)
class Finding:
    """An attack the reader did not refuse as expected."""

    case: str
    path: str
    problem: str


def _default_code(error: BaseException) -> str | None:
    code = getattr(error, "code", None)
    return code if isinstance(code, str) else None


def check_reader(
    read: Callable[[str | bytes], bytes],
    layout: Layout,
    *,
    code_of: Callable[[BaseException], str | None] | None = None,
    extra: Iterable[tuple[str, str | bytes, frozenset[str]]] = (),
) -> list[Finding]:
    """Run every name and link attack against ``read`` (a relative path in, the bytes out).

    An attack passes when ``read`` raises. With ``code_of`` (an exception to its code, e.g. the ``code`` of an
    :class:`orkexa.fs.FsError` or of a product's tool error) the code must be one of the attack's codes too.
    ``extra`` adds attacks of the caller. Returns the findings; an empty list means every attack was refused.
    """
    findings: list[Finding] = []
    attacks = [*NAME_ATTACKS, *link_attacks(layout), *extra]
    absolute = str(layout.secret)
    attacks.append(("N10-02", absolute, frozenset({"fs.absolute_path"})))
    for case, path, codes in attacks:
        shown = path if isinstance(path, str) else repr(path)
        try:
            data = read(path)
        except Exception as exc:  # every refusal counts; the code is checked when asked for
            if code_of is not None:
                code = code_of(exc)
                if code not in codes:
                    findings.append(
                        Finding(case, shown, f"refused with {code!r}, expected one of {sorted(codes)}")
                    )
            continue
        problem = "read the canary outside the root" if CANARY in data else "was not refused"
        if BLOCKED_CANARY in data:
            problem = "read a blocked file"
        findings.append(Finding(case, shown, problem))
    return findings


def swap_race(
    read: Callable[[str], bytes], layout: Layout, *, iterations: int = 10_000, folder: bool = False
) -> list[Finding]:
    """Swap ``root/race`` between a regular file (folder) and a link to the outside while ``read`` runs.

    A second thread renames the entries back and forth as fast as it can; ``read`` is called ``iterations``
    times on ``race`` (``race/secret.txt`` with ``folder``). Any read that returns :data:`CANARY` is a
    finding. The link is a symbolic link, or a junction on Windows for a folder. Returns ``[]`` when the link
    cannot be made here (check :attr:`Layout.created` first).
    """
    root = layout.root
    real, link = root / "race_real", root / "race_link"
    if folder:
        real.mkdir()
        (real / "secret.txt").write_bytes(b"inside")
        made = _try(lambda: os.symlink(layout.outside, link, target_is_directory=True)) or make_junction(
            link, layout.outside
        )
        target = "race/secret.txt"
    else:
        real.write_bytes(b"inside")
        made = _try(lambda: os.symlink(layout.secret, link))
        target = "race"
    if not made:
        return []
    os.rename(real, root / "race")
    stop = threading.Event()

    def retry(old: Path, new: Path) -> None:
        while True:
            try:
                os.rename(old, new)
                return
            except OSError:  # a sharing violation on Windows while the reader holds a handle
                if stop.is_set() and not os.path.lexists(old):
                    return

    def swapper() -> None:
        occupant, spare = real, link  # "race" holds the occupant; the spare waits under its own name
        while not stop.is_set():
            try:
                os.rename(root / "race", root / "race_tmp")
            except OSError:
                continue
            try:
                os.rename(spare, root / "race")
            except OSError:
                retry(root / "race_tmp", root / "race")
                continue
            retry(root / "race_tmp", occupant)
            occupant, spare = spare, occupant

    thread = threading.Thread(target=swapper, daemon=True)
    thread.start()
    findings: list[Finding] = []
    try:
        for _ in range(iterations):
            try:
                data = read(target)
            except Exception:
                continue
            if CANARY in data:
                findings.append(Finding("N10-04", target, "read the canary outside the root during a swap"))
                break
    finally:
        stop.set()
        thread.join(timeout=30)
    return findings
