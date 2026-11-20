"""SafeRoot on Windows: junctions, mount points, 8.3 names, streams, trailing dots, network drives, no admin.

Every test here needs NTFS and the Win32 API, so the module is skipped on Linux and macOS (the name rules
behind these cases run everywhere in test_attack_catalog.py).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import string
import subprocess
import sys
import textwrap

import pytest

from orkexa.fs import FsError, SafeRoot
from orkexa.testing import pathguard
from orkexa.testing.pathguard import CANARY

WINDOWS = sys.platform == "win32"
pytestmark = pytest.mark.skipif(not WINDOWS, reason="needs NTFS and the Win32 API (Windows only)")


def _code(call: object) -> str:
    assert callable(call)
    with pytest.raises(FsError) as raised:
        call()
    return raised.value.code


@pytest.fixture
def layout(tmp_path: Path) -> pathguard.Layout:
    return pathguard.build_layout(tmp_path)


def test_junction_and_folder_link(layout: pathguard.Layout) -> None:
    """N10-13: a junction and a folder symbolic link inside the root are listed as links, never followed."""
    assert layout.created["junction"] and layout.created["link_dir"]
    root = SafeRoot(layout.root)
    for path in ("junction/secret.txt", "junction/dir/secret.txt", "link_dir/secret.txt"):
        assert _code(lambda path=path: root.read_bytes(path)) == "fs.link", path
    assert _code(lambda: root.list("junction")) == "fs.link"
    kinds = {entry.name: entry.kind for entry in root.list("")}
    assert kinds["junction"] == "link" and kinds["link_dir"] == "link"


def _drives() -> list[str]:
    import ctypes

    mask = ctypes.windll.kernel32.GetLogicalDrives()  # type: ignore[attr-defined,unused-ignore]
    fixed = []
    for index, letter in enumerate(string.ascii_uppercase):
        if mask & (1 << index):
            root = f"{letter}:\\"
            if ctypes.windll.kernel32.GetDriveTypeW(root) == 3:  # type: ignore[attr-defined,unused-ignore]
                fixed.append(root)
    return fixed


def test_junction_to_another_volume(layout: pathguard.Layout) -> None:
    """N10-07, N10-13: a mount-point reparse point to another drive (or this drive's root) is refused."""
    own = os.path.splitdrive(str(layout.root))[0].upper() + "\\"
    others = [drive for drive in _drives() if drive != own]
    target = Path(others[0] if others else own)
    assert pathguard.make_junction(layout.root / "volume", target)
    root = SafeRoot(layout.root)
    assert _code(lambda: root.list("volume")) == "fs.link"
    assert _code(lambda: root.read_bytes("volume/Windows/win.ini")) == "fs.link"


def _short_name(path: Path) -> str:
    import ctypes

    buffer = ctypes.create_unicode_buffer(1024)
    ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, 1024)  # type: ignore[attr-defined,unused-ignore]
    return buffer.value


def test_short_names(tmp_path: Path) -> None:
    """N10-14: an 8.3 name of a blocked folder is refused; a root given by its 8.3 name is canonicalized."""
    folder = tmp_path / "LongRootFolderName"
    secrets = folder / "SecretsFolder"
    secrets.mkdir(parents=True)
    (secrets / "key.txt").write_bytes(CANARY)
    root = SafeRoot(folder, blocked=["**/SecretsFolder"])
    short = os.path.basename(_short_name(secrets))
    for name in {short, "SECRET~1", "SECRET~2"}:
        assert _code(lambda name=name: root.read_bytes(f"{name}/key.txt")) in (
            "fs.invalid_path",
            "fs.blocked",
        )
    assert _code(lambda: root.read_bytes("SECRET~1/key.txt")) == "fs.invalid_path"
    short_root = _short_name(folder)
    assert SafeRoot(short_root, blocked=[]).path.casefold() == SafeRoot(folder, blocked=[]).path.casefold()
    assert "~" not in SafeRoot(short_root, blocked=[]).path.rsplit("\\", 1)[-1]


def test_alternate_data_stream(tmp_path: Path) -> None:
    """N10-15: a real alternate data stream is never reachable by name, and not listed."""
    folder = tmp_path / "root"
    folder.mkdir()
    (folder / "file.txt").write_bytes(b"main")
    with open(str(folder / "file.txt") + ":secret", "wb") as stream:
        stream.write(CANARY)
    root = SafeRoot(folder)
    assert root.read_bytes("file.txt") == b"main"
    for path in ("file.txt:secret", "file.txt:secret:$DATA", "file.txt::$DATA"):
        assert _code(lambda path=path: root.read_bytes(path)) == "fs.invalid_path"
    assert [entry.name for entry in root.list("")] == ["file.txt"]


def test_trailing_dot_file(tmp_path: Path) -> None:
    """N10-17: a name with a trailing dot (made through \\\\?\\) is neither reachable nor aliased."""
    folder = tmp_path / "root"
    folder.mkdir()
    with open("\\\\?\\" + str(folder / "secret.txt."), "wb") as handle:
        handle.write(CANARY)
    root = SafeRoot(folder)
    assert _code(lambda: root.read_bytes("secret.txt.")) == "fs.invalid_path"
    assert (
        _code(lambda: root.read_bytes("secret.txt")) == "fs.not_found"
    )  # Win32 would strip the dot; \\?\ does not
    assert root.list("") == []


def test_network_drive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """W06-4: a mapped network drive (DRIVE_REMOTE) and a final path on a UNC share are refused."""
    from orkexa.fs import _windows

    monkeypatch.setattr(_windows, "_drive_type", lambda path: 4)
    assert _code(lambda: SafeRoot(tmp_path)) == "fs.network_path"
    monkeypatch.undo()
    monkeypatch.setattr(_windows, "_final", lambda handle: "\\\\server\\share\\folder")
    assert _code(lambda: SafeRoot(tmp_path)) == "fs.network_path"
    monkeypatch.undo()
    host = os.environ.get("COMPUTERNAME", "localhost")
    drive = os.path.splitdrive(str(tmp_path))[0][0].lower()
    admin_share = f"\\\\{host}\\{drive}$" + os.path.splitdrive(str(tmp_path))[1]
    assert _code(lambda: SafeRoot(admin_share)) == "fs.network_path"


_CHILD = textwrap.dedent(
    """
    import ctypes, json, os, subprocess, sys, traceback
    from pathlib import Path

    base, out = Path(sys.argv[1]), Path(sys.argv[2])


    def main():
        from orkexa.fs import FsError, SafeRoot

        report = {"admin": bool(ctypes.windll.shell32.IsUserAnAdmin())}
        try:
            os.symlink(base / "outside", base / "root" / "symlink", target_is_directory=True)
            report["symlink"] = True
        except OSError as exc:
            report["symlink"] = exc.winerror
        junction = [str(base / "root" / "junction"), str(base / "outside")]
        done = subprocess.run(["cmd", "/c", "mklink", "/J", *junction], capture_output=True)
        report["junction_made"] = done.returncode == 0
        root = SafeRoot(base / "root")
        report["inside"] = root.read_bytes("inside.txt").decode()
        for name in ("junction/secret.txt", "symlink/secret.txt"):
            try:
                report[name] = root.read_bytes(name).decode()
            except FsError as exc:
                report[name] = exc.code
        return report


    try:
        out.write_text(json.dumps(main()))
    except BaseException:
        out.write_text(json.dumps({"error": traceback.format_exc()}))
        sys.exit(1)
    """
)


def _run_restricted(command: str, timeout_ms: int = 120_000) -> int:
    """Run ``command`` with this process's token minus administrator rights (Administrators deny-only, every
    privilege but SeChangeNotify removed), as a UAC-filtered standard user would. ``CreateProcessAsUserW``
    needs
    no extra privilege for a restricted copy of the caller's own token."""
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class SidAndAttributes(ctypes.Structure):
        _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]

    class StartupInfo(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("lpReserved", wintypes.LPWSTR),
            ("lpDesktop", wintypes.LPWSTR),
            ("lpTitle", wintypes.LPWSTR),
            ("dwX", wintypes.DWORD),
            ("dwY", wintypes.DWORD),
            ("dwXSize", wintypes.DWORD),
            ("dwYSize", wintypes.DWORD),
            ("dwXCountChars", wintypes.DWORD),
            ("dwYCountChars", wintypes.DWORD),
            ("dwFillAttribute", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("wShowWindow", wintypes.WORD),
            ("cbReserved2", wintypes.WORD),
            ("lpReserved2", ctypes.c_void_p),
            ("hStdInput", wintypes.HANDLE),
            ("hStdOutput", wintypes.HANDLE),
            ("hStdError", wintypes.HANDLE),
        ]

    class ProcessInformation(ctypes.Structure):
        _fields_ = [
            ("hProcess", wintypes.HANDLE),
            ("hThread", wintypes.HANDLE),
            ("dwProcessId", wintypes.DWORD),
            ("dwThreadId", wintypes.DWORD),
        ]

    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi32.CreateWellKnownSid.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.CreateRestrictedToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(SidAndAttributes),
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.CreateProcessAsUserW.argtypes = [
        wintypes.HANDLE,
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.POINTER(StartupInfo),
        ctypes.POINTER(ProcessInformation),
    ]
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    token = wintypes.HANDLE()
    access = (
        0x0001 | 0x0002 | 0x0008 | 0x0080 | 0x0100
    )  # assign primary, duplicate, query, adjust default, session
    assert advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), access, ctypes.byref(token)), (
        ctypes.get_last_error()
    )
    sid = ctypes.create_string_buffer(68)
    size = wintypes.DWORD(68)
    assert advapi32.CreateWellKnownSid(26, None, sid, ctypes.byref(size)), (
        ctypes.get_last_error()
    )  # Administrators
    disable = SidAndAttributes(ctypes.cast(sid, ctypes.c_void_p), 0)
    restricted = wintypes.HANDLE()
    assert advapi32.CreateRestrictedToken(
        token, 0x1, 1, ctypes.byref(disable), 0, None, 0, None, ctypes.byref(restricted)
    ), ctypes.get_last_error()  # DISABLE_MAX_PRIVILEGE
    startup = StartupInfo()
    startup.cb = ctypes.sizeof(startup)
    info = ProcessInformation()
    line = ctypes.create_unicode_buffer(command)
    created = advapi32.CreateProcessAsUserW(
        restricted,
        None,
        line,
        None,
        None,
        False,
        0x08000000,
        None,
        None,
        ctypes.byref(startup),
        ctypes.byref(info),
    )
    assert created, f"CreateProcessAsUserW failed: {ctypes.get_last_error()}"
    try:
        assert kernel32.WaitForSingleObject(info.hProcess, timeout_ms) == 0, (
            "the restricted child did not finish"
        )
        code = wintypes.DWORD()
        kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(code))
        return int(code.value)
    finally:
        for handle in (info.hProcess, info.hThread, restricted, token):
            kernel32.CloseHandle(handle)


def test_without_administrator_rights(tmp_path: Path) -> None:
    """W06-5: the guard needs no administrator rights; the junction case holds for a standard user too."""
    (tmp_path / "root").mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "root" / "inside.txt").write_bytes(b"inside")
    (tmp_path / "outside" / "secret.txt").write_bytes(CANARY)
    script = tmp_path / "child.py"
    script.write_text(_CHILD, encoding="utf-8")
    out = tmp_path / "report.json"
    command = subprocess.list2cmdline([sys.executable, str(script), str(tmp_path), str(out)])
    assert _run_restricted(command) == 0, out.read_text() if out.exists() else "no report"
    report = json.loads(out.read_text())
    assert report["admin"] is False
    assert report["junction_made"] is True
    assert report["inside"] == "inside"
    assert report["junction/secret.txt"] == "fs.link"
    if report["symlink"] is True:  # developer mode lets a standard user make symbolic links
        assert report["symlink/secret.txt"] == "fs.link"
    else:
        assert report["symlink"] == 1314  # ERROR_PRIVILEGE_NOT_HELD
        assert report["symlink/secret.txt"] == "fs.not_found"
