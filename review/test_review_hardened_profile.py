"""Review only: does a narrower Seatbelt profile close the findings of test_review_macos_sandbox.py?

Runs a patched *copy* of _macos_confine.py (the product code is not changed) whose profile replaces the
global ``(allow sysctl-read)`` with an allowlist and adds explicit denials for SysV IPC and for process
information about other processes. It checks that ordinary tool code still runs and which probes are closed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pytest

from orkexa.dynamic import MacOSSandbox
from orkexa.dynamic import macos as macos_module

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS")
OUT = Path(os.environ.get("REVIEW_OUT", "review-out"))
TOKEN = f"HARDENSECRET{os.getpid()}"

SYSCTL_ALLOW = '''        "(allow sysctl-read (sysctl-name \\"hw.ncpu\\") (sysctl-name \\"hw.activecpu\\") (sysctl-name \\"hw.memsize\\")"
        " (sysctl-name \\"hw.pagesize\\") (sysctl-name \\"hw.cachelinesize\\") (sysctl-name \\"kern.osversion\\")"
        " (sysctl-name \\"kern.osrelease\\") (sysctl-name \\"kern.ostype\\") (sysctl-name \\"kern.version\\")"
        " (sysctl-name \\"kern.argmax\\") (sysctl-name \\"kern.maxfilesperproc\\") (sysctl-name \\"kern.usrstack64\\")"
        " (sysctl-name \\"sysctl.proc_translated\\") (sysctl-name \\"sysctl.proc_cputype\\")"
        " (sysctl-name \\"machdep.cpu.brand_string\\") (sysctl-name-prefix \\"hw.optional.\\")"
        " (sysctl-name-prefix \\"hw.perflevel\\") (sysctl-name-prefix \\"hw.logicalcpu\\")"
        " (sysctl-name-prefix \\"hw.physicalcpu\\") (sysctl-name-prefix \\"hw.l1\\") (sysctl-name-prefix \\"hw.l2\\"))",
        "(deny ipc-sysv*)",
        "(deny process-info* (target others))",'''


@pytest.fixture(scope="module")
def hardened(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = Path(macos_module._CONFINE).read_text()
    assert '        "(allow sysctl-read)",' in source
    patched = source.replace('        "(allow sysctl-read)",', SYSCTL_ALLOW)
    path = tmp_path_factory.mktemp("hardened") / "_macos_confine.py"
    path.write_text(patched)
    return path


PROBES = r'''
import ctypes, hashlib, json, os, platform, time
lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
c = json.loads(context)
def attempt(action):
    try:
        return "ALLOWED: " + repr(action())[:200]
    except BaseException as exc:
        return "refused: " + type(exc).__name__ + ": " + str(exc)[:100]
def procargs(p):
    mib = (ctypes.c_int * 3)(1, 49, p)
    size = ctypes.c_size_t(1 << 18); buf = ctypes.create_string_buffer(size.value)
    if lib.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) != 0: raise OSError(ctypes.get_errno(), "sysctl")
    return ("SECRET VISIBLE" if c["token"].encode() in buf.raw[: size.value] else "no secret", size.value)
def pidpath(p):
    buf = ctypes.create_string_buffer(4096)
    if lib.proc_pidpath(p, buf, 4096) <= 0: raise OSError(ctypes.get_errno(), "proc_pidpath")
    return buf.value
def listpids():
    buf = (ctypes.c_int * 8192)(); n = lib.proc_listallpids(buf, ctypes.sizeof(buf))
    if n <= 0: raise OSError(ctypes.get_errno(), "proc_listallpids")
    return n
def kern_proc_all():
    mib = (ctypes.c_int * 3)(1, 14, 0); size = ctypes.c_size_t(0)
    if lib.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0: raise OSError(ctypes.get_errno(), "sysctl")
    return size.value
def sysv(fn, *args):
    i = fn(*args)
    if i < 0: raise OSError(ctypes.get_errno(), "sysv")
    return i
def hostname():
    return __import__("socket").gethostname()
r = {
    "procargs other": attempt(lambda: procargs(c["pid"])),
    "procargs parent": attempt(lambda: procargs(os.getppid())),
    "proc_pidpath other": attempt(lambda: pidpath(c["pid"])),
    "proc_listallpids": attempt(listpids),
    "KERN_PROC_ALL": attempt(kern_proc_all),
    "sysv shmget": attempt(lambda: sysv(lib.shmget, 0, 4096, 0o1000 | 0o600)),
    "sysv semget": attempt(lambda: sysv(lib.semget, 0, 1, 0o1000 | 0o600)),
    "sysv msgget": attempt(lambda: sysv(lib.msgget, 0, 0o1000 | 0o600)),
    "kern.hostname": attempt(hostname),
    "ordinary: cpu_count": attempt(os.cpu_count),
    "ordinary: platform": attempt(platform.platform),
    "ordinary: hashlib": attempt(lambda: hashlib.sha256(b"x").hexdigest()[:8]),
    "ordinary: time": attempt(lambda: time.strftime("%Y")),
    "ordinary: getpid": attempt(os.getpid),
}
return json.dumps(r)
'''


def _tool() -> str:
    lines = "\n".join("    " + line if line else "" for line in PROBES.strip("\n").splitlines())
    return ("def probe(context: str) -> str:\n"
            '    """Probe.\n\n    Args:\n        context: Input.\n\n    Returns:\n        Text.\n    """\n'
            f"{lines}\n")


def test_hardened_profile(hardened: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                              env={"PATH": "/usr/bin:/bin", "HARDEN_SECRET": TOKEN})
    time.sleep(0.5)
    context = json.dumps({"pid": helper.pid, "token": TOKEN})
    out: dict[str, Any] = {}
    try:
        out["current"] = json.loads(MacOSSandbox(timeout=20).call(_tool(), "probe", {"context": context}))
        monkeypatch.setattr(macos_module, "_CONFINE", hardened)
        try:
            out["hardened"] = json.loads(MacOSSandbox(timeout=20).call(_tool(), "probe", {"context": context}))
        except Exception as exc:  # the profile may break the runner; that is a result too
            out["hardened"] = {"error": f"{type(exc).__name__}: {exc}"[:600]}
    finally:
        helper.kill()
        helper.wait()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hardened.json").write_text(json.dumps(out, indent=1))
    print("\n=== hardened ===\n" + json.dumps(out, indent=1))
